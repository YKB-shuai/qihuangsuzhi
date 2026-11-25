"""大批次向量化（优化版）：解决原 embedding.py 在海量数据下 Chroma 频繁 upsert 太慢的问题。

原理：原脚本每 64 段就 flush 一次（encode + Chroma upsert + 逐条 UPDATE），
3.7 万条数据下 upsert 调用上万次，HNSW 索引反复重建，实测只有约 3 段/秒。
本脚本攒够 512 段才 upsert 一次（调用数减少 8 倍），编码仍分批 64 段送模型，
状态回写用 executemany 批量，实测可提速数倍。

断点续跑：只处理 review_status=1 且 vector_status=0 的记录，
每批先写向量再批量回写状态，中断后重跑自动接着来。

用法（在项目根目录执行）：
    .venv\\Scripts\\python.exe -m scripts.fast_embed --limit 0
    .venv\\Scripts\\python.exe -m scripts.fast_embed --limit 500
    .venv\\Scripts\\python.exe -m scripts.fast_embed --flush-seg 1024
"""

from __future__ import annotations

# 必须最先导入 torch，避免 Windows 上 sentence_transformers 懒加载 DLL 冲突段错误
import torch  # noqa: F401

import argparse
import time

from com.tcm.cfg.settings import settings
from com.tcm.db.conn import executemany, query_all, query_one
from com.tcm.rag import embedding


ENCODE_BATCH = 64          # 单次送模型的段数（编码效率最优，实测 batch 64/128/256 吞吐一致）


def flush(buf: list[dict], pending_ids: set[int], model, col) -> None:
    """一批：分批编码 → 一次 upsert → 批量回写 vector_status。"""
    if not buf:
        return
    texts = [c["text"] for c in buf]
    vecs: list[list[float]] = []
    for i in range(0, len(texts), ENCODE_BATCH):
        part = texts[i:i + ENCODE_BATCH]
        vecs.extend(model.encode(part, batch_size=ENCODE_BATCH, normalize_embeddings=True).tolist())

    col.upsert(
        ids=[c["id"] for c in buf],
        embeddings=vecs,
        documents=texts,
        metadatas=[c["meta"] for c in buf],
    )
    executemany(
        "UPDATE knowledge SET vector_status = 1 WHERE id = %s",
        [(kid,) for kid in pending_ids],
    )


def run(limit: int, flush_seg: int) -> None:
    rows = embedding.fetch_pending(limit)
    if not rows:
        print("没有待向量化的记录（全部已入库）")
        return
    print(f"待向量化：{len(rows)} 条，flush 阈值 {flush_seg} 段，加载模型…", flush=True)
    model = embedding.get_model()
    col = embedding.get_collection()
    print(f"模型就绪，Chroma 当前 {col.count()} 个 chunk，开始编码…", flush=True)

    buf: list[dict] = []
    pending_ids: set[int] = set()
    done = 0
    t0 = time.time()
    seg_total = 0

    for item in rows:
        chunks = embedding.chunk_item(item)
        buf.extend(chunks)
        seg_total += len(chunks)
        pending_ids.add(item["id"])
        if len(buf) >= flush_seg:
            flush(buf, pending_ids, model, col)
            buf, pending_ids = [], set()
        done += 1
        if done % 50 == 0:
            _progress(done, len(rows), seg_total, t0)

    flush(buf, pending_ids, model, col)
    _progress(done, len(rows), seg_total, t0)
    print(f"完成。Chroma 当前共 {col.count()} 个 chunk", flush=True)


def _progress(done: int, total: int, segs: int, t0: float) -> None:
    used = time.time() - t0
    speed = done / used if used > 0 else 0
    left = (total - done) / speed if speed > 0 else 0
    print(f"  已完成 {done}/{total} 条，{segs} 段　用时 {used:.0f}s　"
          f"速度 {speed:.1f} 条/秒　预计剩余 {left/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="大批次向量化（优化版）")
    ap.add_argument("--limit", type=int, default=0, help="本次处理多少条，0 表示不限")
    ap.add_argument("--flush-seg", type=int, default=512, help="攒够多少段 upsert 一次，默认 512")
    args = ap.parse_args()
    run(args.limit, args.flush_seg)
