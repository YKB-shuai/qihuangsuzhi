"""智能体问答：LangGraph 两节点图 retrieve → generate（阶段 4.2 / 4.3 / 4.5）。

State 契约（开工前冻结的三个契约之一）：
    question: str          用户问题
    chunks : list[dict]    检索片段 [{source, fid, title, seg, text, score}]
    answer : str           带引用的回答

对外提供三个入口：
    ask(question)                —— 一次性返回（命令行/测试用）
    iter_answer(question)        —— 流式生成（('sources'|'token'|'done', payload)）
    build_graph()                —— LangGraph 图对象

用法：
    uv run python -m com.tcm.qa.graph "治痞块用什么方"
"""

from __future__ import annotations

import sys
import time

# ⚠️ 这两行导入顺序不能改，是踩过坑的：
# 1) sentence_transformers 是"懒加载"，真正 import torch 发生在第一次用模型时；
# 2) 若让 torch 在 langchain/langgraph 之后加载，Windows 上会 DLL 冲突
#    → 进程直接段错误崩溃，而且一个字都不打印（极难排查）。
# 所以这里显式提前加载 torch，保证它排在所有 langchain/langgraph 之前。
import torch  # noqa: F401

from com.tcm.cfg.settings import settings
from com.tcm.db.conn import execute
from com.tcm.qa import intent
from com.tcm.rag import retriever

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, StateGraph
from typing import TypedDict

TOP_K = 5
MIN_SCORE = 0.5        # 兼容旧逻辑：生成端最低分门槛（保留原值）
GOOD_SCORE = 0.62      # 质量判定阈值：实测相关问题 0.79+，无关问题 0.55-，取中间值
MAX_RETRIES = 2        # agent loop：检索质量不足时最多改写重试 2 轮（共检索 3 次）
REFUSE = "知识库中没有找到相关内容，无法回答。建议咨询专业医师。"

# 查询改写提示词：把口语化问题压成检索友好的中医关键词
REWRITE_PROMPT = (
    "你是中医药知识库的检索词优化助手。\n"
    "把用户问题改写成适合检索的关键词短语，规则：\n"
    "1. 只输出检索关键词，不要解释、不要标点、不要引号；\n"
    "2. 保留核心药材名、方剂名、症状、功效等实体词；\n"
    "3. 去掉\"请问\"\"是什么\"\"怎么\"等口语词，可适当补充常见的中医同义说法；\n"
    "4. 控制在 30 字以内。\n"
    "用户问题：{question}\n"
    "改写后的检索词："
)

SYS_PROMPT = (
    "你是「岐黄溯智」中医药知识助手。规则：\n"
    "1. 只依据下面的资料片段回答，禁止使用资料之外的知识；\n"
    "2. 回答中引用到的信息在句尾标注片段编号，如 [1]；\n"
    "3. 资料不足以回答时，明确说\"知识库中没有找到相关内容\"，不要编造；\n"
    "4. 涉及用药的内容提醒用户咨询医师；\n"
    "5. 若用户上传了图片并询问图中是什么，开头先明确说出识别出的药材/方剂名称，再介绍详情。"
)


class QAState(TypedDict):
    question: str
    intent_ok: bool      # 意图识别：是否为中医药相关问题
    search_query: str    # 当前检索词（改写重试时会更新）
    retries: int         # 已改写重试次数
    chunks: list
    answer: str


def build_llm() -> ChatDeepSeek:
    return ChatDeepSeek(
        api_key=settings.DEEPSEEK_API_KEY,
        api_base=settings.DEEPSEEK_BASE_URL,
        model=settings.DEEPSEEK_MODEL_CHAT,
        temperature=0.3,
    )


def build_prompt(chunks: list[dict], question: str, image_name: str = "") -> str:
    """把片段编成带编号的资料块，让模型能按 [n] 引用。

    image_name: 附图识别出的药材名；有值时提示模型直接介绍该药材，
    避免纠结"资料里没有图片描述"而答非所问。
    """
    lines = [
        f"[{i}]（{c['source']} fid={c['fid']} {c['title']}·{c['seg']}）{c['text']}"
        for i, c in enumerate(chunks, 1)
    ]
    prefix = ""
    if image_name:
        prefix = (
            f"用户上传的图片已由图像识别确认为「{image_name}」。\n"
            f"回答时第一句话必须先明确给出识别结论，例如：\"这张图片中的药材是{image_name}。\"\n"
            f"然后再依据资料介绍{image_name}的相关信息，不要讨论资料中是否含图片。\n\n"
        )
    return prefix + "资料片段：\n" + "\n".join(lines) + f"\n\n问题：{question}"


def quality_ok(chunks: list[dict]) -> bool:
    """检索质量判定：只看真实向量相似度，忽略被人为抬高的纯 BM25 分。

    retriever 里纯 bm25 命中的 score 被映射到 0.5~0.8（不代表真实相关性），
    因此只用 from=vec / from=both 的向量分判断质量，避免无关词误判为命中。
    """
    vec_scores = [c["score"] for c in chunks if c.get("from") != "bm25"]
    return bool(vec_scores) and max(vec_scores) >= GOOD_SCORE


def rewrite_query(question: str) -> str:
    """用 LLM 把问题改写成检索友好的关键词；失败则返回空串。"""
    try:
        llm = ChatDeepSeek(
            api_key=settings.DEEPSEEK_API_KEY,
            api_base=settings.DEEPSEEK_BASE_URL,
            model=settings.DEEPSEEK_MODEL_CHAT,
            temperature=0,
            max_tokens=60,
        )
        resp = llm.invoke([
            HumanMessage(REWRITE_PROMPT.format(question=question[:200]))
        ])
        text = (resp.content or "").strip().strip("\"'：:")
        # 去掉模型偶尔带的前缀
        for junk in ("检索词：", "改写后：", "关键词："):
            if text.startswith(junk):
                text = text[len(junk):].strip()
        return text[:60]
    except Exception:
        return ""


def retrieve_with_retry(question: str, on_retry=None) -> tuple[list[dict], int]:
    """agent loop：检索 → 质量不足则改写重试，最多 MAX_RETRIES 轮。

    返回 (最终chunks, 实际重试次数)。
    on_retry: 可选回调 fn(attempt, new_query)，每轮改写后调用（用于流式提示）。
    """
    q = question
    retries = 0
    chunks = retriever.search(q, TOP_K)
    while not quality_ok(chunks) and retries < MAX_RETRIES:
        new_q = rewrite_query(q)
        if not new_q or new_q == q:   # 改写无效，避免死循环
            break
        q = new_q
        retries += 1
        if on_retry:
            on_retry(retries, q)
        chunks = retriever.search(q, TOP_K)
    return chunks, retries


def node_classify(state: QAState) -> dict:
    """节点零：意图识别。非中医药问题直接打标，后续节点不再检索/生成。"""
    return {"intent_ok": intent.is_tcm_question(state["question"])}


def node_retrieve(state: QAState) -> dict:
    """节点一：向量召回。意图不符时跳过检索。用当前检索词（可能被改写过）。"""
    if not state.get("intent_ok", True):
        return {"chunks": []}
    return {"chunks": retriever.search(state.get("search_query") or state["question"], TOP_K)}


def node_rewrite(state: QAState) -> dict:
    """节点：检索质量不足时，用 LLM 改写检索词，重试次数 +1。"""
    new_q = rewrite_query(state.get("search_query") or state["question"])
    if not new_q or new_q == state.get("search_query"):
        # 改写无效：保持原查询但把次数用尽，避免死循环
        return {"retries": MAX_RETRIES}
    return {"search_query": new_q, "retries": state.get("retries", 0) + 1}


def route_after_retrieve(state: QAState) -> str:
    """条件边：质量达标 → 生成；不达标且还有重试额度 → 改写；否则 → 生成（拒答）。"""
    if not state.get("intent_ok", True):
        return "generate"
    if quality_ok(state.get("chunks", [])):
        return "generate"
    if state.get("retries", 0) < MAX_RETRIES:
        return "rewrite"
    return "generate"


def node_generate(state: QAState) -> dict:
    """节点二：拼提示词 → DeepSeek 生成带引用回答；意图不符或资料不足就拒答。"""
    if not state.get("intent_ok", True):
        return {"answer": intent.OUT_OF_SCOPE}
    chunks = state["chunks"]
    if not quality_ok(chunks):
        return {"answer": REFUSE}

    resp = build_llm().invoke([
        SystemMessage(SYS_PROMPT),
        HumanMessage(build_prompt(chunks, state["question"])),
    ])
    return {"answer": resp.content}


def build_graph():
    g = StateGraph(QAState)
    g.add_node("classify", node_classify)
    g.add_node("retrieve", node_retrieve)
    g.add_node("rewrite", node_rewrite)
    g.add_node("generate", node_generate)
    g.add_edge(START, "classify")
    g.add_edge("classify", "retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieve, {"rewrite": "rewrite", "generate": "generate"})
    g.add_edge("rewrite", "retrieve")      # 回路：改写后重新检索
    g.add_edge("generate", END)
    return g.compile()


def save_log(question: str, chunks: list[dict], answer: str, elapsed_ms: int) -> None:
    """4.5 问答日志落库：问题 / 命中数 / 来源 / 耗时。"""
    sources = ",".join(sorted({c["source"] for c in chunks}))
    top_score = max(c["score"] for c in chunks) if chunks else 0
    execute(
        "INSERT INTO qa_log (question, hit_count, sources, top_score, elapsed_ms, answer) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (question, len(chunks), sources, top_score, elapsed_ms, answer),
    )


def ask(question: str, image_b64: str | None = None) -> dict:
    """非流式：跑一遍图，返回 {question, chunks, answer}。"""
    t0 = time.time()
    q = question
    img_bytes = _decode_image(image_b64)
    if img_bytes:
        try:
            from com.tcm.qa import ocr
            vis = ocr.describe_for_question(img_bytes, question)
            name = vis.get("name", "")
            desc = vis.get("description", "")
            if name:
                q = f"{name}是什么？请说明它的性味归经和功效主治。{question}"
            elif desc:
                q = f"用户问题：{question}\n图片内容：{desc}"
        except Exception:
            pass
    result = build_graph().invoke({
        "question": q, "intent_ok": True, "search_query": q,
        "retries": 0, "chunks": [], "answer": "",
    })
    save_log(question, result["chunks"], result["answer"], int((time.time() - t0) * 1000))
    return result


def _decode_image(image_b64: str | None) -> bytes | None:
    """把前端传来的 base64（支持 data URL）解码为图片字节。"""
    if not image_b64:
        return None
    import base64

    raw = image_b64
    if "," in raw and raw.strip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        return base64.b64decode(raw)
    except Exception:
        return None


def iter_answer(question: str, image_b64: str | None = None):
    """4.3 流式：先吐引用片段，再逐 token 吐回答。

    产出 ('sources', chunks) → 若干 ('token', 文本) → ('done', None)
    用同步 generator，配合 FastAPI 的 def 路由 + StreamingResponse。
    附图时：先用视觉模型描述图片内容，拼进检索问题，让"这张图是什么药材"可回答。
    """
    t0 = time.time()
    display_question = question          # 日志/前端展示用的原始问题
    search_question = question           # 真正用于检索/生成的问题（含图片描述）
    image_name = ""                      # 附图识别出的药材名

    # 图片理解：把图中内容转成文字描述，拼进问题
    img_bytes = _decode_image(image_b64)
    if img_bytes:
        try:
            from com.tcm.qa import ocr
            vis = ocr.describe_for_question(img_bytes, question)
            name = vis.get("name", "")
            desc = vis.get("description", "")
            image_name = name
            if name:
                # 识别出明确药材名：以药名为核心检索，避免形态描述带偏向量
                search_question = f"{name}是什么？请说明它的性味归经和功效主治。{question}"
            elif desc:
                search_question = f"用户问题：{question}\n图片内容：{desc}"
        except Exception as e:
            yield ("token", f"[图片识别失败：{e}]\n")
            yield ("done", None)
            save_log(question, [], f"图片识别失败:{e}", int((time.time() - t0) * 1000))
            return

    # 意图识别前置：非中医药问题不检索、不调生成模型，直接拒答
    # 用 search_question 判断（含图片描述），避免"这张图是什么药材"被误判
    if not intent.is_tcm_question(search_question):
        yield ("out_of_scope", intent.OUT_OF_SCOPE)
        yield ("done", None)
        save_log(display_question, [], intent.OUT_OF_SCOPE, int((time.time() - t0) * 1000))
        return

    # agent loop：检索 → 质量不足则改写重试（最多 MAX_RETRIES 轮）
    # 每轮改写通过 rewriting 事件告知前端，避免用户干等
    def on_retry(attempt, new_query):
        nonlocal retry_events
        retry_events.append((attempt, new_query))

    retry_events = []
    chunks, retries = retrieve_with_retry(search_question, on_retry=on_retry)
    for attempt, new_query in retry_events:
        yield ("rewriting", {"attempt": attempt, "query": new_query})
    yield ("sources", chunks)

    if not quality_ok(chunks):
        yield ("token", REFUSE)
        yield ("done", None)
        save_log(display_question, chunks, REFUSE, int((time.time() - t0) * 1000))
        return

    buf = []
    for piece in build_llm().stream([
        SystemMessage(SYS_PROMPT),
        HumanMessage(build_prompt(chunks, search_question, image_name)),
    ]):
        if piece.content:
            buf.append(piece.content)
            yield ("token", piece.content)

    answer = "".join(buf)
    save_log(display_question, chunks, answer, int((time.time() - t0) * 1000))
    yield ("done", None)


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "治痞块用什么方"
    result = ask(question)

    print(f"问题：{question}")
    print("=" * 60)
    print(result["answer"])
    print("=" * 60)
    print("引用片段：")
    for i, c in enumerate(result["chunks"], 1):
        print(f"  [{i}] {c['source']} fid={c['fid']} {c['title']}（相似度 {c['score']}）")
