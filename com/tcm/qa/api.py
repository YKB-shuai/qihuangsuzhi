"""问答接口（阶段 4.3）：SSE 流式输出。

⚠️ 路由一律用 def，不用 async def：
连接池（PyMySQL）和向量检索都是同步的，async def 里做同步调用会阻塞事件循环。
要做流式，用同步 generator + StreamingResponse 就够了。
"""

from __future__ import annotations

import json

# ⚠️ 同 graph.py：先加载 torch，避免 Windows 上 DLL 冲突段错误
import torch  # noqa: F401

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from com.tcm.db.conn import query_all
from com.tcm.qa import graph, ocr

router = APIRouter(prefix="/api/qa", tags=["qa"])


class AskIn(BaseModel):
    question: str


@router.post("/ask")
def ask(body: AskIn):
    """提问：SSE 流式返回。

    事件格式：data: {"type": "sources"|"token"|"done", "data": ...}
    """
    def gen():
        try:
            for kind, payload in graph.iter_answer(body.question):
                yield "data: " + json.dumps({"type": kind, "data": payload}, ensure_ascii=False) + "\n\n"
        except Exception as e:  # 流式里出错也要告诉前端，别让连接干挂着
            yield "data: " + json.dumps({"type": "error", "data": str(e)}, ensure_ascii=False) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/ocr")
def recognize(file: UploadFile = File(...)):
    """上传图片，自动判断：有文字则提取文字，无文字则识别主体生成问题。

    响应：{"code":0, "data":{"mode","text","subject","kind","question","input"}}
    input 为应填入输入框的内容（文字或自动生成的问题）。
    不直接提问，由用户确认/修改后再点提问。
    """
    try:
        data = file.file.read()
        r = ocr.understand(data)
        return {
            "code": 0,
            "data": {
                "mode": r["mode"],
                "text": r["text"],
                "subject": r["subject"],
                "kind": r["kind"],
                "question": r["question"],
                "input": r["text"] if r["mode"] == "text" else r["question"],
            },
        }
    except ocr.OCRError as e:
        return {"code": 1, "msg": str(e)}
    except Exception as e:
        return {"code": 1, "msg": f"图片识别失败：{e}"}


@router.get("/logs")
def logs(limit: int = 10):
    """最近问答日志（4.5 验收用）。"""
    return query_all(
        "SELECT id, question, hit_count, sources, top_score, elapsed_ms, created_at "
        "FROM qa_log ORDER BY id DESC LIMIT %s",
        (limit,),
    )
