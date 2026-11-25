"""图片文字识别（OCR）：调用阿里云百炼视觉模型提取图片中的文字。

设计要点：
- 复用 .env 里已有的 DASHSCOPE_API_KEY（无需新增密钥）
- 用标准库 urllib 调用 OpenAI 兼容接口，不引入新依赖
- 上传图片先经 Pillow 校验并压缩（最长边 1600px），控制 token 消耗
- 模型 qwen3-vl-plus：实测对中文印刷体、截图文字识别准确

用法：
    from com.tcm.qa import ocr
    text = ocr.extract_text(image_bytes)
"""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request

from PIL import Image

from com.tcm.cfg.settings import settings

# 实测可用的视觉模型（qwen-vl-max 等旧模型名在本账号无权限）
OCR_MODEL = "qwen3-vl-plus"
MAX_IMAGE_MB = 10
MAX_EDGE = 1600          # 上传前缩放的最长边，降低 token 成本
OCR_PROMPT = (
    "提取图片中的所有文字，按原有排版与顺序输出。"
    "只输出识别到的文字，不要描述图片内容，不要添加解释。"
    "如果图片中没有文字，只回答：无文字"
)

# 双模式理解提示词：一次调用同时拿文字 + 主体识别，返回 JSON
UNDERSTAND_PROMPT = (
    "分析这张图片，以严格 JSON 格式返回结果，不要输出 JSON 之外的任何内容。\n"
    "字段说明：\n"
    "- text: 图片中的所有文字，按原排版输出；没有文字则为空字符串 \"\"\n"
    "- subject: 如果图中主体是中药材、中药饮片、方剂或药品，写出它的名称；"
    "否则写图中主要物体的名称，实在无法判断则为空字符串 \"\"\n"
    "- kind: 主体类别，只能是 herb（药材/饮片）、formula（方剂/成药）、object（其他物体）、none（无法判断）之一\n"
    "示例：{\"text\": \"\", \"subject\": \"人参\", \"kind\": \"herb\"}\n"
    "只返回 JSON 对象。"
)


class OCRError(Exception):
    """OCR 失败，携带面向用户的错误信息。"""


def _prepare_image(data: bytes) -> bytes:
    """校验图片格式并压缩，返回 JPEG 字节。"""
    if not data:
        raise OCRError("图片内容为空")
    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        raise OCRError(f"图片过大，请上传 {MAX_IMAGE_MB}MB 以内的图片")

    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()  # 提前触发解码，损坏文件在这里抛错
            im = im.convert("RGB")
            w, h = im.size
            if max(w, h) > MAX_EDGE:
                scale = MAX_EDGE / max(w, h)
                im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=88)
            return buf.getvalue()
    except OCRError:
        raise
    except Exception as e:
        raise OCRError(f"无法读取图片，请确认是有效的 JPG/PNG/WEBP 文件：{e}")


def extract_text(data: bytes) -> str:
    """识别图片中的文字（保留给纯 OCR 场景）；失败抛 OCRError。"""
    text = _call_vision(data, OCR_PROMPT, max_tokens=1000)
    if not text or text in ("无文字", "没有文字", "图片中没有文字"):
        return ""
    return text


def _call_vision(data: bytes, prompt: str, max_tokens: int) -> str:
    """调用视觉模型，返回文本内容（extract_text 与 understand 共用）。"""
    if not settings.DASHSCOPE_API_KEY:
        raise OCRError("未配置阿里云百炼 API Key（DASHSCOPE_API_KEY）")
    image_bytes = _prepare_image(data)
    b64 = base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "model": OCR_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        settings.DASHSCOPE_BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            result = json.load(resp)
        return (result["choices"][0]["message"]["content"] or "").strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise OCRError(f"视觉服务返回错误（HTTP {e.code}）：{body[:160]}")
    except OCRError:
        raise
    except Exception as e:
        raise OCRError(f"视觉服务不可用：{e}")


def _parse_json(raw: str) -> dict:
    """从模型输出里抠出 JSON 对象（容忍 ```json 包裹或前后废话）。"""
    import re

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise OCRError("视觉模型返回格式异常，无法解析")
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise OCRError(f"视觉模型返回格式异常：{e}")
    return obj if isinstance(obj, dict) else {}


def understand(data: bytes) -> dict:
    """图片理解双模式。

    返回：
        {
          "mode": "text" | "subject" | "none",
          "text": 提取到的文字（mode=text 时非空）,
          "subject": 识别出的主体名称（mode=subject 时非空）,
          "kind": herb | formula | object | none,
          "question": 自动生成的问题（mode=subject 时），
        }
    """
    raw = _call_vision(data, UNDERSTAND_PROMPT, max_tokens=600)
    obj = _parse_json(raw)

    text = str(obj.get("text") or "").strip()
    subject = str(obj.get("subject") or "").strip()
    kind = str(obj.get("kind") or "none").strip().lower()
    if kind not in ("herb", "formula", "object", "none"):
        kind = "none"

    # 模型有时会把"无文字"之类塞进 text，统一清掉
    if text in ("无文字", "没有文字", "图片中没有文字", "无", "none", "null"):
        text = ""
    if subject in ("无", "none", "null", "无法判断", ""):
        subject = ""

    # 判定优先级：有文字先用文字（信息更可靠）；否则用识别主体
    if text:
        return {"mode": "text", "text": text, "subject": subject,
                "kind": kind, "question": ""}
    if subject and kind in ("herb", "formula"):
        q = (
            f"图片中的{subject}是什么？请说明它的性味归经、功效主治和用法。"
            if kind == "herb" else
            f"图片中的{subject}是什么方剂？请说明它的组成、功效主治和用法。"
        )
        return {"mode": "subject", "text": "", "subject": subject,
                "kind": kind, "question": q}
    return {"mode": "none", "text": "", "subject": subject,
            "kind": kind, "question": ""}


if __name__ == "__main__":
    import sys
    from pathlib import Path

    path = Path(sys.argv[1] if len(sys.argv) > 1 else "img.png")
    print(f"识别：{path}")
    print(extract_text(path.read_bytes()))
