"""意图识别：判断用户问题是否属于中医药问答范围（阶段 4.6）。

分层判定，兼顾准确率与延迟：
    1) 空问题            → 非中医
    2) 命中中医术语词典   → 中医（零额外延迟，覆盖绝大多数正常提问）
    3) 命中明确闲聊模式   → 非中医（零额外延迟，拦住"你好/写代码/天气"等）
    4) LLM 二分类兜底     → 边界问题交给模型，失败时保守放行

设计要点：
- import torch 必须在 langchain 之前（Windows DLL 段错误，沿用项目约定）
- 判定结果按问题文本缓存，重复提问不再调 LLM
- LLM 调用失败一律放行，避免网络抖动把所有问题都拒掉

用法：
    from com.tcm.qa import intent
    intent.is_tcm_question("阿魏麝香散的处方")   # True
    intent.is_tcm_question("帮我写个冒泡排序")     # False
"""

from __future__ import annotations

import re

# ⚠️ 与 graph.py / api.py 同坑：torch 必须先于 langchain 加载
import torch  # noqa: F401

from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek

from com.tcm.cfg.settings import settings
from com.tcm.rag import tokenizer

# 非中医问题的统一回复（流式与非流式共用）
OUT_OF_SCOPE = (
    "我是「岐黄溯智」中医药知识助手，只回答中医药相关问题，"
    "比如药材性味、方剂组成、食疗养生、经络穴位、名医典故等。"
    "这个问题不在我的范围内，换个中医药相关的问题试试吧。"
)

# 明确不属于中医药的闲聊/其他领域模式（仅在未命中中医术语时生效，避免误伤）
_CASUAL = re.compile(
    r"(你好|您好|哈喽|嗨+|在吗|早上好|晚上好|中午好|"
    r"谢谢|感谢|多谢|再见|拜拜|"
    r"你是谁|你叫什么|你叫什么名字|讲个笑话|说个笑话|讲个故事|"
    r"天气|气温|下雨|下雪|台风|空气质量|"
    r"写(个|一段|一段)?(代码|程序|函数|脚本|python|java|javascript|js|sql|c\+\+)|"
    r"编程|报错|debug|bug|算法|冒泡排序|"
    r"翻译|translate|"
    r"等于几|几加几|\d+\s*[+\-*/]\s*\d+|解方程|"
    r"几点|现在时间|星期几|今天日期|几号|"
    r"股票|股价|基金|汇率|彩票|球赛|足球|篮球|电影|电视剧|游戏攻略)",
    re.IGNORECASE,
)

# 高频中医基础词：作为词典分词之外的直接兜底（子串匹配，成本极低）
_BASIC_TCM = (
    "中医", "中药", "方剂", "药材", "草药", "汤药", "针灸", "艾灸", "推拿",
    "穴位", "经络", "把脉", "脉象", "舌苔", "归经", "性味", "功效", "主治",
    "煎服", "食疗", "养生", "本草", "名医", "古籍", "医案", "调理",
    "君臣佐使", "四气五味", "阴阳五行", "气血", "湿热", "虚寒",
)

# 用于 LLM 二分类的术语集合（懒加载，只加载一次）
_terms: set[str] | None = None

_CLASSIFY_SYS = (
    "你是问题分类器。判断用户问题是否属于中医药领域，"
    "包括：中药材、方剂、中成药、食疗养生、经络穴位、中医理论与诊断、"
    "名医典故、中医古籍、常见症状的中医调理。\n"
    "只回答「是」或「否」一个字，不要任何解释。"
)


def _load_terms() -> set[str]:
    """从 docs/tcm_terms.txt 提取词表（格式：词语 词频 词性）。"""
    global _terms
    if _terms is None:
        path = tokenizer.DICT_PATH
        loaded: set[str] = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if parts and len(parts[0]) >= 2:
                    loaded.add(parts[0])
        _terms = loaded
    return _terms


def _llm_classify(question: str) -> bool:
    """LLM 二分类兜底；异常一律放行。"""
    try:
        llm = ChatDeepSeek(
            api_key=settings.DEEPSEEK_API_KEY,
            api_base=settings.DEEPSEEK_BASE_URL,
            model=settings.DEEPSEEK_MODEL_CHAT,
            temperature=0,
            max_tokens=8,
        )
        resp = llm.invoke([
            SystemMessage(_CLASSIFY_SYS),
            HumanMessage(question),
        ])
        content = (resp.content or "").strip()
        return "是" in content and "否" not in content
    except Exception:
        # 网络/额度异常时放行，交给后续检索链 + 相似度阈值处理
        return True


@lru_cache(maxsize=512)
def is_tcm_question(question: str) -> bool:
    """判断问题是否属于中医药范围。命中缓存则不重复调用。"""
    q = (question or "").strip()
    if not q:
        return False

    # 1) 中医术语命中（分词查词典 + 基础词子串）
    words = tokenizer.cut(q)
    terms = _load_terms()
    if any(len(w) >= 2 and w in terms for w in words):
        return True
    if any(k in q for k in _BASIC_TCM):
        return True

    # 2) 明确闲聊模式
    if _CASUAL.search(q):
        return False

    # 3) LLM 兜底
    return _llm_classify(q)


if __name__ == "__main__":
    samples = [
        "阿魏麝香散的处方是什么？",
        "清暑化湿用什么方",
        "糖尿病吃什么好",
        "你好",
        "今天天气怎么样",
        "帮我写个冒泡排序",
        "1+1等于几",
        "你是谁",
    ]
    for s in samples:
        print(f"{'中医' if is_tcm_question(s) else '非中医'}  {s}")
