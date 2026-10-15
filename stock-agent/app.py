"""
Web 服务层 —— 把 building block 包成网页能用的接口。

这里出现了一个关键动作：**给 block 套上循环**。
到这一步，系统才第一次变成 Anthropic 定义的 Agent：

    "Agents are systems where LLMs dynamically direct their own processes
     and tool usage, maintaining control over how they accomplish tasks."

另外这个文件还负责 Memory —— 也就是 building block 的第三个增强。
注意记忆的取舍：我们只把「用户说了什么、助手答了什么」写进长期记忆，
中间的 tool_calls / tool 结果全部丢掉。理由有两条：

  1. 工具原始返回体积很大，留在历史里会迅速吃掉上下文预算；
  2. 结论已经体现在助手的回答里了，原始数据下一轮几乎用不上。

「什么该进记忆、什么该丢」是面试高频问题，答案永远是「看它下一轮还有没有用」。
"""
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from block import SYSTEM_PROMPT, AugmentedLLM
from llm import build_llm, describe_llm
from tools import TOOLS

MAX_STEPS = 5           # 步数上限：防跑飞，也是成本护栏
HISTORY_TURNS = 12      # 只保留最近 12 条消息，防止上下文无限增长

app = FastAPI(title="Stock Agent · Building Block")
BLOCK = AugmentedLLM(build_llm())
SESSIONS: dict[str, list[dict[str, Any]]] = {}

STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    message: str


class ChatResponse(BaseModel):
    reply: str
    trace: list[dict[str, Any]]
    steps: int
    session_id: str


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
async def config() -> dict[str, Any]:
    """告诉前端当前跑在什么模式，以及块里挂了哪些工具。"""
    info = describe_llm(BLOCK.llm)
    return {
        **info,
        "max_steps": MAX_STEPS,
        "history_turns": HISTORY_TURNS,
        "tools": [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"].split("。")[0] + "。",
                "params": list(t["function"]["parameters"]["properties"].keys()),
            }
            for t in TOOLS
        ],
    }


@app.post("/api/reset")
async def reset(payload: dict[str, Any]) -> dict[str, str]:
    SESSIONS.pop(payload.get("session_id", ""), None)
    return {"status": "ok"}


MISSING_PATTERNS = (
    ("Insufficient Balance", "模型账户余额不足（DeepSeek 返回 402）。"
                             "去 platform.deepseek.com 充值，或换一家的 key 填进 .env，"
                             "也可以先把 .env 里的 key 注释掉、用 Mock 模式继续学。"),
    ("401", "key 无效或已过期（401）。检查 .env 里的 key 是否完整、有没有多余空格。"),
    ("402", "模型账户余额不足（402）。充值后即可恢复，或换一家的 key。"),
    ("429", "触发限流（429）。等一会儿再问，或者在 .env 里换一家。"),
    ("Model Not Exist", "模型名不存在。检查 .env 里的 model 名是否和该供应商提供的一致。"),
)


def explain_model_error(err: Exception) -> str:
    """把上游的报错翻译成「用户知道下一步该干什么」的一句话。

    直接把 402 的原始 JSON 甩给用户是没有意义的 ——
    报错信息的作用是指路，不是自证清白。
    """
    text = str(err)
    for pattern, advice in MISSING_PATTERNS:
        if pattern in text:
            return f"模型调用失败：{advice}"
    return (f"调用模型时出错：{type(err).__name__}: {text[:300]}\n"
            f"（若是网络问题，确认这台机器能访问该供应商的接口）")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    history = SESSIONS.setdefault(req.session_id, [])

    # 组装本轮上下文：system + 长期记忆 + 本轮提问
    messages: list[dict[str, Any]] = (
        [{"role": "system", "content": SYSTEM_PROMPT}] + list(history)
        + [{"role": "user", "content": req.message}]
    )

    trace: list[dict[str, Any]] = []
    reply = ""
    steps = 0

    try:
        for i in range(MAX_STEPS):
            steps = i + 1
            result = await BLOCK.step(messages)        # ← 这里的每一次调用，就是「一个 block」
            trace.extend(result.trace)
            if result.finished:
                reply = result.content
                break
        else:
            reply = ("达到最大步数仍未得出结论。通常说明工具给回的信息不足，"
                     "或者模型在原地打转 —— 这正是「误差累积」的典型表现。")
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=200, content={
            "reply": explain_model_error(e),
            "trace": trace,
            "steps": steps,
            "session_id": req.session_id,
        })

    # 写入长期记忆：只留最终文本，丢弃工具往返细节
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": reply})
    del history[:-HISTORY_TURNS]

    return ChatResponse(reply=reply, trace=trace, steps=steps, session_id=req.session_id)
