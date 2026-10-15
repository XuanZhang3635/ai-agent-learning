"""
这个文件就是 Anthropic 说的那个 building block 本身。

它的全部职责只有一件事：

    把 messages + 工具清单交给模型 → 模型说「我要调这个工具」→ 执行 → 写回上下文

注意它**不知道循环的存在**。它只做「一次往返」。
上面套一层 for 循环，它才升级成 Agent；用代码写死顺序，就是 Workflow。

所以这个类刻意不包含：重试、最大步数、多轮记忆——那些都属于循环层的职责。
边界划清楚了，后面 debug 时你才知道该看哪一层。
"""
from dataclasses import dataclass, field
from typing import Any

from tools import TOOLS, call_tool

SYSTEM_PROMPT = """你是一个严谨的股票信息助手。

工作方式：
1. 先判断用户问的是哪一类信息，再决定调用哪个工具，不要凭记忆回答。
2. 工具选择参考：
   - 问「现在多少钱 / 今天涨跌 / 估值高低」→ get_stock_quote
   - 问「最近走势 / 这几天表现 / 区间涨跌」→ get_price_history
   - 问「公司做什么 / 主营业务 / 有什么风险」→ search_company_notes
3. 如果工具返回 ok=false，先读错误信息、修正参数重试一次；仍然失败就如实告诉用户。
4. 只使用工具返回的数据，绝对不要自己编造任何数字。
5. 回答简洁、分点呈现；末尾标注数据时间。不给出任何买卖建议。
"""


@dataclass
class BlockResult:
    """一次往返的产物。trace 是给前端展示用的，让你能肉眼看见决策过程。"""
    finished: bool
    content: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)


class AugmentedLLM:
    """LLM + Tools + Retrieval + Memory 的组合体。"""

    def __init__(self, llm, tools: list[dict[str, Any]] | None = None) -> None:
        self.llm = llm
        self.tools = tools if tools is not None else TOOLS

    async def step(self, messages: list[dict[str, Any]]) -> BlockResult:
        """执行一次完整往返，并就地更新 messages。

        返回 finished=True 表示模型直接给了最终答案，本轮结束。
        """
        trace: list[dict[str, Any]] = []

        # 1) 把当前上下文和工具清单交给模型，由它决定「回答还是调工具」
        out = await self.llm.chat(messages, self.tools)
        calls = out.get("tool_calls") or []

        # 2) 没有工具调用 → 模型认为可以直接回答，块的使命完成
        if not calls:
            content = out.get("content") or ""
            messages.append({"role": "assistant", "content": content})
            return BlockResult(finished=True, content=content, trace=trace)

        # 3) 有工具调用 → 先把 assistant 这一步（含调用意图）写回上下文，
        #    再逐个执行，把结果作为 tool 消息写回。
        #    顺序不能反：OpenAI 协议要求 tool 消息必须紧跟在对应的 assistant 之后。
        messages.append({
            "role": "assistant",
            "content": out.get("content") or None,
            "tool_calls": [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"]},
                }
                for c in calls
            ],
        })

        for call in calls:
            import json
            try:
                args = json.loads(call["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}

            observation = await call_tool(call["name"], args)
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": observation,
            })

            trace.append({
                "type": "tool_call",
                "name": call["name"],
                "arguments": args,
                "result": json.loads(observation),
            })

        return BlockResult(finished=False, trace=trace)
