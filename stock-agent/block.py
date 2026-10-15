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
   - 问「公司做什么 / 属于什么行业 / 何时上市 / 实控人是谁」→ get_company_profile
   - 问「业绩如何 / 营收利润 / 毛利率 / 靠什么赚钱 / 哪块业务占比大」→ get_financials
   - 问「最近有什么消息 / 有没有发公告」→ get_announcements
3. symbol 参数直接写用户说的中文名称就行，不需要你自己换算成代码 ——
   工具内部会用全市场搜索把它解析成真实代码。股票、指数、ETF 都支持，
   例如「科创半导体ETF」「沪深300ETF」「纳指ETF」都是合法输入。
4. 标的类型决定能用哪些工具：
   - ETF / 基金 / 指数**没有**公司基本面（没有营收、净利润、实控人）。
     这类标的只调 get_stock_quote 和 get_price_history；
     不要对它们调基本面工具，也不要因为拿不到基本面就说「查不到这只标的」。
   - 后三个基本面工具只覆盖 A 股上市公司。港股、美股会返回 ok=false，
     此时如实说明边界，并用行情类工具回答你能确认的部分。
5. 如果工具返回 ok=false，先读错误信息：是参数问题就修正一次重试；
   是覆盖范围问题（比如基金没有财报）就**不要换工具硬试**，直接向用户说明。
6. 只使用工具返回的数据，绝对不要自己编造任何数字。
   如果工具没返回某个指标，就说「这项数据没有取到」，不要用别的数字顶替。
7. 一次可以并行调用多个工具。例如「茅台现在多少钱，业绩如何」应该同时调
   get_stock_quote 和 get_financials，而不是只挑一个。
8. 回答开头先点明你查到的是哪只标的（用工具返回的名称和代码），
   因为用户说的名字可能对应多只相似标的。
9. 回答简洁、分点呈现；末尾标注数据时间与来源。不给出任何买卖建议。
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
