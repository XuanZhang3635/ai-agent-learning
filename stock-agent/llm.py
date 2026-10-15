"""
模型适配层 —— 把「不同家的大模型」抹平成同一个接口。

面试常问：换模型要改多少代码？最好的答案是「改 .env」。

所以这里定一个统一协议，真实模型和 Mock 模型都遵守它：

    async chat(messages, tools) -> {"content": str, "tool_calls": [...]}

接真实模型：在项目根目录建一个 .env（已被 gitignore，不会误传）：

    DEEPSEEK_API_KEY=sk-xxxx

或者用任何 OpenAI 兼容端点：

    OPENAI_API_KEY=sk-xxxx
    OPENAI_BASE_URL=https://api.deepseek.com/v1
    OPENAI_MODEL=deepseek-chat

没配 key 就自动落到 MockLLM —— 用规则模拟模型的决策过程。
这样在你还没配 key 的时候，整条链路依然是通的、可观察的。
"""
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import httpx

from tools import resolve_symbols

# python-dotenv 随 uvicorn[standard] 一起装上了；没装也不影响运行
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:  # pragma: no cover
    pass


class BaseLLM:
    name = "base"

    async def chat(self, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError


# --------------------------------------------------------------------------
# 真实模型：任何 OpenAI 兼容端点（OpenAI / DeepSeek / 通义 / Kimi / Ollama ...）
# --------------------------------------------------------------------------
class OpenAICompatLLM(BaseLLM):
    def __init__(self, api_key: str, base_url: str, model: str,
                 provider: str = "openai") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.provider = provider
        self.name = model

    async def chat(self, messages, tools):
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"模型接口返回 {resp.status_code}: {resp.text[:400]}")
            data = resp.json()

        msg = data["choices"][0]["message"]
        raw_calls = msg.get("tool_calls") or []

        return {
            "content": msg.get("content") or "",
            "tool_calls": [
                {
                    "id": c.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    "name": c["function"]["name"],
                    "arguments": c["function"].get("arguments") or "{}",
                }
                for c in raw_calls
            ],
        }


# --------------------------------------------------------------------------
# Mock 模型：没有 key 时的替身，用规则模拟「模型会怎么决定」
# --------------------------------------------------------------------------
# 忘了这一步，118730083 会以 1.1873e+08 的形式出现在回答里
def _n(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:,.{digits}f}"


class MockLLM(BaseLLM):
    """这不是一个真模型，而是一段「模仿模型决策」的规则。

    它存在的意义：让你在没有任何 API key 的情况下，
    也能完整观察到 schema 定义 → 工具选择 → 参数填充 → 结果写回 的全过程。
    """

    name = "mock-rule-based"

    async def chat(self, messages, tools):
        user_msgs = [m for m in messages if m.get("role") == "user"]
        question = user_msgs[-1]["content"] if user_msgs else ""

        already_called = [m for m in messages if m.get("role") == "tool"]
        if already_called:
            return {"content": self._compose(messages, already_called), "tool_calls": []}

        # 光靠上下文就能回答的问题：不调工具，直接答。这就是 Memory 在起作用。
        if re.search(r"上一句|上一个问题|刚才|之前|还记得|记忆|聊过|我说过", question):
            return {"content": self._recall(user_msgs), "tool_calls": []}

        calls = self._decide(question)
        if not calls:
            return {"content": self._no_tool(tools), "tool_calls": []}
        return {"content": "", "tool_calls": calls}

    def _recall(self, user_msgs) -> str:
        prior = [m["content"] for m in user_msgs[:-1]]
        if not prior:
            return "这是我们这轮对话的第一个问题，还没有可以回忆的内容。多问几轮再试试。"
        recent = "\n".join(f"- {q}" for q in prior[-3:])
        return (f"记得。你前面问过：\n{recent}\n\n"
                "（这段内容来自 Memory —— 它没有出现在你这次的提问里，"
                "是我从对话历史里取回来的。历史一旦不进上下文，我就答不上来。）")

    def _no_tool(self, tools) -> str:
        names = "、".join(t["function"]["name"] for t in tools)
        return (f"我判断这个问题不需要调用工具，但光靠我自己也没有把握回答。\n"
                f"当前块里挂了 {len(tools)} 个工具：{names}。\n"
                f"把问题落到具体标的上会更容易命中，例如「贵州茅台现在多少钱」"
                f"或「宁德时代最近 10 个交易日走势」。")

    # ---- 决策：像模型一样挑工具、填参数 ----
    def _decide(self, q: str) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []

        def emit(name: str, args: dict[str, Any]) -> None:
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            })

        symbols = resolve_symbols(q)
        symbol_arg = ",".join(symbols)

        # 注意顺序：公告里的「最近有什么」也会命中走势的「最近」，
        # 所以先算公告，再把它从走势里排除掉。规则式实现最容易在这种
        # 关键词打架的地方出错 —— 这正是 Mock 与真模型最本质的差距。
        wants_ann = bool(re.search(r"公告|消息|新闻|披露|发生了什么|最近有什么", q))
        wants_history = bool(
            re.search(r"走势|趋势|最近|近期|这几天|日线|区间|涨幅|跌了多少|表现|k线", q)
        ) and not wants_ann
        wants_quote = bool(re.search(
            r"现在|现价|多少钱|股价|报价|涨|跌|行情|估值|市盈率|市值|换手|成交|怎么样|如何|看看", q))
        wants_profile = bool(re.search(r"做什么|主营业务|简介|介绍|行业|是什么公司|上市|实控人|老板|官网|员工", q))
        wants_financials = bool(re.search(
            r"财报|业绩|营收|收入|净利润|利润|毛利|roe|净资产收益率|负债率|赚不赚钱|靠什么赚|收入构成|主营构成|哪块业务|占比", q, re.I))

        if not symbols:
            return []

        if wants_history:
            m = re.search(r"(\d{1,3})\s*(?:个)?\s*(?:交易日|天|日)", q)
            days = int(m.group(1)) if m else 20
            emit("get_price_history", {"symbol": symbols[0], "days": days})

        if wants_profile:
            emit("get_company_profile", {"symbol": symbols[0]})

        if wants_financials:
            emit("get_financials", {"symbol": symbols[0]})

        if wants_ann:
            emit("get_announcements", {"symbol": symbols[0], "limit": 8})

        # 只要在问行情，或者前面什么都没命中，就补一次实时行情
        if wants_quote or not calls:
            emit("get_stock_quote", {"symbol": symbol_arg})

        return calls

    # ---- 收尾：把工具返回值组织成回答 ----
    def _compose(self, messages, tool_msgs) -> str:
        blocks: list[str] = []
        for m in tool_msgs:
            try:
                payload = json.loads(m["content"])
            except (json.JSONDecodeError, KeyError):
                continue
            if not payload.get("ok"):
                blocks.append(f"未能取到数据：{payload.get('error')}")
                continue

            kind = payload.get("kind")
            data = payload["data"]

            if kind == "quote":
                for d in data:
                    chg = d.get("change") or 0
                    direction = "上涨" if chg > 0 else ("下跌" if chg < 0 else "持平")
                    line = (
                        f"{d['name']}（{d['code']} · {d.get('market')}）"
                        f"最新价 {_n(d.get('price'))} {d.get('currency')}，"
                        f"{direction} {_n(abs(chg))}（{d.get('change_pct')}%）；"
                        f"今日区间 {_n(d.get('low'))}–{_n(d.get('high'))}"
                    )
                    if d.get("volume"):
                        line += f"；成交量 {d['volume']:,.0f}{d.get('volume_unit')}"
                    if d.get("turnover_pct"):
                        line += f"；换手率 {d['turnover_pct']}%"
                    if d.get("pe_ttm"):
                        line += f"；市盈率(TTM) {d['pe_ttm']}"
                    if d.get("pb") is not None:
                        line += f"；市净率 {d['pb']}"
                    if d.get("market_cap"):
                        line += f"；总市值 {d['market_cap']:,.2f}{d.get('market_cap_unit')}"
                    line += f"。数据时间 {d.get('updated_at')}。"
                    blocks.append(line)

            elif kind == "history":
                blocks.append(
                    f"{data['code']}（{data.get('market')}）最近 {data['trading_days']} 个交易日"
                    f"（{data['start_date']} 至 {data['end_date']}）"
                    f"区间涨跌 {data['period_change_pct']}%，"
                    f"区间最高 {_n(data['period_high'])}、最低 {_n(data['period_low'])}，"
                    f"日均成交 {_n(data['avg_volume'], 0)}{data.get('volume_unit', '')}。"
                )

            elif kind == "profile":
                p = data["profile"]
                lines = [f"【{data['name']}】公司概况"]
                for label in ("公司全称", "所属行业（东财）", "上市板块", "上市日期",
                              "实际控制人", "员工人数", "公司官网", "办公地址",
                              "主营业务", "经营范围", "公司简介"):
                    if p.get(label):
                        lines.append(f"{label}：{p[label]}")
                blocks.append("\n".join(lines))

            elif kind == "financials":
                lines = [f"{data['code']} 财务与经营结构"
                         f"（报告期 {data.get('report_period', '—')}）"]
                for m in data.get("metrics", []):
                    yoy = f"，同比 {m['同比']}" if m.get("同比") else ""
                    lines.append(f"{m['指标']}：{m['数值']}{m['单位']}{yoy}")
                mb = data.get("main_business") or {}
                if mb:
                    lines.append(f"主营构成（{data.get('main_business_period', '')}）")
                    for dim, items in mb.items():
                        for it in items:
                            lines.append(
                                f"  {dim}｜{it['项目']}：收入 {it['收入']} 亿元，"
                                f"占比 {it['占比']}，毛利率 {it['毛利率']}"
                            )
                blocks.append("\n".join(lines))

            elif kind == "announcements":
                lines = [f"{data['code']} 最近 {data['count']} 条公告"]
                for i, it in enumerate(data["items"], 1):
                    lines.append(f"{i}. [{it['披露时间']}｜{it['类型']}] {it['标题']}")
                blocks.append("\n".join(lines))

        body = "\n".join(f"- {b}" for b in blocks) if blocks else "没有取到可用数据。"
        return body + "\n\n（以上由本地数据源实时返回，不构成投资建议。）"


# --------------------------------------------------------------------------
# 工厂：按 .env / 环境变量选择实现
# --------------------------------------------------------------------------
# 归一化：不管用哪家，调用方看到的都是同一个 BaseLLM。
# 加一家新供应商 = 往这张表里加一行 + 在 .env 里写个 key。
PROVIDERS: dict[str, tuple[str, str, str]] = {
    #  provider    : (key 环境变量,        base_url,                     默认模型)
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1", "deepseek-chat"),
    "openai":   ("OPENAI_API_KEY",   "https://api.openai.com/v1",  "gpt-4o-mini"),
}
# 显式覆盖用的通用变量名（任何 OpenAI 兼容端点都能接）
_ENV_BASE = "OPENAI_BASE_URL"
_ENV_MODEL = "OPENAI_MODEL"


def build_llm() -> BaseLLM:
    """有 key 就用真模型，没有就退回 Mock。顺序：deepseek → openai。"""
    for provider, (key_env, base, model) in PROVIDERS.items():
        api_key = (os.getenv(key_env) or "").strip()
        # 通用变量也认，方便接表里没有的端点
        if not api_key and provider == "openai":
            api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            continue
        return OpenAICompatLLM(
            api_key=api_key,
            base_url=os.getenv(_ENV_BASE) or base,
            model=os.getenv(_ENV_MODEL) or model,
            provider=provider,
        )
    return MockLLM()


def describe_llm(llm: BaseLLM) -> dict[str, str]:
    """给前端展示用：现在跑的到底是真的还是假的。"""
    if isinstance(llm, MockLLM):
        return {"mode": "mock", "model": llm.name,
                "hint": "未检测到 API key，正在用规则模拟模型决策。"
                        "在 stock-agent/.env 里填 DEEPSEEK_API_KEY 即可切到真实模型。"}
    return {
        "mode": "live",
        "model": llm.name,
        "provider": getattr(llm, "provider", "openai"),
        "endpoint": getattr(llm, "base_url", ""),
        "hint": "已接真实模型，工具选择与最终回答都由模型自己决定。",
    }
