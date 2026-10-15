"""
工具层 —— Building block 的三个增强之一：Tools / Retrieval。

Anthropic《Building effective agents》原文：

    "The basic building block of agentic systems is an LLM enhanced with
     augmentations such as retrieval, tools, and memory. Our current models
     can actively use these capabilities—generating their own search queries,
     selecting appropriate tools, and determining what information to retain."

并列出了两条实现建议：

    "We recommend focusing on two key aspects of the implementation:
     tailoring these capabilities to your specific use case and ensuring
     they provide an easy, well-documented interface for your LLM."

翻译成本文件的两条规矩：

  规矩一：schema 是写给模型看的 API 文档，不是注释。
      模型只能靠 name / description / parameters 来决定「选哪个工具、怎么填参数」。
      描述含糊 → 它选错工具；参数说明不全 → 它自己编参数。

  规矩二：工具要薄，失败要体面。
      一个工具只做一件事。出错时返回一句人能看懂、模型能据此自我修正的话，
      而不是抛异常 —— 抛异常会直接打断整个调用链。
"""
import json
import re
from typing import Any

import httpx

QUOTE_API = "http://qt.gtimg.cn/q={symbols}"
KLINE_API = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

# 别名表：把「人话里的名字」映射到接口代码。
# 这是「tailoring to your use case」最直接的一处体现 ——
# 用户说「茅台」时，你不能要求他先学会写 sh600519。
ALIASES: dict[str, str] = {
    "上证指数": "sh000001", "上证综指": "sh000001",
    "深证成指": "sz399001", "创业板指": "sz399006",
    "沪深300": "sh000300", "科创50": "sh000688",
    "贵州茅台": "sh600519", "茅台": "sh600519",
    "五粮液": "sz000858", "宁德时代": "sz300750", "比亚迪": "sz002594",
    "平安银行": "sz000001", "招商银行": "sh600036", "招行": "sh600036",
    "中国平安": "sh601318", "工商银行": "sh601398", "中信证券": "sh600030",
    "东方财富": "sz300059", "长江电力": "sh600900", "隆基绿能": "sh601012",
    "腾讯控股": "hk00700", "腾讯": "hk00700", "美团": "hk03690",
    "小米集团": "hk01810", "阿里巴巴": "hk09988",
    "苹果": "usAAPL", "特斯拉": "usTSLA", "英伟达": "usNVDA",
    "微软": "usMSFT", "亚马逊": "usAMZN", "谷歌": "usGOOGL",
}

# 纯字母被误当成美股代码的情况（PE、AI 这类词太常见了）
_TICKER_STOPWORDS = {
    "A", "I", "PE", "PB", "ROE", "EPS", "AI", "OK", "AND", "OR", "THE",
    "VS", "MA", "KDJ", "MACD", "RSI", "ETF", "IPO", "CEO", "GDP", "CPI",
}


def _infer_a_share(code: str) -> str:
    """6 位数字 → 补上交易所前缀。"""
    head = code[0]
    if head in "659":
        return "sh" + code      # 6=沪市主板/科创  5=沪市基金  9=沪市B股
    if head in "0123":
        return "sz" + code      # 0/3=深市主板与创业板  1=深市基金  2=深市B股
    if head in "48":
        return "bj" + code      # 北交所
    return "sh" + code


def resolve_symbols(text: str) -> list[str]:
    """从一句人话里捞出股票代码，按出现顺序返回，最多 3 个。

    支持：中文简称、600519、sh600519、00700、hk00700、AAPL、usAAPL
    """
    found: list[str] = []

    def push(code: str) -> None:
        if code not in found:
            found.append(code)

    # 1) 中文名 / 别名 —— 长名优先，避免「茅台」先把「贵州茅台」吃掉
    for name in sorted(ALIASES, key=len, reverse=True):
        if name in text:
            push(ALIASES[name])

    # 2) 显式带前缀的代码
    for m in re.finditer(r"\b(sh|sz|bj|hk|us)\s*([0-9]{4,6}|[A-Za-z]{1,5})\b", text, re.I):
        push(m.group(1).lower() + m.group(2).upper())

    # 3) 裸的 6 位数字（A 股）
    for m in re.finditer(r"(?<![0-9])(\d{6})(?![0-9])", text):
        push(_infer_a_share(m.group(1)))

    # 4) 裸的 5 位数字（港股）
    for m in re.finditer(r"(?<![0-9])(\d{5})(?![0-9])", text):
        if not _infer_a_share(m.group(1)) in found:
            push("hk" + m.group(1))

    # 5) 只有在前面都没捞到时，才赌一把纯字母是美股代码
    if not found:
        for m in re.finditer(r"\b([A-Za-z]{1,5})\b", text):
            t = m.group(1).upper()
            if t not in _TICKER_STOPWORDS:
                push("us" + t)

    return found[:3]


# --------------------------------------------------------------------------
# 工具 1：实时行情
# --------------------------------------------------------------------------
# 各市场的字段布局并不一致，而且单位也不一样：
#   A 股  成交量按「手」，成交额按「万元」，市值按「亿元」
#   港股  成交量按「股」，市值按「亿港元」
#   美股  成交量按「股」，市值按「亿美元」
# 所以不能一套索引读到底 —— 这类「看起来一样其实不一样」的坑，
# 是数据类工具最常见的翻车点。
_MARKET_META: dict[str, dict[str, str]] = {
    "sh": {"market": "A股", "currency": "CNY", "volume_unit": "手", "amount_unit": "万元", "cap_unit": "亿元"},
    "sz": {"market": "A股", "currency": "CNY", "volume_unit": "手", "amount_unit": "万元", "cap_unit": "亿元"},
    "bj": {"market": "A股", "currency": "CNY", "volume_unit": "手", "amount_unit": "万元", "cap_unit": "亿元"},
    "hk": {"market": "港股", "currency": "HKD", "volume_unit": "股", "amount_unit": "港元", "cap_unit": "亿港元"},
    "us": {"market": "美股", "currency": "USD", "volume_unit": "股", "amount_unit": "美元", "cap_unit": "亿美元"},
}


def _parse_quote(line: str) -> dict[str, Any] | None:
    """解析一行行情。前缀从 `v_sh600519="..."` 里取，不能从数据体里取 ——
    港股返回的代码字段是 '00700'，本身不带交易所前缀。"""
    key, sep, rest = line.partition('="')
    if not sep:
        return None
    prefix = key.strip()[2:4].lower()
    meta = _MARKET_META.get(prefix, _MARKET_META["sh"])
    f = rest.rstrip('";\n').split("~")

    def num(i: int) -> float | None:
        try:
            return float(f[i])
        except (IndexError, ValueError):
            return None

    # 只有 A 股的 46 位是市净率；港股/美股那个位置是公司英文名
    pb = num(46) if prefix in ("sh", "sz", "bj") else None

    return {
        "code": f[2],
        "name": f[1],
        "market": meta["market"],
        "currency": meta["currency"],
        "price": num(3),
        "prev_close": num(4),
        "open": num(5),
        "change": num(31),
        "change_pct": num(32),
        "high": num(33),
        "low": num(34),
        "volume": num(36),
        "volume_unit": meta["volume_unit"],
        "amount": num(37),
        "amount_unit": meta["amount_unit"],
        "turnover_pct": num(38),
        "pe_ttm": num(39),
        "pb": pb,
        "market_cap": num(45),
        "market_cap_unit": meta["cap_unit"],
        "updated_at": f[30] if len(f) > 30 else None,
    }


async def get_stock_quote(symbol: str) -> dict[str, Any]:
    """取实时行情（A 股 / 港股 / 美股 / 指数）。"""
    codes = resolve_symbols(str(symbol))
    if not codes:
        return {"ok": False, "error": f"无法从 {symbol!r} 识别出股票代码，"
                                      f"请给出 6 位代码（如 600519）或常见简称（如 贵州茅台）。"}

    async with httpx.AsyncClient(timeout=8) as client:
        resp = await client.get(QUOTE_API.format(symbols=",".join(codes)))
        text = resp.content.decode("gbk", errors="replace")

    quotes = []
    for line in text.strip().splitlines():
        try:
            parsed = _parse_quote(line)
        except (IndexError, ValueError):
            continue
        if parsed:
            quotes.append(parsed)

    if not quotes:
        return {"ok": False, "error": f"行情接口没有返回 {codes} 的数据，"
                                      f"请确认代码是否正确、是否为交易日。"}
    return {"ok": True, "kind": "quote", "data": quotes}


# --------------------------------------------------------------------------
# 工具 2：历史走势
# --------------------------------------------------------------------------
async def get_price_history(symbol: str, days: int = 20) -> dict[str, Any]:
    """取最近 N 个交易日的日线，并算出区间涨跌等统计量。

    直接给模型 20 根 K 线，不如给它「算好的结论」——
    省 token、少出错，还能避免它在心里做算术。
    """
    codes = resolve_symbols(str(symbol))
    if not codes:
        return {"ok": False, "error": f"无法从 {symbol!r} 识别出股票代码。"}

    code = codes[0]
    days = max(5, min(int(days or 20), 120))

    async with httpx.AsyncClient(timeout=8) as client:
        resp = await client.get(
            KLINE_API, params={"param": f"{code},day,,,{days},qfq"}
        )
        payload = resp.json()

    node = (payload.get("data") or {}).get(code) or {}
    bars = node.get("qfqday") or node.get("day") or []
    if not bars:
        return {"ok": False, "error": f"没有取到 {code} 的日线数据。"}

    bars = bars[-days:]          # 接口常会多给一根，这里严格裁到调用方要的天数
    prefix = code[:2].lower()
    meta = _MARKET_META.get(prefix, _MARKET_META["sh"])

    rows = [
        {"date": b[0], "open": float(b[1]), "close": float(b[2]),
         "high": float(b[3]), "low": float(b[4]), "volume": float(b[5])}
        for b in bars
    ]
    closes = [r["close"] for r in rows]
    first, last = closes[0], closes[-1]

    return {
        "ok": True,
        "kind": "history",
        "data": {
            "code": code,
            "market": meta["market"],
            "currency": meta["currency"],
            "trading_days": len(rows),
            "start_date": rows[0]["date"],
            "end_date": rows[-1]["date"],
            "start_close": first,
            "end_close": last,
            "period_change_pct": round((last - first) / first * 100, 2),
            "period_high": max(r["high"] for r in rows),
            "period_low": min(r["low"] for r in rows),
            "volume_unit": meta["volume_unit"],
            "avg_volume": round(sum(r["volume"] for r in rows) / len(rows)),
            "recent_5_days": [
                {"date": r["date"], "close": r["close"]} for r in rows[-5:]
            ],
        },
    }


# --------------------------------------------------------------------------
# 工具 3：本地知识库检索（Retrieval）
# --------------------------------------------------------------------------
def search_company_notes(query: str) -> dict[str, Any]:
    """在本地公司资料库里做关键词检索。

    知识点：检索工具的 description 里必须写清楚「库里有什么」。
    否则模型不知道什么时候该用你，只好瞎猜或者干脆不检索。
    """
    from notes import search
    hits = search(str(query))
    if not hits:
        return {"ok": False, "error": f"本地资料库里没有和 {query!r} 相关的内容。"}
    return {"ok": True, "kind": "notes", "data": hits}


# --------------------------------------------------------------------------
# 工具注册表：schema 与实现放在一起，改一处不会漏另一处
# --------------------------------------------------------------------------
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_quote",
            "description": (
                "查询标的的实时行情快照，返回最新价、涨跌幅、最高最低、"
                "成交量额、换手率、市盈率、总市值等。"
                "用户问「现在多少钱」「今天涨了吗」「估值多少」时使用。"
                "支持 A 股、港股、美股和主要指数。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "标的名称或代码，如 '贵州茅台'、'600519'、'hk00700'、'usAAPL'。多个标的用逗号分隔。",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_history",
            "description": (
                "查询标的最近 N 个交易日的日线走势，返回区间涨跌幅、区间最高最低、"
                "均量和最近 5 日收盘价。"
                "用户问「最近走势」「这几天涨了多少」「近期表现」时使用。"
                "注意这是区间统计，不是实时行情；问当前价格请用 get_stock_quote。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "标的名称或代码，如 '宁德时代' 或 '300750'。"},
                    "days": {"type": "integer", "description": "回看的交易日数量，默认 20，范围 5-120。"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_company_notes",
            "description": (
                "检索本地公司资料库，返回公司主营业务、所属行业、核心看点与已知风险。"
                "资料库只收录了有限几只标的，只覆盖基本面描述，不含实时数据。"
                "当用户问「这家公司是做什么的」「主营业务」「有什么风险」时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词，如 '比亚迪 主营' 或 '宁德时代 风险'。"},
                },
                "required": ["query"],
            },
        },
    },
]

# name -> 可调用对象
TOOL_REGISTRY = {
    "get_stock_quote": get_stock_quote,
    "get_price_history": get_price_history,
    "search_company_notes": search_company_notes,
}


async def call_tool(name: str, args: dict[str, Any]) -> str:
    """统一的工具调用入口。永远返回字符串，永远不抛异常。

    为什么永远不抛异常？因为工具报错的正确姿势是「把错误告诉模型」，
    让它自己换个参数重试；而不是让异常一路冒到顶层，把整轮对话打断。
    """
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        available = "、".join(TOOL_REGISTRY)
        return json.dumps(
            {"ok": False, "error": f"没有名为 {name!r} 的工具。可用工具：{available}。"},
            ensure_ascii=False,
        )

    try:
        result = fn(**(args or {}))
        if hasattr(result, "__await__"):
            result = await result
    except TypeError as e:
        result = {"ok": False, "error": f"参数不对：{e}。请检查参数名和类型后重试。"}
    except Exception as e:  # noqa: BLE001 —— 这里就是要兜住一切
        result = {"ok": False, "error": f"工具执行失败：{type(e).__name__}: {e}"}

    return json.dumps(result, ensure_ascii=False)
