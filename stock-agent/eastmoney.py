"""
真实基本面数据源 —— 东方财富 F10 公开接口。

为什么要单独一个文件？
    行情（腾讯）和基本面（东方财富）是两个完全不同的上游，
    把它们混在 tools.py 里，改一个源会牵动另一个。
    数据源归数据源，工具归工具 —— 这是「薄工具」的前提。

关于覆盖范围（很重要，别想当然）：
    RPT_F10_* 这套报表**只有 A 股**（含北交所）。
    港股、美股会返回 `报表配置不存在` 或 `返回数据为空`。
    所以 to_secucode() 对港美股直接返回 None，
    由上层转成一句模型能看懂的提示，而不是让它拿到空数据去瞎编。
"""
from __future__ import annotations

import time
from typing import Any

import httpx

F10_API = "https://datacenter.eastmoney.com/securities/api/data/v1/get"
ANN_API = "https://np-anotice-stock.eastmoney.com/api/security/ann"
SUGGEST_API = "https://searchapi.eastmoney.com/api/suggest/get"

# 搜索接口需要一个公开的 token（就是网页版自己在用的那个，不是用户凭据）
SUGGEST_TOKEN = "D43BF722C8E33BDC906FB84D85E326E8"

# 这套接口对 UA 敏感，不带 UA 容易被掐
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Referer": "https://emweb.securities.eastmoney.com/",
}

# 极简缓存。同一轮对话里模型很可能连着问「贵州茅台主营 + 贵州茅台财务」，
# 不缓存就会把同一个接口打两遍。TTL 取 10 分钟 —— 基本面数据本来就不秒变。
_CACHE: dict[str, tuple[float, Any]] = {}
_TTL = 600


def _cache_get(key: str) -> Any | None:
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    return None


def _cache_put(key: str, value: Any) -> Any:
    _CACHE[key] = (time.time(), value)
    return value


# --------------------------------------------------------------------------
# 代码转换
# --------------------------------------------------------------------------
_SECU_SUFFIX = {"sh": "SH", "sz": "SZ", "bj": "BJ"}


def to_secucode(symbol: str) -> str | None:
    """把内部代码（sh600519）转成东方财富的 SECUCODE（600519.SH）。

    港美股返回 None —— F10 报表不支持，不要在调用方那里假设它有数据。
    """
    s = str(symbol).strip()
    prefix, rest = s[:2].lower(), s[2:]
    suffix = _SECU_SUFFIX.get(prefix)
    if not suffix:
        return None
    if rest.isdigit() and len(rest) == 6:
        return f"{rest}.{suffix}"
    return None


def to_plain_code(symbol: str) -> str | None:
    """公告接口要的是不带交易所后缀的裸代码（600519）。"""
    secu = to_secucode(symbol)
    return secu.split(".")[0] if secu else None


def unsupported_reason(symbol: str) -> str:
    """给模型看的一句话解释。措辞要让它能据此改口，而不是干瞪眼。"""
    market = {"hk": "港股", "us": "美股"}.get(str(symbol)[:2].lower(), "该市场")
    return (f"东方财富 F10 公司资料库暂不覆盖{market}（{symbol}）。"
            f"基本面类问题目前只支持 A 股，请改用行情/走势类工具回答能确认的部分。")


# --------------------------------------------------------------------------
# 通用请求
# --------------------------------------------------------------------------
async def _f10(report: str, secucode: str, *, sort: str | None = None,
               size: int = 20) -> list[dict[str, Any]]:
    """打一次 F10 报表接口，返回 data 列表（失败一律返回空列表）。"""
    key = f"{report}:{secucode}:{sort}:{size}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    params = {
        "reportName": report,
        "columns": "ALL",
        "filter": f'(SECUCODE="{secucode}")',
        "pageSize": size,
        "source": "HSF10",
        "client": "PC",
    }
    if sort:
        params["sortColumns"] = sort
        params["sortTypes"] = "-1"

    try:
        async with httpx.AsyncClient(timeout=10, headers=HEADERS) as client:
            resp = await client.get(F10_API, params=params)
            payload = resp.json()
    except (httpx.HTTPError, ValueError):
        return []

    data = (payload.get("result") or {}).get("data") or []
    return _cache_put(key, data)


# --------------------------------------------------------------------------
# 1) 公司概况
# --------------------------------------------------------------------------
_PROFILE_FIELDS = {
    "ORG_NAME": "公司全称",
    "EM2016": "所属行业（东财）",
    "SECURITY_TYPE": "上市板块",
    "LISTING_DATE": "上市日期",
    "FOUND_DATE": "成立日期",
    "CHAIRMAN": "董事长",
    "ACTUAL_HOLDER": "实际控制人",
    "EMP_NUM": "员工人数",
    "ORG_WEB": "公司官网",
    "ADDRESS": "办公地址",
    "MAIN_BUSINESS": "主营业务",
    "BUSINESS_SCOPE": "经营范围",
}


async def fetch_company_profile(symbol: str) -> dict[str, Any]:
    secucode = to_secucode(symbol)
    if not secucode:
        return {"ok": False, "error": unsupported_reason(symbol)}

    rows = await _f10("RPT_F10_BASIC_ORGINFO", secucode, size=1)
    if not rows:
        return {"ok": False, "error": f"没有查到 {symbol} 的公司资料，请确认代码是否正确。"}
    row = rows[0]

    profile = {label: row[key] for key, label in _PROFILE_FIELDS.items() if row.get(key)}
    # 只取日期的年月日，接口返回带 00:00:00
    for k in ("上市日期", "成立日期"):
        if profile.get(k):
            profile[k] = str(profile[k])[:10]

    # 简介全文通常上千字，截断是为了省 token —— 上下文是稀缺资源
    intro = (row.get("ORG_PROFILE") or "").strip()
    if intro:
        profile["公司简介"] = intro[:1200] + ("…（已截断）" if len(intro) > 1200 else "")

    return {
        "ok": True,
        "kind": "profile",
        "data": {
            "code": symbol,
            "name": row.get("SECURITY_NAME_ABBR"),
            "profile": profile,
            "source": "东方财富 F10 · 公司概况",
        },
    }


# --------------------------------------------------------------------------
# 2) 财务与主营构成
# --------------------------------------------------------------------------
_MAINOP_TYPE = {"1": "按行业", "2": "按产品", "3": "按地区", "4": "按销售模式"}

_FIN_FIELDS = [
    ("TOTALOPERATEREVE", "营业总收入", "yi", "TOTALOPERATEREVETZ"),
    ("PARENTNETPROFIT", "归母净利润", "yi", "PARENTNETPROFITTZ"),
    ("EPSJB", "每股收益", "yuan", "EPSJBTZ"),
    ("ROEJQ", "加权净资产收益率", "pct", None),
    ("XSMLL", "销售毛利率", "pct", None),
    ("ZCFZL", "资产负债率", "pct", None),
]


def _scale(value: Any, unit: str) -> Any:
    """元 → 亿元；百分比保留两位小数。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if unit == "yi":
        return round(v / 1e8, 2)
    if unit == "pct":
        return round(v, 2)
    return round(v, 4)


async def fetch_financials(symbol: str) -> dict[str, Any]:
    secucode = to_secucode(symbol)
    if not secucode:
        return {"ok": False, "error": unsupported_reason(symbol)}

    fin_rows = await _f10("RPT_F10_FINANCE_MAINFINADATA", secucode,
                          sort="REPORT_DATE", size=1)
    op_rows = await _f10("RPT_F10_FN_MAINOP", secucode, sort="REPORT_DATE", size=40)

    if not fin_rows and not op_rows:
        return {"ok": False, "error": f"没有查到 {symbol} 的财务数据。"}

    result: dict[str, Any] = {"code": symbol, "source": "东方财富 F10 · 财务分析"}

    if fin_rows:
        row = fin_rows[0]
        result["report_period"] = row.get("REPORT_DATE_NAME") or str(
            row.get("REPORT_DATE") or "")[:10]
        metrics = []
        for key, label, unit, yoy_key in _FIN_FIELDS:
            value = _scale(row.get(key), unit)
            if value is None:
                continue
            item = {"指标": label, "数值": value,
                    "单位": {"yi": "亿元", "pct": "%", "yuan": "元"}[unit]}
            if yoy_key and row.get(yoy_key) is not None:
                item["同比"] = f"{float(row[yoy_key]):+.2f}%"
            metrics.append(item)
        result["metrics"] = metrics

    # 只保留最新报告期，按「按产品 / 按地区」两个维度给出收入结构
    if op_rows:
        latest = op_rows[0].get("REPORT_DATE")
        result["main_business_period"] = str(latest)[:10]
        breakdown: dict[str, list[dict[str, Any]]] = {}
        for row in op_rows:
            if row.get("REPORT_DATE") != latest:
                continue
            dim = _MAINOP_TYPE.get(str(row.get("MAINOP_TYPE")))
            if not dim:
                continue
            income = row.get("MAIN_BUSINESS_INCOME")
            ratio = row.get("MBI_RATIO")
            gross = row.get("GROSS_RPOFIT_RATIO")   # 拼写是接口自带的
            breakdown.setdefault(dim, []).append({
                "项目": row.get("ITEM_NAME"),
                "收入": round(float(income) / 1e8, 2) if income else None,
                "占比": f"{float(ratio) * 100:.2f}%" if ratio is not None else None,
                "毛利率": f"{float(gross) * 100:.2f}%" if gross is not None else None,
            })
        for items in breakdown.values():
            items.sort(key=lambda x: (x["收入"] is None, -(x["收入"] or 0)))
        result["main_business"] = breakdown

    return {"ok": True, "kind": "financials", "data": result}


# --------------------------------------------------------------------------
# 3) 最近公告
# --------------------------------------------------------------------------
async def fetch_announcements(symbol: str, limit: int = 8) -> dict[str, Any]:
    """最近公告标题列表。

    这是块里最接近「真·检索」的一环：拉回来的是一批**真实文档标题**，
    用户可以据此追问「第三条公告说了什么」，模型再决定要不要继续查。
    """
    plain = to_plain_code(symbol)
    if not plain:
        return {"ok": False, "error": unsupported_reason(symbol)}

    limit = max(1, min(int(limit or 8), 20))
    key = f"ann:{plain}:{limit}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    params = {
        "sr": "-1", "page_size": limit, "page_index": 1,
        "ann_type": "A", "client_source": "web", "stock_list": plain,
    }
    try:
        async with httpx.AsyncClient(timeout=10, headers=HEADERS) as client:
            resp = await client.get(ANN_API, params=params)
            payload = resp.json()
    except (httpx.HTTPError, ValueError):
        return {"ok": False, "error": "公告接口暂时不可用，请稍后重试。"}

    items = ((payload.get("data") or {}).get("list")) or []
    if not items:
        return {"ok": False, "error": f"没有取到 {symbol} 的公告。"}

    anns = [
        {
            "标题": it.get("title"),
            "类型": "、".join(c.get("column_name", "") for c in (it.get("columns") or [])) or "未分类",
            "披露时间": (it.get("notice_date") or "")[:10],
        }
        for it in items
    ]
    return _cache_put(key, {
        "ok": True,
        "kind": "announcements",
        "data": {"code": symbol, "count": len(anns), "items": anns,
                 "source": "东方财富 · 公司公告"},
    })


# --------------------------------------------------------------------------
# 4) 标的搜索：把「人话里的名字」变成代码
# --------------------------------------------------------------------------
# 之前这一步是靠一张硬编码的别名表做的，只能认十几个名字。
# 用户问「科创半导体ETF现在多少钱」时表里没有 → 解析出空 → 工具直接哑火。
# 表现是「回答没意义」，根因是**代码识别能力被写死在一张表里**。
#
# 正确做法是把「认名字」也变成一次真实查询：让上游的搜索接口告诉我们代码。
# 这样全市场 2 万多个标的（股票/ETF/指数/可转债）都能被问到。
_MKT_BY_QUOTEID = {
    "1": "sh",                      # 沪市
    "0": "sz",                      # 深市
    "105": "us", "106": "us", "107": "us",   # 美股（NASDAQ/NYSE/AMEX）
    "116": "hk", "128": "hk", "124": "hk",   # 港股
    "90": "bj", "93": "bj",         # 北交所
}

# SecurityTypeName 是上游给的中文分类，直接用来告诉模型「这是什么」
_TYPE_LABEL = {
    "沪A": "股票", "深A": "股票", "北证A": "股票", "京A": "股票",
    "基金": "ETF/基金", "港股": "港股", "美股": "美股", "指数": "指数",
}


def quoteid_to_symbol(quote_id: Any, code: Any) -> str | None:
    """QuoteID（如 '1.600519'、'116.00700'、'105.NVDA'）→ 内部代码（sh600519）。"""
    if not quote_id or not code:
        return None
    head = str(quote_id).split(".")[0].strip()
    plain = str(code).upper()

    # 北交所的指数（899xxx）挂在前缀 0 下，但行情接口只认 bj899050。
    # 「前缀 0 就是深市」这个假设在这里会翻车，而翻车的表现是「查不到数据」。
    if head == "0" and plain.startswith("899"):
        return "bj" + plain

    market = _MKT_BY_QUOTEID.get(head)
    if not market:
        return None
    return market + plain


async def search_security(keyword: str, limit: int = 8) -> list[dict[str, Any]]:
    """按名称/简称/代码搜索标的，返回带代码的候选列表（按相关度）。

    返回的每一项都是模型能直接拿去调用别的工具的东西，这很关键：
    解析的结果必须**回流到上下文**，模型才知道自己刚才问的到底是哪只标的。
    """
    kw = str(keyword or "").strip()
    if not kw:
        return []

    key = f"sug:{kw}:{limit}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        async with httpx.AsyncClient(timeout=8, headers=HEADERS) as client:
            resp = await client.get(SUGGEST_API, params={
                "input": kw, "type": "14", "token": SUGGEST_TOKEN,
            })
            payload = resp.json()
    except (httpx.HTTPError, ValueError):
        return []

    rows = ((payload.get("QuotationCodeTable") or {}).get("Data")) or []
    out: list[dict[str, Any]] = []
    for row in rows:
        symbol = quoteid_to_symbol(row.get("QuoteID"), row.get("Code"))
        if not symbol:
            continue
        type_name = row.get("SecurityTypeName") or ""
        out.append({
            "symbol": symbol,
            "code": row.get("Code"),
            "name": row.get("Name"),
            "type": _TYPE_LABEL.get(type_name, type_name),
            "is_fund": type_name == "基金" or row.get("Classify") == "Fund",
        })
        if len(out) >= limit:
            break

    return _cache_put(key, out)
