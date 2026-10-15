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
import asyncio
import json
import re
import time
from typing import Any

import httpx

from eastmoney import search_security

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

    # 5) 纯字母**不再**在这里赌是美股代码。
    #    早期版本有这一步，代价是把 "xyz"、随机缩写全都当成美股去查，
    #    行情接口还真的会返回东西（XYZ 是 Block Inc.）——
    #    于是「完全不存在的标的」被回答成一家真公司，这是最坏的一类错误。
    #    现在交给下面的真实搜索接口：搜得到就是搜得到。
    return found[:3]


# --------------------------------------------------------------------------
# 动态识别：别名表认不出来的，就去真实搜索接口查
# --------------------------------------------------------------------------
# 别名表只能保「快路径」——常见标的不必多打一次网络请求。
# 但它绝不能是唯一路径：一旦它成为唯一路径，
# 用户问「科创半导体ETF」「纳指ETF」「黄金ETF」就全部哑火，
# 而这类问题在真实使用里占的比例非常高。
#
# 所以真正的解析器是两级的：
#   本地别名/代码 → 命中就返回（零成本）
#   否则 → 抽关键词 → 调搜索接口 → 拿到真实代码
_NOISE_WORDS = [
    # 尽量按长度从长到短替换，否则「多少钱」会被「多少」先切坏
    "多少钱", "什么价", "哪一个", "怎么样", "是什么", "有没有", "帮我", "请问",
    "哪一只", "哪一只", "哪只票", "哪家好", "跌得多", "涨得多", "跌得少", "涨得少",
    "现在", "目前", "今天", "今日", "最近", "近期", "这几天", "这几天", "这周", "本周",
    "价格", "股价", "报价", "行情", "估值", "市盈率", "市净率", "市值", "换手率", "换手",
    "成交量", "成交额", "成交", "走势", "趋势", "日线", "区间", "涨幅", "跌幅", "涨了",
    "跌了", "表现", "公告", "消息", "新闻", "披露", "主营", "业务", "财报", "业绩",
    "营收", "收入", "净利润", "利润", "毛利", "负债率", "构成", "占比", "行业", "简介",
    "介绍", "上市", "实控人", "老板", "员工", "公司", "股票", "基金", "板块", "对比",
    "比较", "分析", "看一下", "看看", "一下", "多少", "为什么", "怎样", "如何", "哪个",
    "哪只", "哪支", "哪家", "哪些", "跌得", "涨得",
    # 泛指的类别词。这些词本身**不是**标的名称，但市场里总有一堆东西名字里带它们：
    # 「指数」能搜出上证指数、「基金」能搜出一堆基金 —— 把它们留在候选里，
    # 结果是「它跟踪的是什么指数」被解析成了上证指数，指代消解直接被顶掉。
    "指数", "跟踪", "标的", "龙头", "个股", "市场", "代码", "什么",
    "交易日", "涨跌", "走势", "跌", "涨", "的", "了", "吗", "呢", "啊", "和", "与", "跟", "是",
]

# 名称 token：必须以中文或字母开头，但**要允许带上数字**。
# 早期版本写成 [\u4e00-\u9fffA-Za-z]{2,}，把数字排除在外，
# 「北证50」就被切成「北证」——搜出来的是东北证券。这类切词错误非常隐蔽：
# 代码跑得通、结果也「像那么回事」，但它答的根本不是你问的东西。
_TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9]{1,}")


def _name_candidates(text: str) -> list[str]:
    """从问句里抽出「可能是标的名称」的片段，长的优先。

    这是个启发式：把问题词剥掉，剩下的连续中文/字母串就当候选名。
    允许它不够精确 —— 因为后面还有一层真实搜索接口在兜底，
    搜不到就是搜不到，不会编造结果。真正危险的是「猜错还硬答」。
    """
    t = str(text or "")
    for w in sorted(set(_NOISE_WORDS), key=len, reverse=True):
        t = t.replace(w, " ")

    toks = _TOKEN_RE.findall(t)
    toks = [x for x in toks if x.upper() not in _TICKER_STOPWORDS]
    toks.sort(key=len, reverse=True)
    return toks[:3]


# --------------------------------------------------------------------------
# 第二个搜索上游：腾讯 smartbox
# --------------------------------------------------------------------------
# 为什么要有两个上游？因为它们的强项不一样，这不是冗余而是互补：
#   东方财富 suggest —— 港美股和指数准，但中文 ETF 名字覆盖很差
#                      （搜「半导体ETF」只给美股 SOXX，搜「沪深300ETF」给港股）
#   腾讯 smartbox    —— 中文名称/简称覆盖极好，A 股 ETF 一搜一个准
# 只用一家，就会在某些类别的标的上系统性地答错 —— 这比"查不到"更糟。
SMARTBOX_API = "https://smartbox.gtimg.cn/s3/"

# smartbox 的类型码：GP=股票 jj=场外基金 ETF/QDII-ETF/LOF=场内基金 ZS=指数
_TENCENT_TYPE = {
    "GP": ("股票", False), "GP-A": ("股票", False), "GP-B": ("股票", False),
    "ETF": ("ETF/基金", True), "QDII-ETF": ("ETF/基金", True), "LOF": ("ETF/基金", True),
    "ZS": ("指数", False),
}
# 这些市场/类型我们的行情接口取不到，直接不纳入候选
_TENCENT_SKIP_MARKET = {"jj"}      # 场外基金：只有净值，没有实时行情


def _unescape_unicode(s: str) -> str:
    """smartbox 的中文是 \\uXXXX 转义的 ASCII，得手动还原。"""
    return re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s)


async def _search_tencent(keyword: str, limit: int = 6) -> list[dict[str, Any]]:
    kw = str(keyword or "").strip()
    if not kw:
        return []

    key = f"txsug:{kw}:{limit}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(SMARTBOX_API, params={"q": kw, "t": "all"})
            text = resp.content.decode("gbk", errors="replace")
    except httpx.HTTPError:
        return []

    if '="' not in text:
        return []
    body = text.split('="', 1)[1].rstrip('";\n')

    out: list[dict[str, Any]] = []
    for entry in body.split("^"):
        f = entry.split("~")
        if len(f) < 5:
            continue
        market, code, name, type_code = f[0].lower(), f[1], _unescape_unicode(f[2]), f[4]
        if market in _TENCENT_SKIP_MARKET:
            continue
        if market == "us":
            code = code.split(".")[0]          # nvda.oq → NVDA
        label, is_fund = _TENCENT_TYPE.get(type_code, ("其他", False))
        out.append({
            "symbol": market + str(code).upper(),
            "code": str(code).upper(),
            "name": name,
            "type": label,
            "is_fund": is_fund,
        })
        if len(out) >= limit:
            break

    return _cache_put(key, out)


# --------------------------------------------------------------------------
# 相关性排序：两个上游各给一堆候选，得挑出「用户真正说的那个」
# --------------------------------------------------------------------------
_MARKET_HINTS = [
    (re.compile(r"港股|H股|港交所"), "hk"),
    (re.compile(r"美股|纳斯达克|标普|道琼斯|纽交所"), "us"),
]


def _preferred_market(text: str) -> str | None:
    """用户明确提了某个市场，就按他说的来。"""
    for pattern, market in _MARKET_HINTS:
        if pattern.search(text):
            return market
    return None


def _dedupe(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """两个上游会返回同一只标的，合并掉（is_fund 取或）。"""
    merged: dict[str, dict[str, Any]] = {}
    for hit in hits:
        cur = merged.get(hit["symbol"])
        if cur is None:
            merged[hit["symbol"]] = dict(hit)
        else:
            cur["is_fund"] = cur.get("is_fund") or hit.get("is_fund")
            if not cur.get("name"):
                cur["name"] = hit.get("name")
    return list(merged.values())


def _rank_hits(hits: list[dict[str, Any]], keyword: str,
               text: str = "") -> list[dict[str, Any]]:
    """按「和用户说的名字有多像」打分排序。

    这里最重要的一条是**相关性闸门**：字面覆盖率太低的候选直接丢掉。
    不设闸门会怎样？实测过 —— 搜「北证50」会返回「东北证券」，
    搜「完全不存在的标的xyz」会返回 Block Inc.。
    一个查不到就如实说查不到的工具，远比一个"总能给出点什么"的工具可用。
    """
    kw = keyword.lower().strip()
    core = set(re.findall(r"[0-9a-z\u4e00-\u9fff]", kw))
    prefer = _preferred_market(text)
    wants_fund = bool(re.search(r"etf|基金|lof", kw, re.I))

    scored: list[tuple[float, dict[str, Any]]] = []
    for idx, hit in enumerate(hits):
        name = (hit.get("name") or "").lower()
        if not name:
            continue

        cover = sum(1 for ch in core if ch in name) / len(core) if core else 0.0
        if not (kw in name or name in kw or cover >= 0.6):
            continue

        score = cover * 30
        if name == kw:
            score += 100
        elif kw in name:
            score += 60
        elif name in kw:
            score += 40

        market = hit["symbol"][:2]
        if prefer == market:
            score += 50
        elif prefer is None and market in ("sh", "sz", "bj"):
            # 没明确指定市场时，默认偏向 A 股 ——
            # 中文语境下问「黄金ETF」绝大多数是在问 518880，不是美股 GLD
            score += 25
        if wants_fund and hit.get("is_fund"):
            score += 40

        score -= idx * 1.5        # 两个上游本身都按相关度排序，顺序是有信息的
        scored.append((score, hit))

    scored.sort(key=lambda x: -x[0])
    return [hit for _, hit in scored]


def _alias_matched_partially(text: str, local: list[str]) -> bool:
    """本地别名是不是只匹配到了「更长名字的一部分」。

    例：「沪深300ETF」里有「沪深300」这个别名，
    走本地快路径会解析成 sh000300（指数），而用户要的是 510300（ETF）。
    这种情况必须让位给真实搜索。
    """
    toks = _name_candidates(text)
    for name, code in ALIASES.items():
        if code not in local or name not in text:
            continue
        for tok in toks:
            if tok != name and name in tok:
                return True
    return False


_TRIM_ATTEMPTS = 5


async def _lookup(candidate: str, text: str) -> dict[str, Any] | None:
    """查一个候选词，返回最相关的一条；查不到就砍掉尾巴再试。

    为什么要「砍尾巴」？因为抽词是启发式的，问句的尾巴经常粘在名字上：
        「半导体ETF和科创半导体ETF哪个跌得多」→ 抽出来是
        「科创半导体ETF哪个跌得多」——这不是任何标的的名字，直接查必然为空。

    与其把噪声词表维护到完美（永远做不到，中文问法无穷多），
    不如承认抽词会出错，然后用「逐步截短 + 每步都校验相关性」兜住它。
    这类「宁可多试几次也不猜」的兜底，是把工具做成能用的关键。
    """
    for attempt in range(_TRIM_ATTEMPTS + 1):
        kw = candidate[:len(candidate) - attempt] if attempt else candidate
        if len(kw) < 2:
            break

        groups = await asyncio.gather(
            _search_tencent(kw), search_security(kw), return_exceptions=True,
        )
        merged = [h for g in groups if not isinstance(g, Exception) and g for h in g]
        ranked = _rank_hits(_dedupe(merged), kw, text)
        if ranked:
            return ranked[0]
    return None


async def resolve_symbols_async(text: str) -> list[str]:
    """真正的解析器：本地快路径 + 双上游真实搜索。

    为什么必须是异步的？因为「认名字」这件事本身需要联网。
    早期版本坚持纯本地的同步别名表，代价就是只能认识十几个标的，
    用户问「科创半导体ETF」直接哑火 —— 而这不是边缘场景，是日常场景。
    """
    local = resolve_symbols(text)
    if local and not _alias_matched_partially(text, local):
        return local

    candidates = _name_candidates(text)
    if not candidates:
        return local

    # 多个候选词并行查（「A 和 B 哪个好」这种问法要能同时解析出两只）
    hits = await asyncio.gather(*[_lookup(c, text) for c in candidates])

    found: list[str] = []
    for hit in hits:
        if not hit:
            continue
        remember_symbol(hit["symbol"], hit)
        if hit["symbol"] not in found:
            found.append(hit["symbol"])

    return found[:3] or local


# --------------------------------------------------------------------------
# 解析结果的本地状态
# --------------------------------------------------------------------------
# 同名标的会被反复问到（「茅台」「贵州茅台」都会被问很多次），
# 不缓存就是每次都多打两轮搜索。搜索结果的时效性要求很低，缓存久一点没关系。
_SEARCH_CACHE: dict[str, tuple[float, Any]] = {}
_SEARCH_TTL = 1800

# symbol -> 搜索接口给出的元信息（叫什么、是不是基金）
_META_CACHE: dict[str, dict[str, Any]] = {}


def _cache_get(key: str) -> Any | None:
    hit = _SEARCH_CACHE.get(key)
    if hit and time.time() - hit[0] < _SEARCH_TTL:
        return hit[1]
    return None


def _cache_put(key: str, value: Any) -> Any:
    _SEARCH_CACHE[key] = (time.time(), value)
    return value


def _is_fund(symbol: str) -> bool:
    """判断是不是基金/ETF。

    搜索接口的元信息最准；拿不到时用代码规则兜底：
    沪市 5 开头、深市 1 开头且 6 位的都是基金（ETF/LOF）。
    """
    meta = _META_CACHE.get(symbol)
    if meta is not None:
        return bool(meta.get("is_fund"))
    prefix, rest = symbol[:2].lower(), symbol[2:]
    return prefix in ("sh", "sz") and len(rest) == 6 and rest[0] in "51"


def remember_symbol(symbol: str, meta: dict[str, Any]) -> None:
    _META_CACHE[symbol] = meta


def resolved_meta(symbol: str) -> dict[str, Any] | None:
    return _META_CACHE.get(symbol)


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
    symbol = prefix + str(f[2]).upper()
    fund = _is_fund(symbol)
    # 基金和指数没有市净率这个概念，上游给的是 0 —— 把这个 0 显示出来
    # 会让人以为「市净率等于 0」，比不显示更糟。
    # 顺便：真实标的的市净率不可能是 0，所以 0 一律当「没有」处理。
    raw_pb = num(46) if prefix in ("sh", "sz", "bj") else None
    pb = raw_pb if (raw_pb and not fund) else None

    return {
        "code": f[2],
        "name": f[1],
        # ETF / 基金不是「股票」，而且它的「总市值」其实是基金规模 ——
        # 标签写错，模型就会拿基金规模去讲「公司市值」，读起来就是外行话
        "market": _type_label(symbol, meta),
        "security_type": _type_label(symbol, meta),
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
        "cap_label": "基金规模" if fund else "总市值",
        "updated_at": f[30] if len(f) > 30 else None,
    }


def _is_index(symbol: str) -> bool:
    """主要指数：沪 000xxx / 深 399xxx / 北证 899xxx，以及搜索接口标为「指数」的。"""
    meta = _META_CACHE.get(symbol)
    if meta and meta.get("type") == "指数":
        return True
    prefix, rest = symbol[:2].lower(), symbol[2:]
    if len(rest) != 6:
        return False
    return (
        (prefix == "sh" and rest.startswith("000")) or
        (prefix == "sz" and rest.startswith("399")) or
        (prefix == "bj" and rest.startswith("899"))
    )


def _type_label(symbol: str, meta: dict[str, str]) -> str:
    """标的类型标签。股票 / ETF/基金 / 指数 / 港股 / 美股 —— 三处共用一份逻辑。"""
    if _is_fund(symbol):
        return "ETF/基金"
    if _is_index(symbol):
        return "指数"
    return meta.get("market", "股票")


async def get_stock_quote(symbol: str) -> dict[str, Any]:
    """取实时行情（A 股 / 港股 / 美股 / 指数 / ETF）。"""
    codes = await resolve_symbols_async(str(symbol))
    if not codes:
        return {"ok": False, "error": f"没能在全市场里找到叫 {symbol!r} 的标的。"
                                      f"可以换成 6 位代码（如 600519）或更完整的名称"
                                      f"（如 '科创半导体ETF'、'沪深300ETF'）再试一次。"}

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
    codes = await resolve_symbols_async(str(symbol))
    if not codes:
        return {"ok": False, "error": f"没能在全市场里找到叫 {symbol!r} 的标的。"}

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

    # 名称就藏在同一个响应里的 qt 节点中，顺手取出来 —— 白拿的，不用多打一次请求。
    # 少了它，回答里只能写「sh588170（ETF/基金）」，用户根本认不出是哪只。
    name = code
    quotes = node.get("qt") or {}
    row_list = quotes.get(code) or []
    if isinstance(row_list, list) and len(row_list) > 1 and row_list[1]:
        name = str(row_list[1])

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
            "name": name,
            "market": _type_label(code, meta),
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
            # 上游对部分标的（例如北交所指数）只给很少几根日线。
            # 这必须说出来 —— 否则「最近20个交易日」实际只有 1 天，
            # 模型却会照着自己的假设讲成「20 天的区间表现」。
            "note": (f"上游只返回了 {len(rows)} 个交易日（请求的是 {days} 天），"
                     f"该标的日线数据可能不全，区间结论请谨慎使用。"
                     if len(rows) < days else None),
        },
    }


# --------------------------------------------------------------------------
# 工具 3-5：基本面三件套（真实查询东方财富 F10）
# --------------------------------------------------------------------------
# 这三个工具替代了最初的「本地硬编码资料库」。
# 替换的理由很实际：硬编码的资料库只有 10 家公司、永远不会更新，
# 用户问第 11 家就哑火 —— 这种「看起来能跑但一到真实场景就废」的实现，
# 是做一个能用的 block 时最常见的自欺欺人。
#
# 注意每个工具的第一行：**必须先用 resolve_symbols 归一化**。
# 模型传进来的可能是「贵州茅台」「600519」「sh600519」中的任意一种，
# 而下游接口只认其中一种。归一化漏掉一处的直接后果就是：
# 模型参数填得完全正确，工具却报错 —— 这种 bug 最难查，因为两边看起来都没错。
async def _one_symbol(raw: str, *, need_company: bool = False
                      ) -> tuple[str | None, dict[str, Any] | None]:
    """归一化单个标的；失败时返回一句模型能看懂的错误。

    need_company=True 时，如果解析出来的是 ETF/基金，就直接回一句
    「基金没有公司基本面」——这是「失败要体面」的具体做法：
    与其让上游返回一堆空字段、诱导模型去编，不如明确说清边界。
    """
    codes = await resolve_symbols_async(str(raw))
    if not codes:
        return None, {
            "ok": False,
            "error": f"没能在全市场里找到叫 {raw!r} 的标的。"
                     f"可以换成 6 位代码（如 600519）或更完整的名称"
                     f"（如 '科创半导体ETF'、'沪深300ETF'）再试一次。",
        }

    code = codes[0]
    if need_company and _is_fund(code):
        meta = resolved_meta(code)
        name = (meta or {}).get("name") or code
        return None, {
            "ok": False,
            "error": f"{name}（{code}）是 ETF/基金，没有公司基本面资料 —— "
                     f"基金不披露营收、净利润、实控人这些字段。"
                     f"这只标的能用的是行情与走势两类工具（get_stock_quote / "
                     f"get_price_history）；如果你想了解它跟踪的标的，可以直接说明，"
                     f"我去查对应指数的成分股。",
        }
    return code, None


def _with_name(result: dict[str, Any], symbol: str) -> dict[str, Any]:
    """把「这是哪只标的」补进工具返回里。

    看着是小事，其实是可用性的关键：模型拿到的数据里如果只有代码，
    它写出来的回答就只能是代码；用户看到「sh588170 的营收」是没法用的。
    数据源没给名字时，就用解析阶段已经拿到的名字补上。
    """
    if result.get("ok") and isinstance(result.get("data"), dict):
        meta = resolved_meta(symbol) or {}
        if meta.get("name"):
            result["data"].setdefault("name", meta["name"])
    return result


async def get_company_profile(symbol: str) -> dict[str, Any]:
    """公司概况：全称、所属行业、上市信息、实控人、主营与经营范围、公司简介。"""
    code, err = await _one_symbol(symbol, need_company=True)
    if err:
        return err
    from eastmoney import fetch_company_profile
    return _with_name(await fetch_company_profile(code), code)


async def get_financials(symbol: str) -> dict[str, Any]:
    """财务与经营结构：最新报告期的营收/净利/ROE/毛利率/负债率，以及主营构成拆分。"""
    code, err = await _one_symbol(symbol, need_company=True)
    if err:
        return err
    from eastmoney import fetch_financials
    return _with_name(await fetch_financials(code), code)


async def get_announcements(symbol: str, limit: int = 8) -> dict[str, Any]:
    """最近公告标题列表。"""
    code, err = await _one_symbol(symbol)
    if err:
        return err
    from eastmoney import fetch_announcements
    return _with_name(await fetch_announcements(code, int(limit or 8)), code)


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
                "支持 A 股、港股、美股、主要指数，以及 ETF/基金（如「科创半导体ETF」）。"
                "symbol 可以直接写中文名称，工具会自动解析成代码。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "标的名称或代码，如 '贵州茅台'、'600519'、'科创半导体ETF'、'hk00700'、'usAAPL'。多个标的用逗号分隔。",
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
                "支持股票、指数与 ETF。"
                "注意这是区间统计，不是实时行情；问当前价格请用 get_stock_quote。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "标的名称或代码，如 '宁德时代'、'300750' 或 '科创半导体ETF'。"},
                    "days": {"type": "integer", "description": "回看的交易日数量，默认 20，范围 5-120。"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_profile",
            "description": (
                "查询公司的静态基本面资料：公司全称、所属行业、上市日期、实际控制人、"
                "员工人数、主营业务、经营范围，以及公司简介全文。"
                "用户问「这家公司是做什么的」「属于什么行业」「什么时候上市的」"
                "「实际控制人是谁」时使用。"
                "注意：只覆盖 A 股上市公司（含北交所）；港股、美股、ETF/基金都不支持，"
                "它们没有对应的公司资料。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "公司名称或 6 位代码，如 '贵州茅台' 或 '600519'。只接受公司，不接受 ETF。"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_financials",
            "description": (
                "查询公司最新报告期的财务表现与收入结构，返回营业总收入、归母净利润、"
                "每股收益、净资产收益率、销售毛利率、资产负债率（均含同比），"
                "以及按产品/按地区拆分的主营构成（收入、占比、毛利率）。"
                "用户问「赚不赚钱」「业绩如何」「靠什么赚钱」「毛利率多少」"
                "「哪块业务贡献最大」时使用。"
                "注意：只覆盖 A 股上市公司；这是财报数据，不是股价数据。"
                "ETF/基金没有财报，问基金请改用行情与走势工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "公司名称或 6 位代码，如 '宁德时代' 或 '300750'。只接受公司，不接受 ETF。"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_announcements",
            "description": (
                "查询公司最近披露的公告标题列表，返回标题、公告类型和披露日期。"
                "用户问「最近有什么消息」「有没有发公告」「最近发生了什么」时使用。"
                "注意：只返回标题，不含正文内容；只覆盖 A 股。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "标的名称或 6 位代码，如 '贵州茅台'。"},
                    "limit": {"type": "integer", "description": "返回条数，默认 8，范围 1-20。"},
                },
                "required": ["symbol"],
            },
        },
    },
]

# name -> 可调用对象
TOOL_REGISTRY = {
    "get_stock_quote": get_stock_quote,
    "get_price_history": get_price_history,
    "get_company_profile": get_company_profile,
    "get_financials": get_financials,
    "get_announcements": get_announcements,
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
