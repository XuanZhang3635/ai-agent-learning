"""
检索层的数据源 —— 一个极小的本地公司资料库。

面试知识点：retrieval 的效果，80% 取决于「文档怎么切、怎么检索」，
而不是取决于你用了哪个向量库。这里用最朴素的关键词打分，
好让你一眼看懂检索这件事的本质：

    把 query 拆成词 → 和每篇文档做匹配 → 按相关度排序 → 取前 N 条

生产环境只是把「关键词匹配」换成「向量相似度」或「混合检索 + rerank」，
骨架完全一样。所以先跑通这个，再换向量库，你不会迷路。
"""
import re
from typing import Any

DOCS: list[dict[str, Any]] = [
    {
        "symbol": "sh600519", "name": "贵州茅台",
        "industry": "白酒 · 食品饮料",
        "business": "以茅台酒为核心的高端白酒生产与销售，主力产品为飞天茅台，兼营系列酒。",
        "highlights": "品牌壁垒极强，毛利率长期在 90% 以上；出厂价与终端价之间存在价差，提价空间是核心弹性来源。",
        "risks": "消费需求受宏观与政策影响明显；批价波动会直接冲击渠道信心；产能扩张受产地与周期约束。",
    },
    {
        "symbol": "sz000858", "name": "五粮液",
        "industry": "白酒 · 食品饮料",
        "business": "浓香型白酒龙头，核心产品为第八代五粮液，覆盖高端与次高端价格带。",
        "highlights": "浓香型工艺与窖池资源稀缺，渠道改革持续推进，普五批价是观察指标。",
        "risks": "高端价格带竞争激烈，与茅台存在品牌势能差距；渠道库存压力会影响发货节奏。",
    },
    {
        "symbol": "sz300750", "name": "宁德时代",
        "industry": "电力设备 · 锂电池",
        "business": "动力电池与储能电池的研发、生产与销售，客户覆盖全球主流车企。",
        "highlights": "全球动力电池装机份额领先，规模效应与技术迭代能力强，储能业务是第二增长曲线。",
        "risks": "碳酸锂等原材料价格波动影响毛利；车企自研电池与二线厂商扩产带来份额压力；海外政策与关税存在不确定性。",
    },
    {
        "symbol": "sz002594", "name": "比亚迪",
        "industry": "汽车 · 新能源整车",
        "business": "新能源汽车整车制造，同时覆盖电池、半导体、电子代工等垂直产业链。",
        "highlights": "垂直一体化程度高，成本控制能力强；插混与纯电双线布局，出口占比持续提升。",
        "risks": "国内价格战压缩单车利润；海外市场面临关税与本地化建厂压力；高端化突破仍需验证。",
    },
    {
        "symbol": "sh600036", "name": "招商银行",
        "industry": "银行 · 股份制",
        "business": "零售银行龙头，以财富管理和信用卡业务见长，同时经营对公与金融市场业务。",
        "highlights": "零售客户基础扎实，负债成本低，中间业务收入占比在股份行中领先。",
        "risks": "净息差持续收窄；零售信贷资产质量与房地产敞口需要跟踪；财富管理收入受市场行情影响大。",
    },
    {
        "symbol": "sh601318", "name": "中国平安",
        "industry": "非银金融 · 保险",
        "business": "综合金融集团，业务覆盖寿险、财险、银行、资管与科技板块。",
        "highlights": "代理人渠道改革成效逐步释放，寿险新业务价值是核心观察指标。",
        "risks": "长端利率下行压制投资收益；地产相关敞口的历史包袱仍在消化。",
    },
    {
        "symbol": "sz300059", "name": "东方财富",
        "industry": "非银金融 · 互联网券商",
        "business": "以财经门户与股吧起家，核心是证券经纪与基金代销两大变现路径。",
        "highlights": "流量成本低、用户黏性强，牛市周期中业绩弹性极大。",
        "risks": "业绩与市场成交额高度相关，熊市中会明显承压；佣金率长期下行。",
    },
    {
        "symbol": "hk00700", "name": "腾讯控股",
        "industry": "互联网 · 港股",
        "business": "社交平台（微信、QQ）为底座，业务覆盖游戏、广告、金融科技与企业服务。",
        "highlights": "微信生态的流量护城河极深；视频号广告与游戏出海是主要增量。",
        "risks": "游戏版号与监管政策；宏观消费疲软影响广告主预算；大额资本开支的回报周期。",
    },
    {
        "symbol": "usNVDA", "name": "英伟达",
        "industry": "半导体 · 美股",
        "business": "GPU 与加速计算平台，数据中心业务是当前绝对主力。",
        "highlights": "CUDA 软件生态构成事实标准，AI 训练与推理需求驱动数据中心收入高增。",
        "risks": "客户自研芯片替代；对华出口限制；估值对增速预期极度敏感。",
    },
    {
        "symbol": "usAAPL", "name": "苹果",
        "industry": "消费电子 · 美股",
        "business": "以 iPhone 为核心，覆盖 Mac、iPad、可穿戴设备与高毛利的服务业务。",
        "highlights": "服务业务毛利率高、现金流稳定；生态锁定带来的换机黏性。",
        "risks": "硬件创新周期放缓；供应链与关税风险；部分地区市场份额承压。",
    },
]


def _tokens(text: str) -> list[str]:
    """极简分词：中文按 2-gram 切，英文数字按单词切。

    不做中文分词是因为你现在的目标是看懂检索流程，
    不是为了在分词器上花时间 —— 这一层随时可以替换。
    """
    text = text.lower()
    words = [w for w in re.split(r"[^0-9a-z\u4e00-\u9fff]+", text) if w]
    grams: list[str] = []
    for w in words:
        if w.isascii():
            grams.append(w)
        else:
            grams.append(w)
            grams.extend(w[i:i + 2] for i in range(len(w) - 1))
    return grams


def search(query: str, top_k: int = 2) -> list[dict[str, Any]]:
    """关键词打分检索。命中 name/industry 权重更高，命中正文权重较低。"""
    q = _tokens(query)
    if not q:
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    for doc in DOCS:
        haystack_strong = _tokens(doc["name"] + " " + doc["industry"])
        haystack_weak = _tokens(doc["business"] + " " + doc["highlights"] + " " + doc["risks"])

        score = 0.0
        for t in q:
            if t in haystack_strong:
                score += 3.0
            if t in haystack_weak:
                score += 1.0

        # 直接提到了名字，几乎肯定就是它
        if doc["name"] in query:
            score += 10.0

        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return []

    # 相对阈值：只保留和最高分同一量级的命中。
    # 少了这一步，「是什么」这种疑问词切出来的 2-gram 会把一堆无关文档一起捞进来 ——
    # 这就是「检索召回噪声」最朴素的形态，换成向量检索同样会出现。
    cutoff = scored[0][0] * 0.5
    return [doc for score, doc in scored[:top_k] if score >= cutoff]
