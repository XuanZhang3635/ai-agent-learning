# 股票 Agent · Building Block 01

对应 Anthropic《Building effective agents》里 **"Building blocks, workflows, and agents"** 这一节。

原文对这个 block 的定义只有一句话：

> "The basic building block of agentic systems is an LLM enhanced with
> augmentations such as **retrieval, tools, and memory**. Our current models
> can actively use these capabilities—generating their own search queries,
> selecting appropriate tools, and determining what information to retain."

这个项目就是把这句话写成一个**真的能用**的最小系统。**它刻意只做一层**，不含编排、不含多 Agent、不含评估。

---

## 一、怎么跑

**环境要求：Python ≥ 3.10**（fastapi 0.115+ 要求 ≥3.8，0.129+ 要求 ≥3.10）。

```bash
cd stock-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8000
```

打开 http://127.0.0.1:8000

### 接真实大模型

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-xxxx
```

然后用同样的命令启动即可。`.env` 已在 `.gitignore` 里，不会被提交。

不配 key 也能跑 —— 会自动退回 Mock 模式（用规则模拟模型决策），
方便你在没有 key 的时候观察完整链路。但**规则不等于模型**，别把 Mock 的表现当成结论。

支持任何 OpenAI 兼容端点（DeepSeek / 通义 / 智谱 / Kimi / 本地 Ollama），
换家只改 `.env`，代码一行不动。

---

## 二、代码结构和原文的对应关系

| 文件 | 对应原文的哪个概念 | 关键点 |
|---|---|---|
| `tools.py` | **Tools** | schema 就是写给模型看的 API 文档 |
| `eastmoney.py` | **Retrieval 的真实数据源** | 数据源归数据源，工具归工具 |
| `block.py` | **The building block 本体** | 只做一次往返，不知道循环的存在 |
| `llm.py` | 底层模型（可替换） | 换模型 = 改 `.env` |
| `app.py` | **给 block 套上循环 + Memory** | 套上循环那一刻，才变成 Agent |

---

## 三、三个增强分别落在哪里

### Tools —— `tools.py`

块里挂了 **5 个工具，全部是真实查询**：

| 工具 | 数据源 | 回答什么问题 |
|---|---|---|
| `get_stock_quote` | 腾讯行情 | 现在多少钱、今天涨跌、估值多少 |
| `get_price_history` | 腾讯日线 | 最近走势、区间涨跌 |
| `get_company_profile` | 东方财富 F10 | 公司做什么、所属行业、何时上市、实控人 |
| `get_financials` | 东方财富 F10 | 业绩如何、营收利润、毛利率、靠什么赚钱 |
| `get_announcements` | 东方财富公告 | 最近有什么消息 |

> **一个关于「假装能用」的教训。**
> 这个块最初的第 3 个工具是一个本地硬编码的资料库 —— 10 家公司，
> 内容是手写的，永远不会更新。它跑得挺好，演示起来也挺像样。
> 但用户问到第 11 家就哑火，问到去年就答不上来。
> 「看起来能跑，一到真实场景就废」是最常见的自欺欺人。
> 后来换成了东方财富的真实接口，代码多了一倍，但**它终于是个能用的东西了**。

原文的两条实现建议，翻译成代码里的两条规矩：

**规矩一：schema 是写给模型看的 API 文档。**

模型只能靠 `name` / `description` / `parameters` 决定选哪个工具、怎么填参数。
描述含糊，它就选错工具；参数说明不全，它就自己编参数。所以 `description`
不是注释，是接口本身。可以对照 `get_price_history` 的 description ——
里面特意写了「注意这是区间统计，不是实时行情，问当前价格请用 get_stock_quote」，
因为这两个工具最容易被模型混用。

同样地，基本面那三个工具的 description 里都标了「只覆盖 A 股」——
不写这一句，模型会在你问腾讯控股时理直气壮地调用它，然后拿到一个报错。

**规矩二：工具要薄，失败要体面。**

`call_tool()` 永远返回字符串，永远不抛异常：

```python
except TypeError as e:
    result = {"ok": False, "error": f"参数不对：{e}。请检查参数名和类型后重试。"}
```

工具报错的正确姿势是**把错误告诉模型**，让它自己换个参数重试；
而不是让异常冒到顶层，把整轮对话打断。

### Retrieval —— `eastmoney.py`

这一层现在做的是**真实的外部数据查询**，而不是在本地小样本里检索。

不过在实现上，它和「检索」要解决的问题是同一个：**上游返回的原始结构，
不能直接丢给模型。** 所以 `eastmoney.py` 做了三件事：

1. **归一化** —— 接口要 `600519.SH`，模型可能给你 `贵州茅台`、`600519` 或 `sh600519`
2. **裁剪与换算** —— 字段从 45 个压到 11 个，元换算成亿元，简介截断到 1200 字
3. **缓存** —— 模型经常连着问「茅台主营 + 茅台财务」，不缓存就会把同一接口打两遍

第 2 条是这一层的核心。上下文是稀缺资源：把一个 45 字段的原始 JSON 塞进去，
模型要自己找关键字段、自己做单位换算、自己判断哪个数字有用 —— 每一步都是出错机会。
**能在代码里算好的，就不要让模型在上下文里做。**

### Memory —— `app.py`

```python
history.append({"role": "user", "content": req.message})
history.append({"role": "assistant", "content": reply})
del history[:-HISTORY_TURNS]
```

注意这里**只存了用户的提问和助手的最终回答**，中间的 `tool_calls` 和
工具原始返回全部丢弃。理由有两条：

1. 工具原始返回体积很大，留在历史里会迅速吃掉上下文预算；
2. 结论已经体现在助手的回答里，原始数据下一轮几乎用不上。

「什么该进记忆、什么该丢」的答案永远是：**看它下一轮还有没有用**。

想验证 Memory 真的在工作，连着问两句，第二句问「我上一句问的是哪只股票？」。

---

## 四、为什么必须先只做这一层

因为接下来无论你走 Workflow 还是 Agent，所有问题都会先在这里暴露：

- 工具选错 → 是 schema 描述的问题，不是编排的问题
- 参数编造 → 是参数说明的问题
- **参数填对了但工具报错** → 是归一化漏了一处（这个 bug 最难查，两边看起来都没错）
- 回答里出现不存在的数字 → 是 system prompt 没约束住，还是工具给了空数据
- 到第 3 步就乱了 → 是记忆策略的问题

如果一上来就把循环、多 Agent、评估全堆上，出了问题你分不清是哪一层。

---

## 五、接下来往哪走

按原文的顺序，下一步是 **Workflow 的五种模式**。对股票 Agent 来说，最容易落地的是：

1. **Prompt chaining** —— 先「解析意图」再「取数」再「生成结论」，每步之间加校验闸门
2. **Routing** —— 把提问分流到「行情问答 / 基本面问答 / 复盘分析」三条专用链
3. **Parallelization** —— 多只标的的行情并行取，再聚合对比
4. **Orchestrator-workers** —— 主管模型动态拆解「帮我看看新能源板块」，分派给多个 worker
5. **Evaluator-optimizer** —— 一个模型写分析，另一个模型挑毛病（对投资类内容特别有用）

再往外套 `for` 循环 + 中断条件，就是 Agent 了 —— `app.py` 里那几行 `for` 已经是雏形。

---

## 六、常见问题

**`ERROR: No matching distribution found for fastapi>=0.115`**

这不是网络问题，是 **Python 版本太旧**。fastapi 从 0.104 起要求 Python ≥3.8，
0.129 起要求 ≥3.10，pip 会把不满足 `Requires-Python` 的版本全部过滤掉，
于是报成"找不到"。先跑 `python3 -V` 确认版本，用 3.10+ 重建虚拟环境即可。

想看清真相，pip 其实把原因写在报错上面一行（容易被刷屏淹没）：

```
Ignored the following versions that require a different python version:
  0.104.0 Requires-Python >=3.8; ... 0.143.1 Requires-Python >=3.10
```

**装完但 `python -m uvicorn` 提示找不到模块**

确认你用的 `python` 和 `pip` 来自同一个环境：`which python` / `pip -V`。
混用全局 pip 和虚拟环境解释器是高频坑。

**页面显示 mock 模式，但我明明配了 key**

1. `.env` 要放在 `stock-agent/` 目录下（和 `app.py` 同级），不是仓库根目录
2. 改完 `.env` 必须**重启服务**（除非你用了 `--reload`）
3. 用 `curl http://127.0.0.1:8000/api/config` 看后端自己怎么说的

---

## 七、已知限制

- **Mock 模式不是真模型**，它是规则。它证明链路通，但不代表真实模型会做同样决策
- 行情来自腾讯公开接口，**有延迟、非交易所权威数据**，仅供学习
- 基本面来自东方财富 F10，**只覆盖 A 股**；港股、美股会明确返回不支持
- `get_announcements` 只返回公告标题，不含正文
- 记忆存在进程内存里，重启即丢；多进程部署会串会话
- 没有评估集、没有可观测性、没有沙箱 —— 距离生产还很远
- 所有输出都不构成投资建议
