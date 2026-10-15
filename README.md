# AI Agent 学习笔记与实战

跟着 Anthropic 的 [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) 一路做下来的学习记录。
理论部分整理成一份可勾选的面试学习地图，实践部分是一个能跑起来的股票 Agent。

## 目录

```
.
├── docs/
│   └── ai-agent-interview-roadmap.html   面试学习地图（浏览器打开，可勾选进度）
└── stock-agent/                          股票 Agent 实战
    ├── tools.py        Tools 层：5 个工具，全部对接真实数据源
    ├── eastmoney.py    基本面数据源：东方财富 F10（公司概况/财务/公告）
    ├── block.py        ★ building block 本体，只做一次模型往返
    ├── llm.py          模型适配：DeepSeek / 任意 OpenAI 兼容端点 / MockLLM
    ├── app.py          Web 服务 + 循环 + Memory
    ├── eval.py         回归脚本：6 条用例冒烟测试，改完代码先跑它
    ├── .env.example    API key 配置模板（复制成 .env 即可）
    ├── static/         对话前端，执行轨迹可展开
    └── README.md       代码逐文件对应回原文概念
```

## 学习路径

原文把 agentic system 分成三层，本仓库严格按这个顺序推进：

| 阶段 | 原文定义 | 状态 |
|---|---|---|
| **Building block** | "an LLM enhanced with augmentations such as retrieval, tools, and memory" | ✅ 已完成 |
| **Workflows** | "LLMs and tools are orchestrated through predefined code paths" | 进行中 |
| **Agents** | "LLMs dynamically direct their own processes and tool usage" | 待开始 |

分界线只有一条：**控制流由谁决定**。写死的叫 workflow，模型自己决定的叫 agent。

## 跑起来

**环境要求：Python ≥ 3.10**（fastapi 0.115+ 要求 ≥3.8，0.129+ 要求 ≥3.10）。

```bash
cd stock-agent
python3 -m venv .venv          # 建议用虚拟环境，别装到全局
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8000
```

打开 http://127.0.0.1:8000

先确认一下 `python3 -V` 是 3.10 以上。如果机器上有多个 Python，
用绝对路径显式指定，比如 macOS 上常见的
`/Library/Frameworks/Python.framework/Versions/3.10/bin/python3 -m venv .venv`。

**不配 API key 也能跑。** 默认走 Mock 模式，用规则模拟模型的决策过程，
零成本就能看清 `schema → 选工具 → 填参数 → 结果写回` 的完整链路。

想接真实模型（智谱 / DeepSeek / 通义 / Kimi / 本地 Ollama，任何 OpenAI 兼容端点）：

```bash
cp stock-agent/.env.example stock-agent/.env
# 编辑 stock-agent/.env：
#   PROV=zhipu
#   ZHIPU_API_KEY=xxxxxxxx.xxxxxxxx
# 然后重启服务即可
```

**推荐先用智谱 `glm-4-flash`**：国内直连、有免费额度、支持 function calling，
拿来学这个 block 完全够用。DeepSeek 便宜但新账号要先充值，
余额为 0 时接口返回 402，和代码无关。

`.env` 已在 `.gitignore` 里，不会被提交。

### 改完代码先跑回归

```bash
python eval.py        # 6 条用例，覆盖 ETF / 基本面 / 指数 / 指代消解 / 公告 / 边界
python eval.py 3      # 只跑第 3 条
```

每条用例都在考一个具体的失败模式，不是随手编的。它做的事和网页完全一样 ——
同一个 block、同一个循环、同一套工具，只是把入口从浏览器换成命令行，
让你在改完提示词或工具之后的 30 秒内知道有没有把之前修好的东西又改坏。

### 数据来源

工具全部对接真实接口，不是样例数据：

| 数据 | 来源 | 覆盖范围 |
|---|---|---|
| 实时行情 / 日线 | 腾讯公开接口 | A 股、港股、美股、指数、ETF/基金 |
| 公司概况 / 财务 / 公告 | 东方财富 F10 | 仅 A 股（含北交所），ETF 无基本面 |
| 标的识别（名称 → 代码） | 东方财富 suggest + 腾讯 smartbox | 全市场两万多个标的，双上游互备 |

**标的识别是这里最关键的一环。** 直接问「科创半导体ETF现在多少钱」「纳指ETF」
「北证50今天表现」都能答上来 —— 因为「认名字」本身也是一次真实查询，
而不是一张手写的别名表。两个上游各有强项（腾讯认中文名，东财认港美股），
并且带相关性闸门：**搜不到就如实说搜不到，不会硬凑一个答案给你。**

## 面试学习地图

`docs/ai-agent-interview-roadmap.html` 用浏览器直接打开，内容包括：

- 六层知识体系（概念地基 → 核心循环 → 四大支柱 → 多智能体 → 工程落地 → 评估度量）
- 12 道高频面试问题及答题要点
- 可直接背下来的 ReAct 代码骨架
- 四周备考计划清单（进度存在浏览器本地）

## 常见问题

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

1. `.env` 要放在 `stock-agent/` 下（和 `app.py` 同级），不是仓库根目录
2. 改完 `.env` 必须**重启服务**
3. 用 `curl http://127.0.0.1:8000/api/config` 看后端自己怎么说的
4. 检查 `.env` 里有没有 `FORCE_MOCK=1`，它会让所有 key 失效

**key 填对了，但回答是「模型账户余额不足」**

上游返回 **402 Insufficient Balance**，和代码无关。DeepSeek 新账号需先充值。
三个选择：充值 / 换一家 key（智谱 `glm-4-flash` 有免费额度）/
先不加 key 用 Mock 模式 —— **Mock 一样能看到完整的工具调用链路**。

装多个 key 时用 `PROV=zhipu` 指定走哪家。

## 已知限制

- Mock 模式是规则，不是真模型 —— 它证明链路通，不代表真实模型会做同样的决策
- 行情来自公开接口，**有延迟、非交易所权威数据，仅供学习**
- 基本面来自东方财富 F10，**只覆盖 A 股**；港股、美股、ETF 会明确返回不支持
- 同名标的可能解析到不是你心里那一只（「半导体ETF」有十几只），
  回答里会点明查到的是哪只；**没有「基金跟踪什么指数」这类工具**
- 记忆存在进程内存里，重启即丢
- `eval.py` 只做冒烟测试（人工看结果），**没有自动断言** ——
  金融问答的正确答案本身是浮动的（价格每分钟都在变），断言要等定义清楚之后再说
- 没有可观测性（trace）、没有沙箱 —— 距离生产还很远
- 所有输出都不构成投资建议
