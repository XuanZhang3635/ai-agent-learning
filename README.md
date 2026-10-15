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

想接真实模型（DeepSeek / 通义 / 智谱 / Kimi / 本地 Ollama，任何 OpenAI 兼容端点）：

```bash
cp stock-agent/.env.example stock-agent/.env
# 编辑 stock-agent/.env，填入 DEEPSEEK_API_KEY=sk-xxxx
# 然后重启服务即可
```

`.env` 已在 `.gitignore` 里，不会被提交。

### 数据来源

工具全部对接真实接口，不是样例数据：

| 数据 | 来源 | 覆盖范围 |
|---|---|---|
| 实时行情 / 日线 | 腾讯公开接口 | A 股、港股、美股、指数 |
| 公司概况 / 财务 / 公告 | 东方财富 F10 | 仅 A 股（含北交所） |

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

## 已知限制

- Mock 模式是规则，不是真模型 —— 它证明链路通，不代表真实模型会做同样的决策
- 行情来自公开接口，**有延迟、非交易所权威数据，仅供学习**
- 基本面来自东方财富 F10，**只覆盖 A 股**；港股、美股会明确返回不支持
- 记忆存在进程内存里，重启即丢
- 没有评估集、没有可观测性、没有沙箱 —— 距离生产还很远
- 所有输出都不构成投资建议
