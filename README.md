# AI Agent 学习笔记与实战

跟着 Anthropic 的 [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) 一路做下来的学习记录。
理论部分整理成一份可勾选的面试学习地图，实践部分是一个能跑起来的股票 Agent。

## 目录

```
.
├── docs/
│   └── ai-agent-interview-roadmap.html   面试学习地图（浏览器打开，可勾选进度）
└── stock-agent/                          股票 Agent 实战
    ├── tools.py        Tools 层：3 个工具 + 多市场行情字段解析
    ├── notes.py        Retrieval 层：本地资料库 + 关键词检索
    ├── block.py        ★ building block 本体，只做一次模型往返
    ├── llm.py          模型适配：OpenAI 兼容端点 / MockLLM 双实现
    ├── app.py          Web 服务 + 循环 + Memory
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

```bash
cd stock-agent
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8000
```

打开 http://127.0.0.1:8000

**不需要任何 API key**。默认走 Mock 模式，用规则模拟模型的决策过程，
零成本就能看清 `schema → 选工具 → 填参数 → 结果写回` 的完整链路。

想接真实模型，设三个环境变量即可（任何 OpenAI 兼容端点都行）：

```bash
export OPENAI_API_KEY=sk-xxxx
export OPENAI_BASE_URL=https://api.deepseek.com/v1
export OPENAI_MODEL=deepseek-chat
```

## 面试学习地图

`docs/ai-agent-interview-roadmap.html` 用浏览器直接打开，内容包括：

- 六层知识体系（概念地基 → 核心循环 → 四大支柱 → 多智能体 → 工程落地 → 评估度量）
- 12 道高频面试问题及答题要点
- 可直接背下来的 ReAct 代码骨架
- 四周备考计划清单（进度存在浏览器本地）

## 已知限制

- Mock 模式是规则，不是真模型 —— 它证明链路通，不代表真实模型会做同样的决策
- 行情来自公开接口，**有延迟、非交易所权威数据，仅供学习**
- 记忆存在进程内存里，重启即丢
- 没有评估集、没有可观测性、没有沙箱 —— 距离生产还很远
- 所有输出都不构成投资建议
