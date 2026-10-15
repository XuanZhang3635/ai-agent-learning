# 股票 Agent · Building Block 01

对应 Anthropic《Building effective agents》里 **"Building blocks, workflows, and agents"** 这一节。

原文对这个 block 的定义只有一句话：

> "The basic building block of agentic systems is an LLM enhanced with
> augmentations such as **retrieval, tools, and memory**. Our current models
> can actively use these capabilities—generating their own search queries,
> selecting appropriate tools, and determining what information to retain."

这个项目就是把这句话写成一个能跑起来的最小系统。**它刻意只做一层**，不含编排、不含多 Agent、不含评估。

---

## 一、怎么跑

```bash
cd stock-agent
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8000
```

然后打开 http://127.0.0.1:8000

**不需要任何 API key**。默认进入 Mock 模式，用规则模拟模型的决策过程，
让你在零成本的情况下看清完整链路。

想接真实模型，设三个环境变量即可（任何 OpenAI 兼容端点都行）：

```bash
export OPENAI_API_KEY=sk-xxxx
export OPENAI_BASE_URL=https://api.deepseek.com/v1
export OPENAI_MODEL=deepseek-chat
```

---

## 二、代码结构和原文的对应关系

| 文件 | 对应原文的哪个概念 | 关键点 |
|---|---|---|
| `tools.py` | **Tools**（+Retrieval 的数据源） | schema 就是写给模型看的 API 文档 |
| `notes.py` | **Retrieval** | 检索质量取决于切分和排序，不取决于向量库 |
| `block.py` | **The building block 本体** | 只做一次往返，不知道循环的存在 |
| `llm.py` | 底层模型（可替换） | 换模型 = 改环境变量 |
| `app.py` | **给 block 套上循环 + Memory** | 套上循环那一刻，才变成 Agent |

---

## 三、三个增强分别落在哪里

### Tools —— `tools.py`

原文的两条实现建议，翻译成代码里的两条规矩：

**规矩一：schema 是写给模型看的 API 文档。**

模型只能靠 `name` / `description` / `parameters` 决定选哪个工具、怎么填参数。
描述含糊，它就选错工具；参数说明不全，它就自己编参数。所以 `description`
不是注释，是接口本身。可以对照 `get_price_history` 的 description ——
里面特意写了「注意这是区间统计，不是实时行情，问当前价格请用 get_stock_quote」，
因为这两个工具最容易被模型混用。

**规矩二：工具要薄，失败要体面。**

`call_tool()` 永远返回字符串，永远不抛异常：

```python
except TypeError as e:
    result = {"ok": False, "error": f"参数不对：{e}。请检查参数名和类型后重试。"}
```

工具报错的正确姿势是**把错误告诉模型**，让它自己换个参数重试；
而不是让异常冒到顶层，把整轮对话打断。

### Retrieval —— `notes.py`

只用了几十行关键词打分。不是为了省事，而是因为：
检索效果 80% 取决于「文档怎么切、怎么检索」，不取决于用了哪个向量库。
先把骨架跑通，再换向量检索或混合检索 + rerank，你不会迷路。

里面有一处 `cutoff = scored[0][0] * 0.5` 的相对阈值 ——
不加这一步，「是什么」这种疑问词切出来的 2-gram 会把无关文档一起捞进来。
这就是检索召回噪声最朴素的形态，换成向量检索同样会出现。

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

再往外套 `for` 循环 + 中断条件，就是 Agent 了 —— `app.py` 里那 5 行 `for` 已经是雏形。

---

## 六、已知限制

- **Mock 模式不是真模型**，它是规则。它证明链路通，但不代表真实模型会做同样决策
- 行情来自腾讯公开接口，**有延迟、非交易所权威数据**，仅供学习
- 记忆存在进程内存里，重启即丢；多进程部署会串会话
- 没有评估集、没有可观测性、没有沙箱 —— 距离生产还很远
- 所有输出都不构成投资建议
