"""
回归脚本 —— 把「这个 block 到底还会不会答」变成一条可重复执行的命令。

    python eval.py                # 跑全部用例
    python eval.py 3              # 只跑第 3 条

它做的事和网页完全一样：同一个 block、同一个循环、同一套工具，
只是把入口从浏览器换成了命令行，这样在改完提示词或工具之后，
你能在 30 秒内知道有没有把之前修好的东西又改坏。

面试里这对应的是「评估集（eval set）」：没有它，你只能靠记忆判断
「这次改动是变好了还是变坏了」，而记忆是最不可靠的判据。

注意：这里只做「人工看结果」的冒烟测试，不做自动断言。
自动断言要等到你能稳定定义「什么算正确答案」之后再说 ——
金融问答很多答案本身是浮动的（价格每分钟都在变）。
"""
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

from block import SYSTEM_PROMPT, AugmentedLLM  # noqa: E402
from llm import build_llm, describe_llm  # noqa: E402

# 每条用例都在考一个具体的失败模式，不是随手编的：
#   「问 ETF」考的是标的解析，「毛利率」考的是基本面工具路由，
#   「它」考的是 Memory（信息不在这一句里），「几点」考的是不硬答。
CASES = [
    ("ETF：中文简称能不能解析成真实代码", "科创半导体ETF现在多少钱？"),
    ("公司基本面：工具路由 + 主营构成", "茅台的毛利率和收入构成怎么样"),
    ("指数：区间走势工具", "沪深300最近20个交易日走势"),
    ("指代消解：靠对话历史补全标的", "它今天涨了多少"),
    ("公告：按时间取披露列表", "贵州茅台最近有什么公告"),
    ("边界：模型应该承认自己不知道", "你几点下班？"),
]


async def ask(block: AugmentedLLM, question: str, history: list[dict]) -> tuple[str, list[str], list[dict]]:
    messages = ([{"role": "system", "content": SYSTEM_PROMPT}] + history
                + [{"role": "user", "content": question}])
    used: list[str] = []
    for _ in range(5):                      # 和 app.py 的 MAX_STEPS 保持一致
        result = await block.step(messages)
        for t in result.trace:
            used.append(t["name"])
        if result.finished:
            # 只把「用户说了什么 / 助手答了什么」带回下一轮，
            # 和 app.py 的记忆策略保持一致，这样测出来的指代消解才作数
            return result.content, used, [m for m in messages if m["role"] in ("user", "assistant")]
    return "（超过最大步数仍未收敛）", used, history


async def main() -> None:
    only = int(sys.argv[1]) if len(sys.argv) > 1 else None
    llm = build_llm()
    info = describe_llm(llm)
    print(f"模式：{info['mode']}  模型：{info['model']}\n")

    block = AugmentedLLM(llm)
    history: list[dict] = []
    passed = 0
    total = 0

    for i, (title, question) in enumerate(CASES, 1):
        if only and i != only:
            continue
        total += 1
        print("=" * 72)
        print(f"[{i}] {title}")
        print(f"问：{question}")
        answer, used, history = await ask(block, question, history)
        print(f"工具：{' → '.join(used) if used else '（未调用工具）'}")
        print(f"答：{answer.strip()}")
        if used or i == len(CASES):     # 最后一条本来就该不调工具
            passed += 1
        print()

    print("=" * 72)
    print(f"{passed}/{total} 条符合预期（前 {len(CASES) - 1} 条该调工具，最后 1 条该不调）。")
    print("注意：这个数字只说明「工具被用上了」，不说明答案对。")
    print("答案对不对要你自己读一遍 —— 自动断言留给下一阶段。")


asyncio.run(main())
