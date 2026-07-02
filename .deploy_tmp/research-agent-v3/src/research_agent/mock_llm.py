from __future__ import annotations

import json


class MockLLMClient:
    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        del model, temperature, max_tokens

        if "Return only valid JSON" in system_prompt:
            topic = user_prompt.replace("User research topic:", "").strip()
            return json.dumps(
                {
                    "summary": f"研究主题是：{topic}",
                    "key_questions": [
                        "这个主题的核心概念是什么？",
                        "目前有哪些主要应用场景？",
                        "落地时有哪些关键收益和限制？",
                        "下一步应该如何验证或实施？",
                    ],
                    "scope": "覆盖背景、应用、证据、风险和下一步建议；不做实时网页检索。",
                },
                ensure_ascii=False,
            )

        if "Context Researcher" in system_prompt:
            return (
                "## 背景与概念\n\n"
                "- 该主题需要先明确目标用户、使用场景和成功标准。\n"
                "- 研究应区分概念解释、技术实现、业务价值和组织落地。\n"
                "- 当前阶段适合先形成问题框架，再补充外部资料和案例。"
            )

        if "Evidence Researcher" in system_prompt:
            return (
                "## 证据与实践\n\n"
                "- 可从典型用例、成本收益、用户反馈和可观测指标四类证据入手。\n"
                "- 实施上建议先选择低风险、高频、边界清楚的任务做 pilot。\n"
                "- 评估指标包括准确率、节省时间、人工接管率和用户满意度。"
            )

        if "Critical Researcher" in system_prompt:
            return (
                "## 风险与盲点\n\n"
                "- 常见风险包括事实错误、权限边界不清、上下文泄露和不可解释决策。\n"
                "- 需要设计人工审核、日志追踪、失败回退和数据隔离机制。\n"
                "- 对开放式任务应避免过度自动化，先把 workflow 约束清楚。"
            )

        return (
            "# Executive summary\n\n"
            "这是一次 mock 运行，证明 workflow 已经按 LLM1 -> 三个并行 LLM -> LLM1 整合的结构执行。\n\n"
            "# Key findings\n\n"
            "- 意图理解节点会先把用户 topic 转成 research brief。\n"
            "- 三个研究节点会并行生成背景、证据和批判视角。\n"
            "- 整合节点会把分支结果汇总成最终报告。\n\n"
            "# Suggested next steps\n\n"
            "配置 `.env` 后即可切换到真实 LLM 调用。"
        )
