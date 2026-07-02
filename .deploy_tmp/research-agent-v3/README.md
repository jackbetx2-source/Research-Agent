# Research Agent Workflow

一个轻量的 research agent，实现下面的流程：

```text
User topic
  -> LLM1: intent understanding
  -> LLM2/LLM3/LLM4: parallel research branches
  -> LLM1: synthesis and final report
```

## 快速开始

1. 安装依赖：

```powershell
pip install -r requirements.txt
```

2. 配置环境变量：

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`，填入你的 API key 和模型名。

3. 运行：

```powershell
python research_agent.py "AI agent 在企业知识库中的应用"
```

启动本地网页：

```powershell
python web_app.py 8000
```

然后打开：

```text
http://127.0.0.1:8000
```

Windows 也可以双击 `start_web.bat` 启动服务。不要直接双击 `web/index.html`
作为正式使用方式，因为 research API 需要本地 Python 服务。

也可以进入交互模式：

```powershell
python research_agent.py
```

保存完整结果到 Markdown：

```powershell
python research_agent.py "AI agent 在企业知识库中的应用" --output outputs/report.md
```

更快地生成短版报告：

```powershell
python research_agent.py "AI agent 在企业知识库中的应用" --fast --quiet --output outputs/report.md
```

## 配置项

- `OPENAI_API_KEY`: 必填，LLM API key。
- `OPENAI_MODEL`: 必填，所有节点默认使用的模型。
- `OPENAI_BASE_URL`: 可选，兼容 OpenAI SDK 的服务地址。
- `INTENT_MODEL`: 可选，意图理解节点专用模型。
- `RESEARCH_MODEL`: 可选，并行研究节点专用模型。
- `SYNTHESIS_MODEL`: 可选，整合节点专用模型。

## 项目结构

- `research_agent.py`: CLI 入口。
- `src/research_agent/workflow.py`: workflow 编排。
- `src/research_agent/llm.py`: OpenAI-compatible LLM client。
- `src/research_agent/prompts.py`: 各节点 prompt。
- `src/research_agent/types.py`: 数据结构。
