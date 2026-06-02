# Research Agent

轻量研究 agent，支持三类工作流：

- **研究工作台**：输入主题后，系统先理解意图，再并行生成研究分支，最后合成 Markdown 报告。
- **文献助手**：提交 DOI、arXiv 链接、论文页面链接或 PDF，系统补全元数据并生成结构化文献分析表。
- **系统性文献综述 SLR**：输入综述主题后，系统会聚合 arXiv、OpenAlex、PubMed 和 Crossref，随后批量抽取论文信息、综合主题/共识/分歧/研究空白，并输出 APA、IEEE 或 BibTeX 格式引用。

## 快速开始

安装依赖：

```powershell
pip install -r requirements.txt
```

创建配置文件：

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`，填入模型服务配置：

```text
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=your_model
OPENAI_BASE_URL=https://your-compatible-endpoint/v1
WEB_HOST=0.0.0.0
RESEARCH_AGENT_CONTACT_EMAIL=admin@example.com
```

启动 Web 服务：

```powershell
python web_app.py 8000
```

开发环境可打开：

```text
http://127.0.0.1:8000
```

部署到服务器时，可通过服务器域名、反向代理地址或服务器 IP 访问；服务默认监听 `0.0.0.0`，也可以通过 `WEB_HOST` 指定监听地址。

也可以使用 CLI：

```powershell
python research_agent.py "AI agent 在企业知识库中的应用" --fast --output outputs/report.md
```

## SLR 模式

网页中的“系统综述”页面支持：

- 主题输入
- 论文数量：10 / 20 / 30 / 50
- arXiv 分类过滤，例如 `cs.CL`、`cs.CV`、`stat.ML`
- 日期范围
- 引用格式：APA / IEEE / BibTeX

输出报告会保存到 `outputs/slr_<timestamp>_<topic>.md`。

检索源说明：

- arXiv：适合 CS、AI、物理、数学等预印本。
- OpenAlex：开放学术索引，覆盖面广。
- PubMed：适合医学、生物和生命科学主题。
- Crossref： DOI 和出版物元数据兜底来源。

可选环境变量：

```text
OPENALEX_EMAIL=you@example.com
NCBI_EMAIL=you@example.com
NCBI_API_KEY=optional_ncbi_key
```

## 项目结构

```text
research_agent.py                  CLI 入口
web_app.py                         Web API 和静态文件服务
web/                               前端页面
src/research_agent/workflow.py     普通研究 workflow
src/research_agent/literature_workflow.py
                                   给定文献分析 workflow
src/research_agent/slr_workflow.py 系统性文献综述 workflow
src/research_agent/arxiv_search.py arXiv 检索
src/research_agent/crossref_search.py
                                   Crossref 备用检索
src/research_agent/citations.py    APA / IEEE / BibTeX 引用格式化
src/research_agent/doi.py          DOI、arXiv、网页元数据补全
```

## 说明

SLR 模式会搜索多个开放学术源，并按 DOI、URL 和标题去重。某个来源如果超时或限流，系统会继续使用其他可用来源，并在接口结果中返回来源统计和不可用来源信息。

arXiv 有额外保护：如果遇到 `HTTP 429` 或连续超时，系统会让 arXiv 进入约 20 分钟冷却期。冷却期间 SLR 会直接跳过 arXiv，避免用户每次都等待超时，同时继续使用 OpenAlex、PubMed 和 Crossref 生成报告。
