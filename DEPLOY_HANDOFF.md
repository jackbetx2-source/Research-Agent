# Research Agent Deployment Handoff

This package contains the current deployable source for the Research Agent web app.

Included:
- Python backend and CLI: `web_app.py`, `research_agent.py`
- Research Agent package: `src/research_agent/`
- Frontend assets: `web/`
- Dependency list: `requirements.txt`
- Runtime configuration: `.env` and `.env.example`
- Existing project notes: `README.md`, `UI_HANDOFF.md`

Not included:
- Local logs: `web_app.out.log`, `web_app.err.log`
- Python caches: `__pycache__/`
- Historical backups and old deployment packages: `.backups/`, old `.deploy_tmp/` contents
- Generated reports from `outputs/`
- Unused external checkout: `external/deer-flow/`

Quick start:

```powershell
pip install -r requirements.txt
python web_app.py 8000
```

Then open:

```text
http://127.0.0.1:8000
```

Server deployment notes:
- `.env` is intentionally included for trusted handoff and contains the current model provider settings.
- The web server defaults can be controlled with environment variables such as `WEB_HOST`.
- Generated reports will be written to `outputs/`; the folder is included empty so the app has a natural destination.
