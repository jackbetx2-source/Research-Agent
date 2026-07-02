from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
import traceback
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from research_agent import render_markdown_result
from src.research_agent.llm import LLMClient
from src.research_agent.workflow import ResearchWorkflow


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
OUTPUT_DIR = ROOT / "outputs"
JOBS: dict[str, dict[str, object]] = {}
JOBS_LOCK = threading.Lock()


class ResearchWebHandler(BaseHTTPRequestHandler):
    server_version = "ResearchAgentWeb/0.1"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
            return

        if path == "/styles.css":
            self._send_file(WEB_DIR / "styles.css", "text/css; charset=utf-8")
            return

        if path == "/app.js":
            self._send_file(WEB_DIR / "app.js", "application/javascript; charset=utf-8")
            return

        if path == "/health":
            self._send_json({"ok": True})
            return

        if path.startswith("/api/research/"):
            job_id = path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = dict(JOBS.get(job_id, {}))
            if not job:
                self._send_json({"error": "Job not found."}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(job)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/chat":
            self._handle_chat()
            return

        if path != "/api/research":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            payload = self._read_json()
            topic = str(payload.get("topic", "")).strip()
            if not topic:
                self._send_json({"error": "Topic cannot be empty."}, HTTPStatus.BAD_REQUEST)
                return

            fast = bool(payload.get("fast", True))
            job_id = uuid.uuid4().hex
            with JOBS_LOCK:
                JOBS[job_id] = {"status": "queued"}

            thread = threading.Thread(
                target=self._run_job,
                args=(job_id, topic, fast),
                daemon=True,
            )
            thread.start()
            self._send_json({"job_id": job_id, "status": "queued"}, HTTPStatus.ACCEPTED)
        except BrokenPipeError:
            print("[web] client disconnected before response was sent", flush=True)
        except Exception as error:
            traceback.print_exc()
            try:
                self._send_json(
                    {"error": f"{type(error).__name__}: {error}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            except BrokenPipeError:
                print("[web] client disconnected before error response was sent", flush=True)

    def _handle_chat(self) -> None:
        try:
            payload = self._read_json()
            message = str(payload.get("message", "")).strip()
            if not message:
                self._send_json({"error": "Message cannot be empty."}, HTTPStatus.BAD_REQUEST)
                return

            history = payload.get("history", [])
            if not isinstance(history, list):
                history = []

            answer = asyncio.run(self._answer_project_question(message, history))
            self._send_json({"answer": answer})
        except BrokenPipeError:
            print("[web] client disconnected before chat response was sent", flush=True)
        except Exception as error:
            traceback.print_exc()
            try:
                self._send_json(
                    {"error": f"{type(error).__name__}: {error}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            except BrokenPipeError:
                print("[web] client disconnected before chat error response was sent", flush=True)

    def _run_job(self, job_id: str, topic: str, fast: bool) -> None:
        with JOBS_LOCK:
            JOBS[job_id] = {"status": "running", "stage": "Starting research..."}

        try:
            result = asyncio.run(ResearchWorkflow(fast=fast, verbose=True).run(topic))
            output_path = self._save_result(topic, result)
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "done",
                    "intent": result.intent.summary,
                    "final_report": result.final_report,
                    "output_path": str(output_path),
                }
        except Exception as error:
            traceback.print_exc()
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }

    async def _answer_project_question(self, message: str, history: list) -> str:
        recent_messages = []
        for item in history[-8:]:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = str(item.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            recent_messages.append(f"{role}: {content[:800]}")

        context = "\n".join(recent_messages) or "No prior conversation."
        system_prompt = (
            "You are the built-in assistant for a local Research Agent web app. "
            "Answer in concise, friendly Chinese. Help users understand this project, "
            "its workflow, configuration, and how to use the web UI. If the user asks "
            "for unrelated general research, briefly redirect them to the research workspace. "
            "Do not invent hidden features."
        )
        user_prompt = (
            "Project facts:\n"
            "- The app runs locally with Python and a static web UI.\n"
            "- Users enter a topic in the research workspace.\n"
            "- The workflow first understands intent, then runs three research branches "
            "in parallel, then synthesizes a final Markdown report.\n"
            "- Results are shown in the browser and saved under the outputs directory.\n"
            "- Fast mode produces a shorter report with lower token use.\n\n"
            f"Recent conversation:\n{context}\n\n"
            f"User question:\n{message}"
        )
        return await LLMClient().complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.25,
            max_tokens=700,
        )

    def log_message(self, format: str, *args: object) -> None:
        print(f"[web] {self.address_string()} - {format % args}", flush=True)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        data = json.loads(raw or "{}")
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        return data

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _save_result(topic: str, result) -> Path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "_", topic).strip("_")[:40]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"web_{timestamp}_{slug or 'research'}.md"
        output_path.write_text(render_markdown_result(result), encoding="utf-8")
        return output_path.resolve()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    env_path = ROOT / ".env"
    example_env_path = ROOT / ".env.example"
    load_dotenv(env_path if env_path.exists() else example_env_path)
    host = "127.0.0.1"
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer((host, port), ResearchWebHandler)
    print(f"Research Agent Web is running at http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
