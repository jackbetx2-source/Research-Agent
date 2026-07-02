from __future__ import annotations

import asyncio
import cgi
import html
import io
import json
import re
import sys
import threading
import traceback
import uuid
import zipfile
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree

from dotenv import load_dotenv

from research_agent import render_markdown_result
from src.research_agent.doi import enrich_references_with_doi_metadata
from src.research_agent.literature_workflow import LiteratureAnalysisWorkflow
from src.research_agent.llm import LLMClient, LLMServiceError
from src.research_agent.slr_workflow import SLRWorkflow
from src.research_agent.types import ResearchReference
from src.research_agent.workflow import ResearchWorkflow


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
OUTPUT_DIR = ROOT / "outputs"
MAX_PDF_UPLOAD_BYTES = 25 * 1024 * 1024
PDF_EXTRACT_PAGE_LIMIT = 30
PDF_REFERENCE_EXCERPT_CHARS = 12000
RESEARCH_FILE_EXCERPT_CHARS = 5000
RESEARCH_CONTEXT_CHARS = 12000
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

        if path.startswith("/api/slr/"):
            job_id = path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = dict(JOBS.get(job_id, {}))
            if not job:
                self._send_json({"error": "Job not found."}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(job)
            return

        if path.startswith("/api/literature-analysis/"):
            job_id = path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = dict(JOBS.get(job_id, {}))
            if not job:
                self._send_json({"error": "Job not found."}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(job)
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

        if path == "/api/export/pdf":
            self._handle_pdf_export()
            return

        if path == "/api/literature-analysis":
            self._handle_literature_analysis()
            return

        if path == "/api/literature-analysis/pdf":
            self._handle_literature_pdf_analysis()
            return

        if path == "/api/slr":
            self._handle_slr()
            return

        if path == "/api/slr/upload":
            self._handle_slr_upload()
            return

        if path == "/api/research":
            self._handle_research()
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _handle_research(self) -> None:
        try:
            content_type = self.headers.get("Content-Type", "")
            media_type, _ = cgi.parse_header(content_type)
            uploaded_references: list[ResearchReference] = []
            uploaded_documents: list[dict] = []
            if media_type == "multipart/form-data":
                topic, fast, uploaded_references, uploaded_documents = self._read_research_upload()
            elif media_type in {"application/json", "text/json", ""}:
                payload = self._read_json()
                topic = str(payload.get("topic", "")).strip()
                if not topic:
                    self._send_json({"error": "Topic cannot be empty."}, HTTPStatus.BAD_REQUEST)
                    return
                fast = bool(payload.get("fast", True))
            else:
                raise ValueError("Expected JSON or multipart/form-data with PDF/DOCX files.")

            job_id = uuid.uuid4().hex
            with JOBS_LOCK:
                JOBS[job_id] = {"status": "queued"}

            thread = threading.Thread(
                target=self._run_job,
                args=(job_id, topic, fast, uploaded_references, uploaded_documents),
                daemon=True,
            )
            thread.start()
            self._send_json({"job_id": job_id, "status": "queued"}, HTTPStatus.ACCEPTED)
        except BrokenPipeError:
            print("[web] client disconnected before response was sent", flush=True)
        except ValueError as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except UnicodeDecodeError as error:
            self._send_json(
                {
                    "error": (
                        "Uploaded Word content could not be decoded. "
                        "Please upload a real .docx file or export the document as PDF. "
                        "Legacy .doc files are not supported."
                    )
                },
                HTTPStatus.BAD_REQUEST,
            )
        except Exception as error:
            traceback.print_exc()
            try:
                self._send_json(
                    {"error": f"{type(error).__name__}: {error}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            except BrokenPipeError:
                print("[web] client disconnected before error response was sent", flush=True)

    def _handle_literature_analysis(self) -> None:
        try:
            payload = self._read_json()
            references = payload.get("references", [])
            final_report = str(payload.get("final_report", "") or "")
            if not isinstance(references, list):
                self._send_json(
                    {"error": "References must be a list."},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            if not references and not final_report.strip():
                self._send_json(
                    {"error": "Please provide references, uploaded files, or text context."},
                    HTTPStatus.BAD_REQUEST,
                )
                return

            topic = str(payload.get("topic", "") or "current research").strip()
            job_id = uuid.uuid4().hex
            with JOBS_LOCK:
                JOBS[job_id] = {"status": "queued", "kind": "literature_analysis"}

            thread = threading.Thread(
                target=self._run_literature_analysis_job,
                args=(job_id, topic, references, final_report),
                daemon=True,
            )
            thread.start()
            self._send_json({"job_id": job_id, "status": "queued"}, HTTPStatus.ACCEPTED)
        except BrokenPipeError:
            print("[web] client disconnected before literature response was sent", flush=True)
        except Exception as error:
            traceback.print_exc()
            try:
                self._send_json(
                    {"error": f"{type(error).__name__}: {error}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            except BrokenPipeError:
                print("[web] client disconnected before literature error response was sent", flush=True)

    def _handle_slr(self) -> None:
        try:
            payload = self._read_json()
            topic = str(payload.get("topic", "")).strip()
            if not topic:
                self._send_json({"error": "Topic cannot be empty."}, HTTPStatus.BAD_REQUEST)
                return

            max_results = int(payload.get("max_results", 20) or 20)
            category = str(payload.get("category", "") or "").strip()
            start_date = str(payload.get("start_date", "") or "").strip()
            end_date = str(payload.get("end_date", "") or "").strip()
            citation_format = str(payload.get("citation_format", "APA") or "APA").strip()
            source_mode = str(payload.get("source_mode", "search") or "search").strip()
            user_papers = payload.get("user_papers", [])
            if not isinstance(user_papers, list):
                user_papers = []
            job_id = uuid.uuid4().hex
            with JOBS_LOCK:
                JOBS[job_id] = {"status": "queued", "kind": "slr"}

            thread = threading.Thread(
                target=self._run_slr_job,
                args=(job_id, topic, max_results, category, start_date, end_date, citation_format, source_mode, user_papers),
                daemon=True,
            )
            thread.start()
            self._send_json({"job_id": job_id, "status": "queued"}, HTTPStatus.ACCEPTED)
        except BrokenPipeError:
            print("[web] client disconnected before SLR response was sent", flush=True)
        except ValueError as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            traceback.print_exc()
            try:
                self._send_json(
                    {"error": f"{type(error).__name__}: {error}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            except BrokenPipeError:
                print("[web] client disconnected before SLR error response was sent", flush=True)

    def _handle_slr_upload(self) -> None:
        try:
            files, link_references, fields = self._read_pdf_uploads_with_fields(allow_empty=True)
            topic = str(fields.get("topic", "") or "").strip()
            if not topic:
                self._send_json({"error": "Topic cannot be empty."}, HTTPStatus.BAD_REQUEST)
                return

            references = list(link_references)
            uploaded_references = [
                self._uploaded_file_to_reference(filename, content)
                for filename, content in files
            ]
            paper_references, context_documents = self._split_reference_roles(
                references + uploaded_references
            )
            references = paper_references
            if context_documents:
                topic = self._build_uploaded_context_topic(topic, context_documents)
            max_results = int(fields.get("max_results", "20") or 20)
            category = str(fields.get("category", "") or "").strip()
            start_date = str(fields.get("start_date", "") or "").strip()
            end_date = str(fields.get("end_date", "") or "").strip()
            citation_format = str(fields.get("citation_format", "APA") or "APA").strip()
            source_mode = str(fields.get("source_mode", "mixed") or "mixed").strip()
            if references:
                references = enrich_references_with_doi_metadata(references)
            elif source_mode in {"upload", "mixed"}:
                source_mode = "search"
            job_id = uuid.uuid4().hex
            with JOBS_LOCK:
                JOBS[job_id] = {"status": "queued", "kind": "slr"}

            thread = threading.Thread(
                target=self._run_slr_job,
                args=(job_id, topic, max_results, category, start_date, end_date, citation_format, source_mode, references),
                daemon=True,
            )
            thread.start()
            self._send_json(
                {
                    "job_id": job_id,
                    "status": "queued",
                    "references": references,
                    "context_documents": context_documents,
                },
                HTTPStatus.ACCEPTED,
            )
        except BrokenPipeError:
            print("[web] client disconnected before SLR upload response was sent", flush=True)
        except ValueError as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            traceback.print_exc()
            try:
                self._send_json(
                    {"error": f"{type(error).__name__}: {error}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            except BrokenPipeError:
                print("[web] client disconnected before SLR upload error response was sent", flush=True)

    def _handle_literature_pdf_analysis(self) -> None:
        try:
            files, link_references, fields = self._read_pdf_uploads_with_fields(allow_empty=True)
            references = list(link_references)
            uploaded_references = [
                self._uploaded_file_to_reference(filename, content)
                for filename, content in files
            ]
            references, context_documents = self._split_reference_roles(
                references + uploaded_references
            )
            user_context = fields.get("user_context", "").strip()
            if not references and not user_context and not context_documents:
                self._send_json(
                    {"error": "Please provide references, uploaded files, or text context."},
                    HTTPStatus.BAD_REQUEST,
                )
                return

            topic = str(fields.get("topic", "") or "").strip() or "user-provided literature links and PDF analysis"
            final_report = self._build_pdf_context(references)
            if context_documents:
                context_block = self._build_uploaded_context(context_documents)
                final_report = f"{final_report}\n\n{context_block}".strip()
            if user_context:
                final_report = (
                    f"{final_report}\n\n" if final_report else ""
                ) + f"User-provided text context or instructions:\n{user_context}"
            job_id = uuid.uuid4().hex
            with JOBS_LOCK:
                JOBS[job_id] = {"status": "queued", "kind": "literature_analysis"}

            thread = threading.Thread(
                target=self._run_literature_analysis_job,
                args=(job_id, topic, references, final_report),
                daemon=True,
            )
            thread.start()
            self._send_json(
                {
                    "job_id": job_id,
                    "status": "queued",
                    "references": references,
                },
                HTTPStatus.ACCEPTED,
            )
        except BrokenPipeError:
            print("[web] client disconnected before document literature response was sent", flush=True)
        except ValueError as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            traceback.print_exc()
            try:
                self._send_json(
                    {"error": f"{type(error).__name__}: {error}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            except BrokenPipeError:
                print("[web] client disconnected before document literature error response was sent", flush=True)

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
        except LLMServiceError as error:
            status = HTTPStatus.SERVICE_UNAVAILABLE
            if error.status_code and error.status_code not in {502, 503, 504}:
                status = HTTPStatus.BAD_GATEWAY
            self._send_json({"error": str(error)}, status)
        except Exception as error:
            traceback.print_exc()
            try:
                self._send_json(
                    {"error": f"{type(error).__name__}: {error}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            except BrokenPipeError:
                print("[web] client disconnected before chat error response was sent", flush=True)

    def _handle_pdf_export(self) -> None:
        try:
            payload = self._read_json()
            title = str(payload.get("title", "") or "Research report").strip()
            markdown = str(payload.get("markdown", "") or "").strip()
            if not markdown:
                self._send_json({"error": "Markdown content cannot be empty."}, HTTPStatus.BAD_REQUEST)
                return
            pdf = self._markdown_to_pdf_bytes(title, markdown)
            filename = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "_", title).strip("_")[:80] or "research_report"
            self._send_binary(
                pdf,
                "application/pdf",
                f'{filename}.pdf',
            )
        except BrokenPipeError:
            print("[web] client disconnected before PDF export response was sent", flush=True)
        except RuntimeError as error:
            self._send_json({"error": str(error)}, HTTPStatus.SERVICE_UNAVAILABLE)
        except Exception as error:
            traceback.print_exc()
            try:
                self._send_json(
                    {"error": f"{type(error).__name__}: {error}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            except BrokenPipeError:
                print("[web] client disconnected before PDF export error response was sent", flush=True)

    def _run_job(
        self,
        job_id: str,
        topic: str,
        fast: bool,
        uploaded_references: list[ResearchReference] | None = None,
        uploaded_documents: list[dict] | None = None,
    ) -> None:
        with JOBS_LOCK:
            JOBS[job_id] = {"status": "running", "stage": "Starting research..."}

        try:
            if uploaded_documents:
                with JOBS_LOCK:
                    JOBS[job_id] = {
                        "status": "running",
                        "stage": "Running research workflow...",
                    }
                topic = self._build_research_context_topic(topic, uploaded_documents)
            result = asyncio.run(
                ResearchWorkflow(
                    fast=fast,
                    verbose=True,
                ).run(topic)
            )
            if uploaded_references:
                existing_keys = {
                    re.sub(r"\W+", "", reference.title.casefold())
                    for reference in result.references
                }
                for reference in uploaded_references:
                    key = re.sub(r"\W+", "", reference.title.casefold())
                    if key and key not in existing_keys:
                        result.references.insert(0, reference)
                        existing_keys.add(key)
            output_path = self._save_result(topic, result)
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "done",
                    "intent": result.intent.summary,
                    "final_report": result.final_report,
                    "references": [self._reference_to_dict(item) for item in result.references],
                    "output_path": str(output_path),
                }
        except LLMServiceError as error:
            print(f"[web] LLM service error: {error}", flush=True)
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "error",
                    "error": str(error),
                }
        except Exception as error:
            traceback.print_exc()
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }

    def _run_literature_analysis_job(
        self,
        job_id: str,
        topic: str,
        references: list[dict],
        final_report: str,
    ) -> None:
        with JOBS_LOCK:
            JOBS[job_id] = {
                "status": "running",
                "kind": "literature_analysis",
                "stage": "Starting literature analysis...",
            }

        try:
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "running",
                    "kind": "literature_analysis",
                    "stage": "Resolving DOI metadata...",
                }
            references = enrich_references_with_doi_metadata(references)
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "running",
                    "kind": "literature_analysis",
                    "stage": "Running LLM literature analysis...",
                }
            analysis_result = asyncio.run(
                LiteratureAnalysisWorkflow(verbose=True).run(
                    topic=topic,
                    references=references,
                    final_report=final_report,
                )
            )
            if isinstance(analysis_result, dict):
                rows = analysis_result.get("rows", [])
                summary = analysis_result.get("summary", {})
            else:
                rows = analysis_result
                summary = {}
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "done",
                    "kind": "literature_analysis",
                    "rows": rows,
                    "summary": summary,
                }
        except LLMServiceError as error:
            print(f"[web] literature LLM service error: {error}", flush=True)
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "error",
                    "kind": "literature_analysis",
                    "error": str(error),
                }
        except Exception as error:
            traceback.print_exc()
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "error",
                    "kind": "literature_analysis",
                    "error": f"{type(error).__name__}: {error}",
                }

    def _run_slr_job(
        self,
        job_id: str,
        topic: str,
        max_results: int,
        category: str,
        start_date: str,
        end_date: str,
        citation_format: str,
        source_mode: str,
        user_papers: list[dict],
    ) -> None:
        with JOBS_LOCK:
            JOBS[job_id] = {
                "status": "running",
                "kind": "slr",
                "stage": "Preparing SLR sources...",
            }

        try:
            result = asyncio.run(
                SLRWorkflow(verbose=True).run(
                    topic=topic,
                    max_results=max_results,
                    category=category,
                    start_date=start_date,
                    end_date=end_date,
                    citation_format=citation_format,
                    source_mode=source_mode,
                    user_papers=user_papers,
                )
            )
            output_path = self._save_slr_result(topic, result.report_markdown)
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "done",
                    "kind": "slr",
                    "topic": result.topic,
                    "search_query": result.search_query,
                    "citation_format": result.citation_format,
                    "source_mode": source_mode,
                    "papers": result.papers,
                    "source_counts": result.source_counts,
                    "source_errors": result.source_errors,
                    "annotations": result.annotations,
                    "synthesis": result.synthesis,
                    "references": result.references,
                    "report_markdown": result.report_markdown,
                    "output_path": str(output_path),
                }
        except LLMServiceError as error:
            print(f"[web] SLR LLM service error: {error}", flush=True)
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "error",
                    "kind": "slr",
                    "error": str(error),
                }
        except Exception as error:
            traceback.print_exc()
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "error",
                    "kind": "slr",
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
            "- The workflow first understands intent, plans dynamic research perspectives, "
            "runs parallel research branches, plans an outline, then synthesizes a final Markdown report.\n"
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
        try:
            raw = self.rfile.read(length).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                "Request body is not valid UTF-8 JSON. If you are uploading a file, "
                "please submit it as PDF or DOCX through the upload control."
            ) from error
        data = json.loads(raw or "{}")
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        return data

    def _read_research_upload(self) -> tuple[str, bool, list[ResearchReference], list[dict]]:
        content_type = self.headers.get("Content-Type", "")
        media_type, _ = cgi.parse_header(content_type)
        if media_type != "multipart/form-data":
            raise ValueError("Expected multipart/form-data with PDF or DOCX files.")

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("Upload body cannot be empty.")
        if content_length > MAX_PDF_UPLOAD_BYTES:
            raise ValueError("Upload is too large. Please keep it under 25 MB.")

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": str(content_length),
            },
        )
        raw_topic = form.getvalue("topic", "")
        if isinstance(raw_topic, list):
            raw_topic = raw_topic[0] if raw_topic else ""
        topic = str(raw_topic or "").strip()
        raw_fast = form.getvalue("fast", "true")
        if isinstance(raw_fast, list):
            raw_fast = raw_fast[0] if raw_fast else "true"
        fast = str(raw_fast).strip().lower() not in {"false", "0", "no", "off"}

        items = []
        for field_name in ("document", "file", "pdf"):
            if field_name not in form:
                continue
            field_items = form[field_name]
            if not isinstance(field_items, list):
                field_items = [field_items]
            items.extend(field_items)

        documents = []
        references: list[ResearchReference] = []
        for item in items:
            filename = Path(item.filename or "uploaded-document").name
            content = item.file.read()
            if not content:
                continue
            document = self._extract_research_document(filename, content)
            documents.append(document)
            references.append(
                ResearchReference(
                    title=document["title"],
                    relevance=document["relevance"],
                    source=filename,
                    branch_name="Uploaded document",
                )
            )

        if not topic and not documents:
            raise ValueError("Please enter a research topic or upload at least one PDF/DOCX file.")
        return topic, fast, references, documents

    @staticmethod
    def _extract_research_document(filename: str, content: bytes) -> dict[str, str]:
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            extracted = ResearchWebHandler._extract_pdf_content(content)
            text = extracted["text"]
            note = extracted["note"]
            metadata_title = str((extracted["metadata"] or {}).get("title", "") or "").strip()
            title = metadata_title or Path(filename).stem or filename
            if not text:
                text = "No readable text was extracted from this PDF; it may be scanned, image-only, or protected."
            return {
                "title": title,
                "filename": filename,
                "kind": "PDF",
                "text": text[:RESEARCH_FILE_EXCERPT_CHARS],
                "note": note,
                "relevance": "User-uploaded PDF used as context for the research workspace.",
            }
        if suffix == ".docx":
            text = ResearchWebHandler._extract_docx_text(content)
            if not text:
                text = "No readable text was extracted from this DOCX."
            return {
                "title": Path(filename).stem or filename,
                "filename": filename,
                "kind": "DOCX",
                "text": text[:RESEARCH_FILE_EXCERPT_CHARS],
                "note": f"Extracted text from DOCX, clipped to {RESEARCH_FILE_EXCERPT_CHARS} characters.",
                "relevance": "User-uploaded DOCX used as context for the research workspace.",
            }
        if suffix == ".doc":
            raise ValueError(
                f"{filename} is a legacy .doc file. Please save/export it as .docx or PDF, then upload again."
            )
        raise ValueError(f"{filename} is not supported. Please upload PDF or DOCX files.")

    @staticmethod
    def _extract_docx_text(content: bytes) -> str:
        try:
            archive = zipfile.ZipFile(io.BytesIO(content))
        except zipfile.BadZipFile as error:
            raise ValueError("DOCX file is invalid or corrupted.") from error

        parts = []
        namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        xml_names = [
            name
            for name in archive.namelist()
            if name == "word/document.xml"
            or name.startswith("word/header")
            or name.startswith("word/footer")
        ]
        for xml_name in xml_names:
            try:
                root = ElementTree.fromstring(archive.read(xml_name))
            except (ElementTree.ParseError, UnicodeDecodeError):
                continue
            for paragraph in root.findall(".//w:p", namespaces):
                runs = [
                    node.text or ""
                    for node in paragraph.findall(".//w:t", namespaces)
                    if node.text
                ]
                line = "".join(runs).strip()
                if line:
                    parts.append(line)
        return re.sub(r"\n{3,}", "\n\n", "\n".join(parts)).strip()

    @staticmethod
    def _build_research_context_topic(topic: str, documents: list[dict[str, str]]) -> str:
        base_topic = topic or "Use the uploaded document(s) to infer the research task and produce an appropriate research report."
        sections = [
            "The user provided uploaded document(s) as auxiliary material for the research workspace.",
            "Important: uploaded documents may be writing requirements, grading rubrics, assignment prompts, style constraints, background notes, or source evidence.",
            "First infer each document's role. Treat requirement/rubric/style documents as instructions or constraints, not as evidence about the research subject.",
            "Use uploaded content as source evidence only when it clearly contains substantive material about the research topic.",
            f"User text topic:\n{base_topic}",
        ]
        remaining = RESEARCH_CONTEXT_CHARS - sum(len(section) for section in sections)
        for index, document in enumerate(documents, start=1):
            if remaining <= 0:
                break
            excerpt = document["text"][: max(0, remaining)]
            block = (
                f"\n\nUploaded document {index}: {document['title']}\n"
                f"Filename: {document['filename']}\n"
                f"Type: {document['kind']}\n"
                f"Extraction note: {document['note']}\n"
                "Possible role: classify as requirement/rubric/style guide/background/source evidence before using it.\n"
                "Extracted excerpt:\n"
                f"{excerpt}"
            )
            sections.append(block)
            remaining -= len(block)
        return "\n\n".join(sections)

    def _read_pdf_uploads(self) -> tuple[list[tuple[str, bytes]], list[dict]]:
        files, references, _ = self._read_pdf_uploads_with_fields()
        return files, references

    def _read_pdf_uploads_with_fields(
        self,
        *,
        allow_empty: bool = False,
    ) -> tuple[list[tuple[str, bytes]], list[dict], dict[str, str]]:
        content_type = self.headers.get("Content-Type", "")
        media_type, _ = cgi.parse_header(content_type)
        if media_type != "multipart/form-data":
            raise ValueError("Expected multipart/form-data with PDF or DOCX files.")

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("Upload body cannot be empty.")
        if content_length > MAX_PDF_UPLOAD_BYTES:
            raise ValueError("Upload is too large. Please keep it under 25 MB.")

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": str(content_length),
            },
        )
        items = form["pdf"] if "pdf" in form else []
        if not isinstance(items, list):
            items = [items]

        references = self._multipart_references(form)
        fields = self._multipart_fields(form)
        files: list[tuple[str, bytes]] = []
        for item in items:
            filename = Path(item.filename or "uploaded-document").name
            suffix = Path(filename).suffix.lower()
            if suffix == ".doc":
                raise ValueError(
                    f"{filename} is a legacy .doc file. Please save/export it as .docx or PDF, then upload again."
                )
            if suffix not in {".pdf", ".docx"}:
                raise ValueError(f"{filename} is not supported. Please upload PDF or DOCX files.")
            content = item.file.read()
            if not content:
                continue
            files.append((filename, content))

        if not allow_empty and not files and not references:
            raise ValueError("Please upload at least one non-empty PDF/DOCX file or provide references.")
        return files, references, fields

    @staticmethod
    def _multipart_references(form: cgi.FieldStorage) -> list[dict]:
        raw = form.getvalue("references", "[]")
        if isinstance(raw, list):
            raw = raw[0] if raw else "[]"
        try:
            data = json.loads(str(raw or "[]"))
        except json.JSONDecodeError as error:
            raise ValueError("References field must be valid JSON.") from error
        if not isinstance(data, list):
            raise ValueError("References field must be a JSON list.")
        return [dict(item) for item in data if isinstance(item, dict) and str(item.get("title", "")).strip()]

    @staticmethod
    def _multipart_fields(form: cgi.FieldStorage) -> dict[str, str]:
        fields: dict[str, str] = {}
        for key in ["topic", "max_results", "category", "start_date", "end_date", "citation_format", "source_mode", "user_context"]:
            value = form.getvalue(key, "")
            if isinstance(value, list):
                value = value[0] if value else ""
            fields[key] = str(value or "")
        return fields

    @staticmethod
    def _uploaded_file_to_reference(filename: str, content: bytes) -> dict:
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            extracted = ResearchWebHandler._extract_pdf_content(content)
            text = extracted["text"]
            has_text = bool(text)
            if not has_text:
                text = "未能从 PDF 中提取到可读文本，可能是扫描件或受保护文档。"
            excerpt = text[:PDF_REFERENCE_EXCERPT_CHARS]
            metadata = extracted["metadata"]
            metadata_title = str(metadata.get("title", "") or "").strip()
            return {
                "title": metadata_title or Path(filename).stem or filename,
                "source": filename,
                "relevance": (
                    "用户上传 PDF，系统已提取正文片段用于分析。"
                    if has_text
                    else "用户上传 PDF，但系统未提取到可读正文；需基于元数据谨慎分析。"
                ),
                "branch_name": "文件上传",
                "abstract": excerpt,
                "content_excerpt": excerpt,
                "pdf_text_available": has_text,
                "pdf_page_count": extracted["page_count"],
                "pdf_extracted_pages": extracted["extracted_pages"],
                "pdf_extraction_note": extracted["note"],
                "authors": metadata.get("author", ""),
                "year": "",
                "journal": "",
                "pdf_metadata": metadata,
                "document_type": "PDF",
            }

        if suffix == ".docx":
            text = ResearchWebHandler._extract_docx_text(content)
            has_text = bool(text)
            if not has_text:
                text = "未能从 DOCX 中提取到可读文本。"
            excerpt = text[:PDF_REFERENCE_EXCERPT_CHARS]
            return {
                "title": Path(filename).stem or filename,
                "source": filename,
                "relevance": (
                    "用户上传 DOCX，系统已提取正文片段用于分析。"
                    if has_text
                    else "用户上传 DOCX，但系统未提取到可读正文；需谨慎分析。"
                ),
                "branch_name": "文件上传",
                "abstract": excerpt,
                "content_excerpt": excerpt,
                "pdf_text_available": has_text,
                "pdf_page_count": 0,
                "pdf_extracted_pages": 0,
                "pdf_extraction_note": "Extracted text from DOCX." if has_text else "No readable text extracted from DOCX.",
                "authors": "",
                "year": "",
                "journal": "",
                "pdf_metadata": {},
                "document_type": "DOCX",
            }

        if suffix == ".doc":
            raise ValueError(
                f"{filename} is a legacy .doc file. Please save/export it as .docx or PDF, then upload again."
            )
        raise ValueError(f"{filename} is not supported. Please upload PDF or DOCX files.")

    @staticmethod
    def _infer_uploaded_document_role(
        *,
        filename: str,
        text: str,
        metadata: dict | None = None,
        document_type: str = "",
    ) -> str:
        metadata = metadata or {}
        sample = f"{filename}\n{metadata.get('title', '')}\n{text[:6000]}".casefold()
        requirement_terms = [
            "assignment",
            "rubric",
            "grading",
            "marking criteria",
            "assessment",
            "writing requirement",
            "format requirement",
            "style guide",
            "instructions",
            "brief",
            "word count",
            "submission",
            "deadline",
            "课程要求",
            "作业要求",
            "写作要求",
            "评分标准",
            "评分细则",
            "格式要求",
            "字数",
            "提交",
            "老师要求",
            "导师要求",
        ]
        scholarly_terms = [
            "abstract",
            "introduction",
            "methodology",
            "methods",
            "results",
            "discussion",
            "conclusion",
            "references",
            "doi:",
            "关键词",
            "摘要",
            "引言",
            "研究方法",
            "实验",
            "结果",
            "讨论",
            "参考文献",
        ]
        requirement_score = sum(1 for term in requirement_terms if term in sample)
        scholarly_score = sum(1 for term in scholarly_terms if term in sample)
        requirement_score += sum(
            1
            for term in [
                "\u8bfe\u7a0b\u8981\u6c42",
                "\u4f5c\u4e1a\u8981\u6c42",
                "\u5199\u4f5c\u8981\u6c42",
                "\u8bc4\u5206\u6807\u51c6",
                "\u8bc4\u5206\u7ec6\u5219",
                "\u683c\u5f0f\u8981\u6c42",
                "\u5b57\u6570",
                "\u63d0\u4ea4",
                "\u8001\u5e08\u8981\u6c42",
                "\u5bfc\u5e08\u8981\u6c42",
            ]
            if term in sample
        )
        scholarly_score += sum(
            1
            for term in [
                "\u5173\u952e\u8bcd",
                "\u6458\u8981",
                "\u5f15\u8a00",
                "\u7814\u7a76\u65b9\u6cd5",
                "\u5b9e\u9a8c",
                "\u7ed3\u679c",
                "\u8ba8\u8bba",
                "\u53c2\u8003\u6587\u732e",
            ]
            if term in sample
        )
        has_identifier = bool(re.search(r"\bdoi\b|10\.\d{4,9}/|arxiv", sample, re.IGNORECASE))
        has_pdf_title_metadata = document_type == "PDF" and bool(str(metadata.get("title", "")).strip())

        if requirement_score >= 2 and scholarly_score < 4:
            return "instructions"
        if document_type == "DOCX" and requirement_score >= 1 and not has_identifier:
            return "instructions"
        if has_identifier or has_pdf_title_metadata or scholarly_score >= 3:
            return "literature"
        if requirement_score >= 1:
            return "instructions"
        return "literature"

    @staticmethod
    def _split_reference_roles(references: list[dict]) -> tuple[list[dict], list[dict]]:
        literature = []
        context_documents = []
        for reference in references:
            if not isinstance(reference, dict):
                continue
            item = dict(reference)
            role = str(item.get("document_role") or "").strip().lower()
            if not role and item.get("document_type"):
                role = ResearchWebHandler._infer_uploaded_document_role(
                    filename=str(item.get("source") or item.get("title") or ""),
                    text=str(item.get("content_excerpt") or item.get("abstract") or ""),
                    metadata=item.get("pdf_metadata") if isinstance(item.get("pdf_metadata"), dict) else {},
                    document_type=str(item.get("document_type") or ""),
                )
                item["document_role"] = role
                item["is_literature_source"] = role == "literature"
            if role and role != "literature":
                context_documents.append(item)
            else:
                literature.append(item)
        return literature, context_documents

    @staticmethod
    def _build_uploaded_context_topic(topic: str, documents: list[dict]) -> str:
        context = ResearchWebHandler._build_uploaded_context(documents)
        return (
            f"{topic}\n\n"
            "User-uploaded auxiliary documents are provided below. First infer their role. "
            "Treat writing requirements, rubrics, assignment prompts, and style constraints "
            "as instructions for the review, not as papers or evidence.\n\n"
            f"{context}"
        ).strip()

    @staticmethod
    def _build_uploaded_context(documents: list[dict]) -> str:
        sections = [
            "Uploaded auxiliary documents. These may be writing requirements, rubrics, assignment prompts, style constraints, background notes, or source evidence.",
            "Use them as source evidence only when they clearly contain substantive research content.",
        ]
        for index, document in enumerate(documents, start=1):
            title = document.get("title") or document.get("source") or f"Uploaded document {index}"
            source = document.get("source", "")
            role = document.get("document_role", "unknown")
            excerpt = (document.get("content_excerpt") or document.get("abstract") or "")[:PDF_REFERENCE_EXCERPT_CHARS]
            sections.append(
                f"## Auxiliary document {index}: {title}\n\n"
                f"Source: {source}\n\n"
                f"Detected role: {role}\n\n"
                "Use note: classify this document before using it; if it is a rubric or writing requirement, apply it as task constraints rather than literature evidence.\n\n"
                f"Extracted excerpt:\n{excerpt}"
            )
        return "\n\n".join(section.strip() for section in sections if section).strip()

    @staticmethod
    def _extract_pdf_content(content: bytes) -> dict:
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError("PDF extraction requires the pypdf package. Run: pip install pypdf") from error

        reader = PdfReader(io.BytesIO(content))
        page_count = len(reader.pages)
        metadata = ResearchWebHandler._clean_pdf_metadata(reader.metadata)
        parts = []
        extracted_pages = 0
        for page_number, page in enumerate(reader.pages[:PDF_EXTRACT_PAGE_LIMIT], start=1):
            page_text = page.extract_text() or ""
            page_text = re.sub(r"\s+", " ", page_text).strip()
            if not page_text:
                continue
            extracted_pages += 1
            parts.append(f"[Page {page_number}]\n{page_text}")

        text = "\n\n".join(parts).strip()
        if text:
            note = (
                f"Extracted text from {extracted_pages}/{page_count} pages "
                f"(first {min(page_count, PDF_EXTRACT_PAGE_LIMIT)} pages scanned)."
            )
        else:
            note = "No readable text extracted; the PDF may be scanned, image-only, or protected."
        return {
            "text": text,
            "page_count": page_count,
            "extracted_pages": extracted_pages,
            "metadata": metadata,
            "note": note,
        }

    @staticmethod
    def _clean_pdf_metadata(metadata) -> dict:
        if not metadata:
            return {}
        cleaned = {}
        for key, value in dict(metadata).items():
            normalized_key = str(key).lstrip("/").lower()
            if normalized_key in {"title", "author", "subject", "creator", "producer"}:
                cleaned[normalized_key] = str(value or "").strip()[:500]
        return cleaned

    @staticmethod
    def _build_pdf_context(references: list[dict]) -> str:
        sections = []
        for reference in references:
            title = reference.get("title", "Uploaded PDF")
            source = reference.get("source", "")
            excerpt = reference.get("content_excerpt") or reference.get("abstract", "")
            extraction_note = reference.get("pdf_extraction_note", "")
            metadata = reference.get("pdf_metadata") or {}
            metadata_lines = []
            if metadata:
                metadata_lines.append("PDF metadata:")
                metadata_lines.extend(
                    f"- {key}: {value}"
                    for key, value in metadata.items()
                    if value
                )
            if extraction_note:
                metadata_lines.append(f"Extraction note: {extraction_note}")
            metadata_block = "\n".join(metadata_lines)
            sections.append(
                f"## {title}\n\n"
                f"Source: {source}\n\n"
                f"{metadata_block}\n\n"
                "Content excerpt for grounded paper review:\n"
                f"{excerpt}"
            )
        return "\n\n".join(sections)

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

    def _send_binary(self, body: bytes, content_type: str, filename: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _markdown_to_pdf_bytes(title: str, markdown: str) -> bytes:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import mm
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer
        except ImportError as error:
            raise RuntimeError("PDF export requires reportlab. Run: pip install -r requirements.txt") from error

        font_name = "Helvetica"
        for font_path in [
            Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("C:/Windows/Fonts/simsun.ttc"),
            Path("C:/Windows/Fonts/arial.ttf"),
        ]:
            if not font_path.exists():
                continue
            try:
                pdfmetrics.registerFont(TTFont("ResearchFont", str(font_path)))
                font_name = "ResearchFont"
                break
            except Exception:
                continue

        styles = getSampleStyleSheet()
        normal = ParagraphStyle(
            "ResearchNormal",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=10.5,
            leading=15,
            spaceAfter=6,
        )
        heading1 = ParagraphStyle(
            "ResearchHeading1",
            parent=styles["Heading1"],
            fontName=font_name,
            fontSize=18,
            leading=23,
            spaceBefore=8,
            spaceAfter=8,
        )
        heading2 = ParagraphStyle(
            "ResearchHeading2",
            parent=styles["Heading2"],
            fontName=font_name,
            fontSize=14,
            leading=19,
            spaceBefore=8,
            spaceAfter=6,
        )
        heading3 = ParagraphStyle(
            "ResearchHeading3",
            parent=styles["Heading3"],
            fontName=font_name,
            fontSize=12,
            leading=17,
            spaceBefore=6,
            spaceAfter=4,
        )

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title=title,
        )
        story = [Paragraph(ResearchWebHandler._pdf_inline_text(title), heading1), Spacer(1, 4)]
        pending_list = []

        def flush_list() -> None:
            nonlocal pending_list
            if not pending_list:
                return
            story.append(
                ListFlowable(
                    [ListItem(Paragraph(item, normal)) for item in pending_list],
                    bulletType="bullet",
                    leftIndent=14,
                )
            )
            pending_list = []

        for raw_line in markdown.splitlines():
            line = raw_line.strip()
            if not line:
                flush_list()
                story.append(Spacer(1, 4))
                continue
            if line.startswith("### "):
                flush_list()
                story.append(Paragraph(ResearchWebHandler._pdf_inline_text(line[4:]), heading3))
            elif line.startswith("## "):
                flush_list()
                story.append(Paragraph(ResearchWebHandler._pdf_inline_text(line[3:]), heading2))
            elif line.startswith("# "):
                flush_list()
                story.append(Paragraph(ResearchWebHandler._pdf_inline_text(line[2:]), heading1))
            elif re.match(r"^[-*]\s+", line):
                pending_list.append(ResearchWebHandler._pdf_inline_text(re.sub(r"^[-*]\s+", "", line)))
            else:
                flush_list()
                story.append(Paragraph(ResearchWebHandler._pdf_inline_text(line), normal))
        flush_list()
        doc.build(story)
        return buffer.getvalue()

    @staticmethod
    def _pdf_inline_text(value: str) -> str:
        text = html.escape(str(value or ""))
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 (\2)", text)
        return text

    @staticmethod
    def _save_result(topic: str, result) -> Path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "_", topic).strip("_")[:40]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"web_{timestamp}_{slug or 'research'}.md"
        output_path.write_text(render_markdown_result(result), encoding="utf-8")
        return output_path.resolve()

    @staticmethod
    def _save_slr_result(topic: str, markdown: str) -> Path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "_", topic).strip("_")[:40]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"slr_{timestamp}_{slug or 'review'}.md"
        output_path.write_text(markdown, encoding="utf-8")
        return output_path.resolve()

    @staticmethod
    def _reference_to_dict(reference) -> dict:
        return {
            "title": reference.title,
            "relevance": reference.relevance,
            "source": reference.source,
            "branch_name": reference.branch_name,
        }


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
