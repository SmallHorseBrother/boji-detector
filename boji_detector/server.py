"""HTTP service: POST a photo, get the analysis back as JSON.

    boji serve --host 0.0.0.0 --port 8000

Photos are decoded and analysed in memory; they are never written to disk or
to logs. Set BOJI_SERVICE_TOKEN to require an ``X-Boji-Token`` header (use it
when the service is reachable by anything other than your own backend).
"""

from __future__ import annotations

import hmac
import io
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool

from . import __version__
from .analyze import analyze_image, read_image
from .messages import MESSAGES
from .scoring import ScoringConfig, profile_config

logger = logging.getLogger("boji_detector.server")

DEFAULT_MAX_UPLOAD_MB = 10.0
MAX_PIXELS = 40_000_000         # refuse decompression bombs before decoding


def _max_upload_bytes() -> int:
    try:
        mb = float(os.environ.get("BOJI_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB))
    except ValueError:
        mb = DEFAULT_MAX_UPLOAD_MB
    return int(max(0.5, mb) * 1024 * 1024)


def _messages_for(report: dict[str, Any]) -> dict[str, dict[str, str]]:
    codes = set(report.get("quality", []))
    for metric in report.get("metrics", {}).values():
        codes.update(metric.get("reasons", []))
    return {code: {"zh": MESSAGES[code][0], "en": MESSAGES[code][1]} for code in sorted(codes) if code in MESSAGES}


def _default_perceiver():
    from .perception_mp import MediaPipePerceiver

    return MediaPipePerceiver()


def create_app(perceiver_factory: Callable[[], Any] = _default_perceiver) -> FastAPI:
    state: dict[str, Any] = {"perceiver": None}
    lock = threading.Lock()          # MediaPipe graphs are not safe to share across threads

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["perceiver"] = perceiver_factory()   # load models once, before the first request
        try:
            yield
        finally:
            closer = getattr(state["perceiver"], "close", None)
            if closer:
                closer()

    app = FastAPI(title="boji-detector", version=__version__, docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": state["perceiver"] is not None, "version": __version__}

    def _analyze(data: bytes, config: ScoringConfig) -> dict[str, Any]:
        from PIL import Image

        try:
            with Image.open(io.BytesIO(data)) as probe:
                width, height = probe.size
        except Exception:
            raise HTTPException(status_code=400, detail="unreadable_image") from None
        if width * height > MAX_PIXELS:
            raise HTTPException(status_code=413, detail="image_too_large")
        try:
            bgr = read_image(io.BytesIO(data))
        except Exception:
            raise HTTPException(status_code=400, detail="unreadable_image") from None
        with lock:
            report, _ = analyze_image(bgr, config, perceiver=state["perceiver"])
        payload = report.to_dict()
        return {"report": payload, "messages": _messages_for(payload)}

    @app.post("/v1/analyze")
    async def analyze(
        request: Request,
        x_boji_token: str | None = Header(default=None),
        content_length: int | None = Header(default=None),
        sex: str = Query(default="male"),
    ) -> dict[str, Any]:
        token = os.environ.get("BOJI_SERVICE_TOKEN")
        if token and not hmac.compare_digest(token, x_boji_token or ""):
            raise HTTPException(status_code=401, detail="invalid_token")
        try:
            config = profile_config(sex)
        except ValueError:
            raise HTTPException(status_code=400, detail="unknown_profile") from None
        limit = _max_upload_bytes()
        if content_length is not None and content_length > limit:
            raise HTTPException(status_code=413, detail="file_too_large")
        data = await request.body()
        if not data:
            raise HTTPException(status_code=400, detail="empty_body")
        if len(data) > limit:
            raise HTTPException(status_code=413, detail="file_too_large")
        try:
            return await run_in_threadpool(_analyze, data, config)
        except HTTPException:
            raise
        except Exception:
            # Log the failure type only: never the image or anything derived from it.
            logger.exception("analysis failed")
            raise HTTPException(status_code=500, detail="analysis_failed") from None

    return app


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, workers=1, access_log=False)
