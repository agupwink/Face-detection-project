import asyncio
import base64
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, field_validator

from pipeline import DetectionPipeline
from embeddings import get_face_embedding
from storage import store_detection, get_session_results, find_similar_faces, find_user_profile, update_user_profile, get_session_embedding
from trainer import save_sample, start_finetune_async, update_age_group_bias

# ---------------------------------------------------------------------------
# Logging setup — replace all print() calls with structured log entries
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("face_detection.api")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_PATH = os.getenv("FRAMES_PATH", os.path.join(_PROJECT_ROOT, "data", "frames"))
os.makedirs(FRAMES_PATH, exist_ok=True)

_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_ALLOWED_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
_MAX_REQUEST_BYTES = 10 * 1024 * 1024   # 10 MB
_FRAME_TIMEOUT_SECONDS = 30             # max time allowed for pipeline.process_frame

_pipeline: DetectionPipeline | None = None
_last_save: dict[str, float] = {}
_identified_sessions: set[str] = set()   # sessions where user has already been identified
_processing_sessions: set[str] = set()   # concurrency guard — sessions mid-inference


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    logger.info("Initialising detection pipeline…")
    _pipeline = DetectionPipeline()
    logger.info("Pipeline ready.")
    yield
    _pipeline.shutdown()


app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------------------------
# Middleware — CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware — request size limit (10 MB)
# ---------------------------------------------------------------------------
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > _MAX_REQUEST_BYTES:
        return JSONResponse(
            {"success": False, "error": "Request too large", "details": f"Max allowed size is {_MAX_REQUEST_BYTES} bytes"},
            status_code=413,
        )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Middleware — global exception handler (never leak stack traces)
# ---------------------------------------------------------------------------
@app.middleware("http")
async def global_exception_handler(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("Unhandled exception for %s %s", request.method, request.url.path)
        return JSONResponse(
            {
                "success": False,
                "error": "Internal server error",
                "details": str(exc),
            },
            status_code=500,
        )


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "pipeline_ready": _pipeline is not None}


@app.get("/api/sessions")
async def list_sessions():
    from storage import _collection
    try:
        all_results = _collection.get(include=["metadatas"])
        metas = all_results.get("metadatas") or []
        sessions: dict[str, dict] = {}
        for m in metas:
            sid = m.get("session_id", "")
            if not sid:
                continue
            if sid not in sessions:
                sessions[sid] = {
                    "session_id": sid,
                    "first_seen": m.get("timestamp"),
                    "last_seen": m.get("timestamp"),
                    "detections": 0,
                }
            sessions[sid]["detections"] += 1
            ts = m.get("timestamp", "")
            if ts < sessions[sid]["first_seen"]:
                sessions[sid]["first_seen"] = ts
            if ts > sessions[sid]["last_seen"]:
                sessions[sid]["last_seen"] = ts
        return {
            "total_sessions": len(sessions),
            "sessions": sorted(sessions.values(), key=lambda s: s["last_seen"], reverse=True),
        }
    except Exception as exc:
        logger.exception("list_sessions failed")
        return {
            "success": False,
            "error": "Failed to list sessions",
            "details": str(exc),
            "total_sessions": 0,
            "sessions": [],
        }


@app.post("/api/session/start")
async def start_session():
    return {"session_id": str(uuid.uuid4())}


@app.post("/api/session/{session_id}/end")
async def end_session(session_id: str):
    try:
        detections = get_session_results(session_id)
        return _build_summary(session_id, detections)
    except Exception as exc:
        logger.exception("end_session failed for session %s", session_id)
        return JSONResponse(
            {"success": False, "error": "Failed to end session", "details": str(exc)},
            status_code=500,
        )


class FeedbackRequest(BaseModel):
    real_age: int

    @field_validator("real_age")
    @classmethod
    def age_in_range(cls, v: int) -> int:
        if not (1 <= v <= 120):
            raise ValueError("Age must be between 1 and 120")
        return v


@app.post("/api/session/{session_id}/feedback")
async def session_feedback(session_id: str, body: FeedbackRequest):
    try:
        detections = get_session_results(session_id)
        if not detections:
            return JSONResponse(
                {"success": False, "error": "No detections found for this session", "details": f"session_id={session_id}"},
                status_code=404,
            )

        frame_paths = [
            str(Path(FRAMES_PATH) / d["frame_path"])
            for d in detections
            if d.get("frame_path")
        ]
        ages = [int(d["age"]) for d in detections if d.get("age") and str(d["age"]).isdigit()]
        corrected_avg = round(sum(ages) / len(ages)) if ages else None
        current_bias = _pipeline._session_bias if _pipeline else 0.0
        predicted_age = round(corrected_avg - current_bias) if corrected_avg is not None else None

        # Update per-user profile
        embedding = get_session_embedding(session_id)
        if embedding:
            existing_user_id, _ = find_user_profile(embedding)
            user_id, new_bias = update_user_profile(embedding, existing_user_id, body.real_age, predicted_age)
            logger.info("feedback: user %s… bias updated to %+.1f yrs", user_id[:8], new_bias)
        else:
            new_bias = 0.0

        # Update age group bias for new users
        if predicted_age is not None:
            update_age_group_bias(body.real_age, predicted_age)
            if _pipeline:
                _pipeline.reload_age_group_bias()

        # Save to global trainer for fine-tuning
        stats = save_sample(session_id, body.real_age, predicted_age, frame_paths)

        training_started = False
        if stats["can_finetune"] and _pipeline:
            training_started = start_finetune_async(_pipeline)

        return {
            "status": "saved",
            "n_samples": stats["n_samples"],
            "personal_bias": new_bias,
            "training_started": training_started,
        }
    except Exception as exc:
        logger.exception("session_feedback failed for session %s", session_id)
        return JSONResponse(
            {"success": False, "error": "Failed to process feedback", "details": str(exc)},
            status_code=500,
        )


@app.get("/api/session/{session_id}/summary")
async def get_summary(session_id: str):
    try:
        detections = get_session_results(session_id)
        return _build_summary(session_id, detections)
    except Exception as exc:
        logger.exception("get_summary failed for session %s", session_id)
        return JSONResponse(
            {"success": False, "error": "Failed to get summary", "details": str(exc)},
            status_code=500,
        )


@app.get("/api/session/{session_id}/timeline")
async def get_timeline(session_id: str):
    try:
        detections = get_session_results(session_id)
        return {
            "session_id": session_id,
            "total": len(detections),
            "detections": [
                {
                    "timestamp": d.get("timestamp"),
                    "age": d.get("age") or None,
                    "confidence": d.get("confidence"),
                    "accessories": d.get("accessories", "[]"),
                    "watches": d.get("watches", "[]"),
                    "fashion": d.get("fashion", "[]"),
                    "frame": d.get("frame_path") or None,
                }
                for d in detections
            ],
        }
    except Exception as exc:
        logger.exception("get_timeline failed for session %s", session_id)
        return JSONResponse(
            {"success": False, "error": "Failed to get timeline", "details": str(exc)},
            status_code=500,
        )


@app.get("/api/frames/{filename}")
async def get_frame(filename: str):
    # Path traversal protection: strip any directory component and work only with
    # the bare filename. Path(filename).name handles "..", "/", backslashes etc.
    safe_name = Path(filename).name

    # Extra guard: reject if the resolved stem is empty or still contains separators
    if not safe_name or safe_name != filename.split("/")[-1].split("\\")[-1]:
        return JSONResponse(
            {"success": False, "error": "Invalid filename", "details": "Path traversal not allowed"},
            status_code=400,
        )

    # MIME / extension whitelist
    ext = Path(safe_name).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        return JSONResponse(
            {"success": False, "error": "Unsupported file type", "details": f"Allowed: {', '.join(_ALLOWED_EXTENSIONS)}"},
            status_code=400,
        )

    path = Path(FRAMES_PATH) / safe_name

    # Resolve and double-check the real path is still inside FRAMES_PATH
    try:
        resolved = path.resolve()
        frames_resolved = Path(FRAMES_PATH).resolve()
        if not str(resolved).startswith(str(frames_resolved)):
            logger.warning("Path traversal attempt blocked: %s", filename)
            return JSONResponse(
                {"success": False, "error": "Access denied", "details": "Path outside allowed directory"},
                status_code=403,
            )
    except Exception:
        return JSONResponse(
            {"success": False, "error": "Invalid path", "details": "Could not resolve file path"},
            status_code=400,
        )

    if not path.exists():
        return JSONResponse(
            {"success": False, "error": "File not found", "details": f"{safe_name} does not exist"},
            status_code=404,
        )

    mime = _ALLOWED_MIME.get(ext, "image/jpeg")
    return FileResponse(str(path), media_type=mime)


# ---------------------------------------------------------------------------
# Background storage helper
# ---------------------------------------------------------------------------

async def _store_faces(session_id: str, results: dict, frame, frame_count: int):
    loop = asyncio.get_event_loop()
    now = time.time()
    save_allowed = (now - _last_save.get(session_id, 0)) >= 2.0

    for face in results["faces"]:
        x1, y1, x2, y2 = face["box"]
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        try:
            embedding = await loop.run_in_executor(None, get_face_embedding, face_roi)
        except Exception as exc:
            logger.warning("get_face_embedding failed: %s", exc)
            embedding = None

        # Identify user from first real embedding and apply their personal bias
        if embedding and session_id not in _identified_sessions and _pipeline:
            _identified_sessions.add(session_id)
            try:
                user_id, bias = await loop.run_in_executor(None, find_user_profile, embedding)
                if user_id and bias != 0.0:
                    _pipeline.set_session_bias(bias)
                    logger.info("session %s: recognised user %s… bias %+.1f yrs", session_id, user_id[:8], bias)
            except Exception as exc:
                logger.warning("find_user_profile failed: %s", exc)

        frame_path = None
        if save_allowed:
            try:
                fname = f"{session_id}_{frame_count}.jpg"
                cv2.imwrite(str(Path(FRAMES_PATH) / fname), face_roi)
                frame_path = fname
                _last_save[session_id] = now
                save_allowed = False
            except Exception as exc:
                logger.warning("Failed to save face frame: %s", exc)

        try:
            store_detection(
                session_id=session_id,
                embedding=embedding,
                age=face.get("age"),
                accessories=face.get("accessories", []),
                watches=results["watches"],
                fashion=results["fashion"],
                confidence=face["confidence"],
                frame_path=frame_path,
            )
        except Exception as exc:
            logger.warning("store_detection failed: %s", exc)


# ---------------------------------------------------------------------------
# WebSocket — live frame streaming
# ---------------------------------------------------------------------------

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    _identified_sessions.discard(session_id)
    _processing_sessions.discard(session_id)
    if _pipeline:
        _pipeline.reset_for_new_session()
    frame_count = 0
    loop = asyncio.get_event_loop()

    try:
        while True:
            raw = await websocket.receive_text()

            # --- Per-frame error isolation: a bad frame must never crash the socket ---
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning("WebSocket %s: invalid JSON — %s", session_id, exc)
                continue

            if msg.get("type") != "frame":
                continue

            # --- Concurrency guard: skip if still processing previous frame ---
            if session_id in _processing_sessions:
                logger.debug("WebSocket %s: skipping frame (pipeline busy)", session_id)
                continue

            try:
                img_bytes = base64.b64decode(msg["data"])
            except Exception as exc:
                logger.warning("WebSocket %s: base64 decode failed — %s", session_id, exc)
                continue

            arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                logger.warning("WebSocket %s: cv2.imdecode returned None (corrupt frame?)", session_id)
                continue

            # --- Timeout-protected model inference ---
            _processing_sessions.add(session_id)
            try:
                try:
                    results = await asyncio.wait_for(
                        loop.run_in_executor(None, _pipeline.process_frame, frame),
                        timeout=_FRAME_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    logger.warning("WebSocket %s: process_frame timed out after %ds", session_id, _FRAME_TIMEOUT_SECONDS)
                    results = {"faces": [], "watches": [], "fashion": []}
                except Exception as exc:
                    logger.exception("WebSocket %s: process_frame raised — %s", session_id, exc)
                    results = {"faces": [], "watches": [], "fashion": []}
            finally:
                _processing_sessions.discard(session_id)

            frame_count += 1

            # Send detection to browser IMMEDIATELY — don't wait for storage
            try:
                await websocket.send_text(json.dumps({
                    "type": "detection",
                    "faces": results["faces"],
                    "watches": results["watches"],
                    "fashion": results["fashion"],
                    "frame_count": frame_count,
                }))
            except Exception as exc:
                logger.warning("WebSocket %s: send_text failed — %s", session_id, exc)
                break

            # Store embeddings + frames in background so it never blocks the response
            if results["faces"]:
                asyncio.create_task(
                    _store_faces(session_id, results, frame.copy(), frame_count)
                )

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("WebSocket unhandled error (session %s): %s", session_id, exc)
    finally:
        _processing_sessions.discard(session_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_summary(session_id: str, detections: list) -> dict:
    if not detections:
        return {
            "session_id": session_id,
            "total_detections": 0,
            "avg_age": None,
            "age_range": None,
            "accessories": {},
            "fashion_items": {},
            "captured_faces": [],
        }

    ages = [int(d["age"]) for d in detections if d.get("age") and str(d["age"]).isdigit()]
    avg_age = round(sum(ages) / len(ages)) if ages else None

    acc_counts: dict[str, int] = {}
    fashion_counts: dict[str, int] = {}
    for d in detections:
        for item in json.loads(d.get("accessories", "[]")):
            acc_counts[item] = acc_counts.get(item, 0) + 1
        for item in json.loads(d.get("fashion", "[]")):
            fashion_counts[item] = fashion_counts.get(item, 0) + 1
        for item in json.loads(d.get("watches", "[]")):
            acc_counts[item] = acc_counts.get(item, 0) + 1

    captured_faces = [
        {
            "timestamp": d.get("timestamp"),
            "age": d.get("age"),
            "frame_path": d.get("frame_path"),
            "confidence": d.get("confidence"),
        }
        for d in detections
        if d.get("frame_path")
    ][:20]

    return {
        "session_id": session_id,
        "total_detections": len(detections),
        "avg_age": avg_age,
        "age_range": {"min": min(ages), "max": max(ages)} if ages else None,
        "accessories": acc_counts,
        "fashion_items": fashion_counts,
        "captured_faces": captured_faces,
    }
