import asyncio
import base64
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pipeline import DetectionPipeline
from embeddings import get_face_embedding
from storage import store_detection, get_session_results, find_similar_faces
from trainer import save_sample, start_finetune_async

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_PATH = os.getenv("FRAMES_PATH", os.path.join(_PROJECT_ROOT, "data", "frames"))
os.makedirs(FRAMES_PATH, exist_ok=True)

_pipeline: DetectionPipeline | None = None
# Throttle frame saves per session: max one save every 2 seconds
_last_save: dict[str, float] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    print("Initialising detection pipeline…")
    _pipeline = DetectionPipeline()
    print("Pipeline ready.")
    yield
    _pipeline.shutdown()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
                sessions[sid] = {"session_id": sid, "first_seen": m.get("timestamp"), "last_seen": m.get("timestamp"), "detections": 0}
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
    except Exception as e:
        return {"total_sessions": 0, "sessions": [], "error": str(e)}


@app.post("/api/session/start")
async def start_session():
    return {"session_id": str(uuid.uuid4())}


@app.post("/api/session/{session_id}/end")
async def end_session(session_id: str):
    detections = get_session_results(session_id)
    return _build_summary(session_id, detections)


class FeedbackRequest(BaseModel):
    real_age: int


@app.post("/api/session/{session_id}/feedback")
async def session_feedback(session_id: str, body: FeedbackRequest):
    if not (1 <= body.real_age <= 120):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Age must be between 1 and 120"}, status_code=422)

    detections = get_session_results(session_id)
    frame_paths = [
        str(Path(FRAMES_PATH) / d["frame_path"])
        for d in detections
        if d.get("frame_path")
    ]
    ages = [int(d["age"]) for d in detections if d.get("age") and str(d["age"]).isdigit()]
    corrected_avg = round(sum(ages) / len(ages)) if ages else None
    # Reverse the bias to recover the raw model prediction for accurate bias computation
    current_bias = _pipeline._age_bias if _pipeline else 0.0
    predicted_age = round(corrected_avg - current_bias) if corrected_avg is not None else None

    stats = save_sample(session_id, body.real_age, predicted_age, frame_paths)

    if _pipeline:
        _pipeline.reload_bias()

    training_started = False
    if stats["can_finetune"] and _pipeline:
        training_started = start_finetune_async(_pipeline)

    return {
        "status": "saved",
        "n_samples": stats["n_samples"],
        "bias": stats["bias"],
        "training_started": training_started,
    }


@app.get("/api/session/{session_id}/summary")
async def get_summary(session_id: str):
    detections = get_session_results(session_id)
    return _build_summary(session_id, detections)


@app.get("/api/session/{session_id}/timeline")
async def get_timeline(session_id: str):
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


@app.get("/api/frames/{filename}")
async def get_frame(filename: str):
    # Sanitise: strip path separators
    safe_name = Path(filename).name
    path = Path(FRAMES_PATH) / safe_name
    if not path.exists():
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(path), media_type="image/jpeg")


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

        embedding = await loop.run_in_executor(None, get_face_embedding, face_roi)

        frame_path = None
        if save_allowed:
            fname = f"{session_id}_{frame_count}.jpg"
            cv2.imwrite(str(Path(FRAMES_PATH) / fname), face_roi)
            frame_path = fname
            _last_save[session_id] = now
            save_allowed = False

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


# ---------------------------------------------------------------------------
# WebSocket — live frame streaming
# ---------------------------------------------------------------------------

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if _pipeline:
        _pipeline.reset_for_new_session()
    frame_count = 0
    loop = asyncio.get_event_loop()

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") != "frame":
                continue

            img_bytes = base64.b64decode(msg["data"])
            arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            results = await loop.run_in_executor(None, _pipeline.process_frame, frame)
            frame_count += 1

            # Send detection to browser IMMEDIATELY — don't wait for storage
            await websocket.send_text(json.dumps({
                "type": "detection",
                "faces": results["faces"],
                "watches": results["watches"],
                "fashion": results["fashion"],
                "frame_count": frame_count,
            }))

            # Store embeddings + frames in background so it never blocks the response
            if results["faces"]:
                asyncio.create_task(
                    _store_faces(session_id, results, frame.copy(), frame_count)
                )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error (session {session_id}): {e}")


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
