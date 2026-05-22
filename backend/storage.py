import os
import json
import uuid
from datetime import datetime, timezone
from typing import Optional

import chromadb

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMADB_PATH = os.getenv("CHROMADB_PATH", os.path.join(_HERE, "data", "chromadb"))
FRAMES_PATH = os.getenv("FRAMES_PATH", os.path.join(_HERE, "data", "frames"))

os.makedirs(CHROMADB_PATH, exist_ok=True)
os.makedirs(FRAMES_PATH, exist_ok=True)

_client = chromadb.PersistentClient(path=CHROMADB_PATH)
_collection = _client.get_or_create_collection(
    name="face_detections",
    metadata={"hnsw:space": "cosine"},
)

_ZERO_EMBEDDING = [0.0] * 512


def store_detection(
    session_id: str,
    embedding: Optional[list[float]],
    age: Optional[str],
    accessories: list,
    watches: list,
    fashion: list,
    confidence: float,
    frame_path: Optional[str] = None,
) -> str:
    doc_id = str(uuid.uuid4())
    metadata = {
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "age": age or "",
        "accessories": json.dumps([a["label"] for a in accessories]),
        "watches": json.dumps([w["label"] for w in watches]),
        "fashion": json.dumps([f["label"] for f in fashion]),
        "confidence": round(confidence, 4),
        "frame_path": frame_path or "",
    }
    _collection.add(
        ids=[doc_id],
        embeddings=[embedding if embedding else _ZERO_EMBEDDING],
        metadatas=[metadata],
    )
    return doc_id


def get_session_results(session_id: str) -> list:
    try:
        results = _collection.get(
            where={"session_id": session_id},
            include=["metadatas"],
        )
        return results["metadatas"] or []
    except Exception:
        return []


def find_similar_faces(embedding: list[float], n_results: int = 5, exclude_session: Optional[str] = None) -> list:
    try:
        count = _collection.count()
        if count == 0:
            return []
        where = {"session_id": {"$ne": exclude_session}} if exclude_session else None
        results = _collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, count),
            where=where,
            include=["metadatas", "distances"],
        )
        return list(zip(results["metadatas"][0], results["distances"][0]))
    except Exception:
        return []
