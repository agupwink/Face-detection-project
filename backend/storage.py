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
_profiles = _client.get_or_create_collection(
    name="user_profiles",
    metadata={"hnsw:space": "cosine"},
)

_ZERO_EMBEDDING = [0.0] * 512
_PROFILE_MATCH_THRESHOLD = 0.4  # cosine distance — lower = stricter


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


def get_session_embedding(session_id: str) -> Optional[list[float]]:
    """Return the first real (non-zero) face embedding stored for a session."""
    try:
        results = _collection.get(
            where={"session_id": session_id},
            include=["embeddings"],
        )
        embeddings = results.get("embeddings")
        if embeddings is None:
            return None
        import numpy as np
        for emb in embeddings:
            arr = np.array(emb)
            if arr.size > 0 and np.any(arr != 0.0):
                return arr.tolist()
    except Exception:
        pass
    return None


def find_user_profile(embedding: list[float]) -> tuple:
    """Return (user_id, bias) for the closest matching profile, or (None, 0.0)."""
    try:
        if _profiles.count() == 0:
            return None, 0.0
        results = _profiles.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["metadatas", "distances"],
        )
        if not results["ids"][0]:
            return None, 0.0
        distance = results["distances"][0][0]
        if distance > _PROFILE_MATCH_THRESHOLD:
            return None, 0.0
        return results["ids"][0][0], float(results["metadatas"][0][0].get("bias", 0.0))
    except Exception:
        return None, 0.0


def update_user_profile(embedding: list[float], user_id: Optional[str], real_age: int, raw_predicted_age: Optional[int]) -> tuple:
    """Create or update a user profile. Returns (user_id, new_bias)."""
    existing = None
    if user_id:
        try:
            r = _profiles.get(ids=[user_id], include=["metadatas"])
            if r["ids"]:
                existing = r["metadatas"][0]
        except Exception:
            pass

    now = datetime.now(timezone.utc).isoformat()

    if existing:
        n = int(existing.get("n_samples", 0))
        old_bias = float(existing.get("bias", 0.0))
        if raw_predicted_age is not None:
            new_bias = round((old_bias * n + (real_age - raw_predicted_age)) / (n + 1), 2)
            n += 1
        else:
            new_bias = old_bias
        _profiles.update(
            ids=[user_id],
            embeddings=[embedding],
            metadatas=[{"real_age": real_age, "bias": new_bias, "n_samples": n, "last_seen": now}],
        )
        return user_id, new_bias
    else:
        new_id = user_id or str(uuid.uuid4())
        bias = round(float(real_age - raw_predicted_age), 2) if raw_predicted_age is not None else 0.0
        _profiles.add(
            ids=[new_id],
            embeddings=[embedding],
            metadatas=[{"real_age": real_age, "bias": bias, "n_samples": 1 if raw_predicted_age is not None else 0, "last_seen": now}],
        )
        return new_id, bias


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
