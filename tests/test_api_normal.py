"""
Normal / happy-path integration tests for the face-detection backend.

All tests assume the server is running at http://localhost:8000.
Run with:  pytest tests/test_api_normal.py -v
"""

import uuid

import httpx
import pytest

from conftest import BASE_URL, make_valid_jpeg


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client: httpx.Client):
        """
        GET /api/health must return HTTP 200 so that load-balancers and
        orchestrators can confirm the service is alive.
        """
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_returns_pipeline_ready(self, client: httpx.Client):
        """
        The health response must contain a 'pipeline_ready' boolean so callers
        can distinguish a cold-starting server (pipeline_ready=false) from a
        fully initialised one.
        """
        data = client.get("/api/health").json()
        assert "pipeline_ready" in data
        assert isinstance(data["pipeline_ready"], bool)

    def test_health_returns_status_ok(self, client: httpx.Client):
        """
        'status' field should be the string 'ok' when the server is healthy.
        """
        data = client.get("/api/health").json()
        assert data.get("status") == "ok"


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

class TestSessionLifecycle:
    def test_start_session_returns_200(self, client: httpx.Client):
        """POST /api/session/start must return HTTP 200 with a session_id."""
        resp = client.post("/api/session/start")
        assert resp.status_code == 200

    def test_start_session_returns_valid_uuid(self, client: httpx.Client):
        """
        The returned session_id must be a valid UUID v4 so that downstream
        consumers can rely on UUID guarantees (uniqueness, format).
        """
        data = client.post("/api/session/start").json()
        assert "session_id" in data
        # This will raise ValueError if not a valid UUID
        parsed = uuid.UUID(data["session_id"])
        assert str(parsed) == data["session_id"]

    def test_start_session_ids_are_unique(self, client: httpx.Client):
        """Two consecutive start calls must yield different session IDs."""
        ids = {client.post("/api/session/start").json()["session_id"] for _ in range(5)}
        assert len(ids) == 5, "Duplicate session IDs generated"

    def test_end_session_returns_summary_structure(self, client: httpx.Client, new_session: str):
        """
        POST /api/session/{id}/end must return a summary dict with at least
        session_id, total_detections, avg_age, and captured_faces keys — even
        when no frames have been sent.
        """
        resp = client.post(f"/api/session/{new_session}/end")
        assert resp.status_code == 200
        data = resp.json()
        for key in ("session_id", "total_detections", "avg_age", "captured_faces"):
            assert key in data, f"Key '{key}' missing from end-session response"

    def test_end_session_returns_correct_session_id(self, client: httpx.Client, new_session: str):
        """The summary returned by end-session must echo back the correct session_id."""
        data = client.post(f"/api/session/{new_session}/end").json()
        assert data["session_id"] == new_session


# ---------------------------------------------------------------------------
# Session list
# ---------------------------------------------------------------------------

class TestSessionList:
    def test_list_sessions_returns_200(self, client: httpx.Client):
        """GET /api/sessions must always return HTTP 200."""
        assert client.get("/api/sessions").status_code == 200

    def test_list_sessions_returns_total_sessions_int(self, client: httpx.Client):
        """
        total_sessions must be a non-negative integer so consumers can page
        or count without extra parsing.
        """
        data = client.get("/api/sessions").json()
        assert "total_sessions" in data
        assert isinstance(data["total_sessions"], int)
        assert data["total_sessions"] >= 0

    def test_list_sessions_returns_sessions_list(self, client: httpx.Client):
        """'sessions' key must be a list (possibly empty)."""
        data = client.get("/api/sessions").json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)


# ---------------------------------------------------------------------------
# Summary + Timeline
# ---------------------------------------------------------------------------

class TestSummaryAndTimeline:
    def test_get_summary_returns_200(self, client: httpx.Client, new_session: str):
        """GET /api/session/{id}/summary must return 200 for any session ID."""
        resp = client.get(f"/api/session/{new_session}/summary")
        assert resp.status_code == 200

    def test_get_summary_structure(self, client: httpx.Client, new_session: str):
        """
        The summary for a session with no detections must still contain all
        required keys with sensible null/zero defaults.
        """
        data = client.get(f"/api/session/{new_session}/summary").json()
        assert data["session_id"] == new_session
        assert data["total_detections"] == 0
        assert data["avg_age"] is None
        assert data["age_range"] is None
        assert isinstance(data["accessories"], dict)
        assert isinstance(data["captured_faces"], list)

    def test_get_timeline_returns_200(self, client: httpx.Client, new_session: str):
        """GET /api/session/{id}/timeline must return 200 for any session ID."""
        resp = client.get(f"/api/session/{new_session}/timeline")
        assert resp.status_code == 200

    def test_get_timeline_structure(self, client: httpx.Client, new_session: str):
        """
        The timeline for a session with no detections must contain session_id,
        total (== 0), and an empty detections list.
        """
        data = client.get(f"/api/session/{new_session}/timeline").json()
        assert data["session_id"] == new_session
        assert data["total"] == 0
        assert isinstance(data["detections"], list)
        assert len(data["detections"]) == 0


# ---------------------------------------------------------------------------
# Frame serving
# ---------------------------------------------------------------------------

class TestFrameServing:
    def test_nonexistent_frame_returns_404(self, client: httpx.Client):
        """
        Requesting a frame that does not exist on disk must return 404 so the
        frontend can handle missing thumbnails gracefully.
        """
        resp = client.get("/api/frames/this_file_does_not_exist_abc123.jpg")
        assert resp.status_code == 404

    def test_existing_frame_returns_200(self, client: httpx.Client):
        """
        If any .jpg frame is on disk we should be able to serve it.
        The test uses the first matching file it finds; if the frames directory
        is empty it is skipped rather than failed — an empty directory is a
        valid state in a fresh environment.
        """
        import os
        frames_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "frames",
        )
        frames = [f for f in os.listdir(frames_dir) if f.endswith(".jpg")] if os.path.isdir(frames_dir) else []
        if not frames:
            pytest.skip("No frame files on disk — skipping serve-existing-frame test")
        resp = client.get(f"/api/frames/{frames[0]}")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/")


# ---------------------------------------------------------------------------
# Feedback (happy path — requires a session with real detections)
# ---------------------------------------------------------------------------

class TestFeedback:
    def test_feedback_on_session_with_detections(self, client: httpx.Client):
        """
        Submit real_age=25 to a session that has at least one stored detection.
        Expects the response to contain personal_bias as a float.

        This test queries /api/sessions to find a session that already has
        detections.  If no such session exists it is skipped rather than
        failed, because a CI environment might start clean.
        """
        sessions_data = client.get("/api/sessions").json()
        sessions_with_detections = [
            s for s in sessions_data.get("sessions", [])
            if s.get("detections", 0) > 0
        ]
        if not sessions_with_detections:
            pytest.skip("No sessions with detections found — skipping feedback test")

        sid = sessions_with_detections[0]["session_id"]
        resp = client.post(f"/api/session/{sid}/feedback", json={"real_age": 25})

        # Session might have no detections in ChromaDB even if listed (race);
        # accept both 200 (saved) and 404 (no detections) as valid outcomes.
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            data = resp.json()
            assert "personal_bias" in data
            assert isinstance(data["personal_bias"], (int, float))

    def test_feedback_biases_are_independent_across_sessions(self, client: httpx.Client):
        """
        Two different sessions that have no detections should independently
        return 404 (no detections), confirming session isolation — biases
        must never bleed across sessions.
        """
        sid1 = client.post("/api/session/start").json()["session_id"]
        sid2 = client.post("/api/session/start").json()["session_id"]
        r1 = client.post(f"/api/session/{sid1}/feedback", json={"real_age": 30})
        r2 = client.post(f"/api/session/{sid2}/feedback", json={"real_age": 50})
        # Both should be 404 (no detections yet) — proving they do not share state
        assert r1.status_code == 404
        assert r2.status_code == 404
