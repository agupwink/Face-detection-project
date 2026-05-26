"""
Negative / edge-case integration tests for the face-detection backend.

These tests deliberately send bad inputs, missing data, path traversal
attempts, and unusual session IDs to verify that the backend degrades
gracefully and never panics or leaks stack traces.

All tests assume the server is running at http://localhost:8000.
Run with:  pytest tests/test_api_negative.py -v
"""

import uuid

import httpx
import pytest

from conftest import BASE_URL


# ---------------------------------------------------------------------------
# Feedback — age validation (Pydantic field_validator enforces 1..120)
# ---------------------------------------------------------------------------

class TestFeedbackAgeValidation:
    """real_age must be an integer in [1, 120]; everything else → 422."""

    def _post_feedback(self, client: httpx.Client, payload: dict) -> httpx.Response:
        """Helper: POST feedback to a freshly created session."""
        sid = client.post("/api/session/start").json()["session_id"]
        return client.post(f"/api/session/{sid}/feedback", json=payload)

    def test_age_zero_returns_422(self, client: httpx.Client):
        """
        real_age=0 is below the minimum of 1.
        Pydantic's field_validator must reject this with 422 Unprocessable Entity.
        """
        resp = self._post_feedback(client, {"real_age": 0})
        assert resp.status_code == 422

    def test_age_121_returns_422(self, client: httpx.Client):
        """
        real_age=121 exceeds the maximum of 120.
        Must be rejected before the handler logic runs.
        """
        resp = self._post_feedback(client, {"real_age": 121})
        assert resp.status_code == 422

    def test_age_negative_returns_422(self, client: httpx.Client):
        """
        Negative ages are physiologically impossible; must be rejected (422).
        """
        resp = self._post_feedback(client, {"real_age": -1})
        assert resp.status_code == 422

    def test_age_large_negative_returns_422(self, client: httpx.Client):
        """Large negative integer must also be rejected (422)."""
        resp = self._post_feedback(client, {"real_age": -9999})
        assert resp.status_code == 422

    def test_age_string_returns_422(self, client: httpx.Client):
        """
        Pydantic expects real_age to be an integer; passing a string must
        trigger a type-coercion failure → 422.
        """
        resp = self._post_feedback(client, {"real_age": "twenty-five"})
        assert resp.status_code == 422

    def test_age_float_string_is_coerced_or_rejected(self, client: httpx.Client):
        """
        Pydantic v2 coerces '25.0' to int 25 (valid age), so the request
        proceeds to session lookup — expect 422 (validation fail) or 404
        (session not found after coercion succeeds). Both are acceptable.
        """
        resp = self._post_feedback(client, {"real_age": "25.0"})
        assert resp.status_code in (422, 404)

    def test_missing_real_age_field_returns_422(self, client: httpx.Client):
        """
        Omitting the required real_age field entirely must return 422.
        The error response should come from Pydantic's model validation.
        """
        resp = self._post_feedback(client, {})
        assert resp.status_code == 422

    def test_null_real_age_returns_422(self, client: httpx.Client):
        """Passing null/None for real_age must be rejected (422)."""
        resp = self._post_feedback(client, {"real_age": None})
        assert resp.status_code == 422

    def test_invalid_json_body_returns_422(self, client: httpx.Client):
        """
        Sending a raw non-JSON string body must result in 422 from FastAPI's
        request parsing layer, never a 500.
        """
        sid = client.post("/api/session/start").json()["session_id"]
        resp = client.post(
            f"/api/session/{sid}/feedback",
            content=b"this is not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_empty_json_body_returns_422(self, client: httpx.Client):
        """An empty JSON object {} has no real_age → 422."""
        sid = client.post("/api/session/start").json()["session_id"]
        resp = client.post(f"/api/session/{sid}/feedback", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Feedback — session not found / no detections
# ---------------------------------------------------------------------------

class TestFeedbackSessionErrors:
    def test_feedback_nonexistent_session_returns_404(self, client: httpx.Client):
        """
        Feedback for a session UUID that was never created must return 404.
        The storage layer returns an empty list for unknown sessions.
        """
        fake_id = str(uuid.uuid4())
        resp = client.post(f"/api/session/{fake_id}/feedback", json={"real_age": 30})
        assert resp.status_code == 404

    def test_feedback_fresh_session_no_detections_returns_404(self, client: httpx.Client):
        """
        A brand-new session with no WebSocket frames submitted yet has zero
        detections; feedback must return 404 not 500.
        """
        sid = client.post("/api/session/start").json()["session_id"]
        resp = client.post(f"/api/session/{sid}/feedback", json={"real_age": 25})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Frame serving — path traversal + file-not-found
# ---------------------------------------------------------------------------

class TestFrameServing:
    def test_nonexistent_jpg_returns_404(self, client: httpx.Client):
        """
        Requesting a .jpg that does not exist on disk must return 404.
        The server must not return 200 with empty body or 500.
        """
        resp = client.get("/api/frames/does_not_exist_abc123.jpg")
        assert resp.status_code == 404

    def test_path_traversal_etc_passwd(self, client: httpx.Client):
        """
        Attempting path traversal via ../../../etc/passwd must be blocked.
        Expected: 400 (bad request / invalid filename) or 404 (sanitised name
        doesn't exist).  Never 200 or 500.
        """
        resp = client.get("/api/frames/../../../etc/passwd")
        assert resp.status_code in (400, 403, 404, 422)

    def test_path_traversal_relative_secrets(self, client: httpx.Client):
        """
        ../../secrets.txt is another traversal attempt.
        After stripping path components the safe_name becomes 'secrets.txt',
        which has a .txt extension and is blocked by the MIME whitelist → 400.
        Or it simply doesn't exist → 404.  Either is acceptable; 200 is not.
        """
        resp = client.get("/api/frames/../../secrets.txt")
        assert resp.status_code in (400, 403, 404, 422)

    def test_path_traversal_absolute_path(self, client: httpx.Client):
        """
        An absolute path like /etc/passwd should be sanitised away.
        """
        resp = client.get("/api/frames/%2Fetc%2Fpasswd")
        assert resp.status_code in (400, 403, 404, 422)

    def test_unsupported_extension_returns_400(self, client: httpx.Client):
        """
        A .txt file must be rejected before even checking the filesystem,
        because the MIME whitelist only allows .jpg/.jpeg/.png.
        """
        resp = client.get("/api/frames/somefile.txt")
        assert resp.status_code == 400

    def test_unsupported_extension_exe_returns_400(self, client: httpx.Client):
        """
        .exe files must never be served regardless of whether they exist.
        """
        resp = client.get("/api/frames/malware.exe")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Session endpoints — graceful handling of unknown session IDs
# ---------------------------------------------------------------------------

class TestUnknownSessions:
    def test_summary_fake_id_returns_gracefully(self, client: httpx.Client):
        """
        GET /api/session/{fake-id}/summary with a UUID that has no data must
        return 200 with total_detections == 0 — never a 500.
        """
        fake_id = str(uuid.uuid4())
        resp = client.get(f"/api/session/{fake_id}/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("total_detections") == 0

    def test_timeline_fake_id_returns_gracefully(self, client: httpx.Client):
        """
        GET /api/session/{fake-id}/timeline with no data must return 200 with
        total == 0 and an empty detections list.
        """
        fake_id = str(uuid.uuid4())
        resp = client.get(f"/api/session/{fake_id}/timeline")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("total") == 0

    def test_end_fake_session_returns_gracefully(self, client: httpx.Client):
        """
        POST /api/session/{fake-id}/end must return 200 with a sensible (empty)
        summary, not a 404 or 500.
        """
        fake_id = str(uuid.uuid4())
        resp = client.post(f"/api/session/{fake_id}/end")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("session_id") == fake_id


# ---------------------------------------------------------------------------
# Unusual session IDs
# ---------------------------------------------------------------------------

class TestUnusualSessionIds:
    def test_extremely_long_session_id(self, client: httpx.Client):
        """
        A 1000-character session_id must not crash the server.
        The server should either return 200 with empty data or a 4xx client error.
        """
        long_id = "a" * 1000
        resp = client.get(f"/api/session/{long_id}/summary")
        assert resp.status_code in (200, 400, 404, 422)

    def test_special_characters_in_session_id(self, client: httpx.Client):
        """
        Session IDs containing URL-encoded special characters must not cause a
        500.  FastAPI may reject them with 422 or treat them as unknown IDs
        returning empty data (200).
        """
        # Use percent-encoded form so httpx passes it as a path param correctly
        resp = client.get("/api/session/foo%40bar%21/summary")
        assert resp.status_code in (200, 400, 404, 422)

    def test_null_byte_in_session_id(self, client: httpx.Client):
        """
        A null byte (%00) in the session ID is a common injection technique.
        The server must respond with a non-500 status code.
        """
        resp = client.get("/api/session/abc%00def/summary")
        assert resp.status_code in (200, 400, 404, 422)

    def test_slash_in_session_id(self, client: httpx.Client):
        """
        Slashes in the session_id path segment could cause unexpected routing.
        The URL would resolve to a different path; either 404 or an expected
        response is acceptable.
        """
        resp = client.get("/api/session/abc%2Fdef/summary")
        assert resp.status_code in (200, 400, 404, 422)


# ---------------------------------------------------------------------------
# Unsupported HTTP methods
# ---------------------------------------------------------------------------

class TestUnsupportedMethods:
    def test_delete_health_returns_405(self, client: httpx.Client):
        """
        DELETE /api/health is not a registered route; FastAPI must return 405.
        """
        resp = client.request("DELETE", "/api/health")
        assert resp.status_code == 405

    def test_put_sessions_returns_405(self, client: httpx.Client):
        """
        PUT /api/sessions is not a registered route; must return 405.
        """
        resp = client.request("PUT", "/api/sessions")
        assert resp.status_code == 405

    def test_patch_session_start_returns_405(self, client: httpx.Client):
        """
        PATCH /api/session/start is not a registered route; must return 405.
        """
        resp = client.request("PATCH", "/api/session/start")
        assert resp.status_code == 405

    def test_delete_session_returns_405(self, client: httpx.Client):
        """
        DELETE on an individual session endpoint is not registered; must return 405.
        """
        fake_id = str(uuid.uuid4())
        resp = client.request("DELETE", f"/api/session/{fake_id}/summary")
        assert resp.status_code == 405
