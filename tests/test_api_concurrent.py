"""
Concurrency integration tests for the face-detection backend.

These tests fire many requests in parallel using ThreadPoolExecutor to check
that the backend handles concurrent load without data corruption, crashes, or
deadlocks.

All tests assume the server is running at http://localhost:8000.
Run with:  pytest tests/test_api_concurrent.py -v
"""

import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pytest

from conftest import BASE_URL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client() -> httpx.Client:
    """Return a fresh httpx.Client — each thread needs its own connection."""
    return httpx.Client(base_url=BASE_URL, timeout=30.0)


def _run_parallel(fn, n_workers: int, n_tasks: int | None = None) -> list:
    """
    Run fn() n_tasks times (default = n_workers) across n_workers threads.
    Returns a list of results in completion order.
    """
    n_tasks = n_tasks or n_workers
    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(fn) for _ in range(n_tasks)]
        for future in as_completed(futures):
            results.append(future.result())
    return results


# ---------------------------------------------------------------------------
# 20 simultaneous POST /api/session/start
# ---------------------------------------------------------------------------

class TestConcurrentSessionStart:
    def test_20_simultaneous_session_starts_all_succeed(self):
        """
        Fire 20 session-start requests at the same time from separate threads.
        Every response must be HTTP 200 with a unique, valid UUID session_id.
        This verifies that uuid.uuid4() and the endpoint are thread-safe.
        """
        def start_one():
            with _make_client() as c:
                resp = c.post("/api/session/start")
                return resp.status_code, resp.json().get("session_id")

        results = _run_parallel(start_one, n_workers=20)
        statuses = [r[0] for r in results]
        ids = [r[1] for r in results]

        assert all(s == 200 for s in statuses), f"Non-200 responses: {statuses}"
        assert all(i is not None for i in ids), "Some session_ids were None"

        # All UUIDs must be valid
        parsed = [uuid.UUID(i) for i in ids]
        assert len(parsed) == 20

        # All UUIDs must be unique
        assert len(set(ids)) == 20, "Duplicate session IDs returned under concurrency"

    def test_average_response_time_session_start(self):
        """
        Measure and report average response time for POST /api/session/start
        under 20 concurrent threads. Flags if average exceeds 5 seconds.
        """
        timings = []

        def timed_start():
            with _make_client() as c:
                t0 = time.perf_counter()
                c.post("/api/session/start")
                return time.perf_counter() - t0

        timings = _run_parallel(timed_start, n_workers=20)
        avg = statistics.mean(timings)
        print(f"\n[concurrent] POST /api/session/start  avg={avg*1000:.1f}ms  "
              f"min={min(timings)*1000:.1f}ms  max={max(timings)*1000:.1f}ms")
        assert avg < 5.0, f"session/start average response time {avg:.2f}s exceeds 5s threshold"


# ---------------------------------------------------------------------------
# 20 simultaneous GET /api/sessions
# ---------------------------------------------------------------------------

class TestConcurrentSessionList:
    def test_20_simultaneous_list_sessions_all_200(self):
        """
        Fire 20 GET /api/sessions requests in parallel.
        All must return 200 — the ChromaDB read path must handle concurrent
        access without errors or race conditions.
        """
        def list_sessions():
            with _make_client() as c:
                resp = c.get("/api/sessions")
                return resp.status_code

        results = _run_parallel(list_sessions, n_workers=20)
        assert all(s == 200 for s in results), f"Non-200 responses: {results}"

    def test_average_response_time_list_sessions(self):
        """
        Measure average response time for GET /api/sessions under 20 concurrent
        requests. Reports timing; flags if average exceeds 5 seconds.
        """
        def timed_list():
            with _make_client() as c:
                t0 = time.perf_counter()
                c.get("/api/sessions")
                return time.perf_counter() - t0

        timings = _run_parallel(timed_list, n_workers=20)
        avg = statistics.mean(timings)
        print(f"\n[concurrent] GET /api/sessions  avg={avg*1000:.1f}ms  "
              f"min={min(timings)*1000:.1f}ms  max={max(timings)*1000:.1f}ms")
        assert avg < 5.0, f"sessions list average {avg:.2f}s exceeds threshold"


# ---------------------------------------------------------------------------
# 10 simultaneous feedback submissions on different sessions
# ---------------------------------------------------------------------------

class TestConcurrentFeedback:
    def test_10_simultaneous_feedback_no_crashes(self):
        """
        Submit feedback for 10 different sessions simultaneously.
        Each session has no detections so we expect 404 responses — the key
        assertion is that no request returns 500 (server error or crash).
        """
        # Pre-create 10 session IDs (each is a fresh UUID with no detections)
        sessions = [str(uuid.uuid4()) for _ in range(10)]

        def submit_feedback(sid: str):
            with _make_client() as c:
                resp = c.post(f"/api/session/{sid}/feedback", json={"real_age": 30})
                return resp.status_code

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(submit_feedback, sid): sid for sid in sessions}
            for future in as_completed(futures):
                status = future.result()
                assert status != 500, f"Session {futures[future]} returned 500"
                assert status in (200, 404), f"Unexpected status {status}"


# ---------------------------------------------------------------------------
# 50 rapid health checks
# ---------------------------------------------------------------------------

class TestRapidHealthChecks:
    def test_50_rapid_health_checks_all_200(self):
        """
        Send 50 health-check requests as fast as possible (10-thread pool).
        All must return 200, confirming no connection pool exhaustion or
        file-descriptor leak under rapid sequential hammering.
        """
        def check_health():
            with _make_client() as c:
                return c.get("/api/health").status_code

        results = _run_parallel(check_health, n_workers=10, n_tasks=50)
        failures = [r for r in results if r != 200]
        assert not failures, f"{len(failures)}/50 health checks returned non-200: {set(failures)}"

    def test_average_response_time_health(self):
        """
        Measure and report average response time for GET /api/health under
        50 requests across 10 threads. Flags if average exceeds 5 seconds.
        """
        def timed_health():
            with _make_client() as c:
                t0 = time.perf_counter()
                c.get("/api/health")
                return time.perf_counter() - t0

        timings = _run_parallel(timed_health, n_workers=10, n_tasks=50)
        avg = statistics.mean(timings)
        print(f"\n[concurrent] GET /api/health (50x)  avg={avg*1000:.1f}ms  "
              f"min={min(timings)*1000:.1f}ms  max={max(timings)*1000:.1f}ms")
        assert avg < 5.0, f"health check average {avg:.2f}s exceeds threshold"
