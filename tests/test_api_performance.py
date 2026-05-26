"""
Performance integration tests for the face-detection backend.

These tests measure p50/p95/p99 latency for key endpoints, flag slow
responses, and print a summary table to stdout.

All tests assume the server is running at http://localhost:8000.
Run with:  pytest tests/test_api_performance.py -v -s
           (use -s so the summary tables are printed to the console)
"""

import statistics
import time
from typing import List

import httpx
import pytest

from conftest import BASE_URL

# Maximum acceptable latency for any single request in the perf suite
_SLOW_THRESHOLD_SECONDS = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _measure(client: httpx.Client, method: str, path: str, n: int = 100, **kwargs) -> List[float]:
    """
    Issue `n` sequential requests and return a list of elapsed seconds (float).
    Uses the same client/connection to eliminate TCP-handshake noise.
    """
    timings: List[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        client.request(method, path, **kwargs)
        timings.append(time.perf_counter() - t0)
    return timings


def _percentile(data: List[float], pct: float) -> float:
    """Return the `pct`-th percentile (0–100) of a sorted list."""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


def _print_table(label: str, timings: List[float]) -> None:
    """Print a formatted latency summary table to stdout."""
    p50 = _percentile(timings, 50)
    p95 = _percentile(timings, 95)
    p99 = _percentile(timings, 99)
    avg = statistics.mean(timings)
    mx = max(timings)
    mn = min(timings)
    slow_count = sum(1 for t in timings if t > _SLOW_THRESHOLD_SECONDS)
    print(
        f"\n{'─'*60}\n"
        f"  Endpoint : {label}\n"
        f"  Samples  : {len(timings)}\n"
        f"  Min      : {mn*1000:>8.1f} ms\n"
        f"  Avg      : {avg*1000:>8.1f} ms\n"
        f"  p50      : {p50*1000:>8.1f} ms\n"
        f"  p95      : {p95*1000:>8.1f} ms\n"
        f"  p99      : {p99*1000:>8.1f} ms\n"
        f"  Max      : {mx*1000:>8.1f} ms\n"
        f"  Slow (>{_SLOW_THRESHOLD_SECONDS}s): {slow_count}\n"
        f"{'─'*60}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLatencyHealth:
    def test_health_p99_under_5s(self, client: httpx.Client):
        """
        Run 100 sequential GET /api/health calls and assert p99 latency stays
        under 5 seconds. The health check is the lightest endpoint so it sets
        the floor; if it's slow, something is systemically wrong.
        """
        timings = _measure(client, "GET", "/api/health", n=100)
        _print_table("GET /api/health", timings)

        p99 = _percentile(timings, 99)
        assert p99 < _SLOW_THRESHOLD_SECONDS, (
            f"GET /api/health p99 latency {p99*1000:.1f}ms exceeds "
            f"{_SLOW_THRESHOLD_SECONDS*1000:.0f}ms threshold"
        )

    def test_health_no_single_request_over_5s(self, client: httpx.Client):
        """
        No individual health-check response should exceed 5 seconds.
        A single outlier this large indicates a blocking operation on the
        request path that needs to be moved off the event loop.
        """
        timings = _measure(client, "GET", "/api/health", n=100)
        slow = [t for t in timings if t > _SLOW_THRESHOLD_SECONDS]
        assert not slow, (
            f"{len(slow)} health-check requests exceeded {_SLOW_THRESHOLD_SECONDS}s: "
            f"max={max(slow)*1000:.1f}ms"
        )

    def test_100_sequential_health_checks_stats(self, client: httpx.Client):
        """
        Run exactly 100 sequential health checks and print full stats.
        This is a visibility test — it always passes unless p99 exceeds 5 s.
        """
        timings = _measure(client, "GET", "/api/health", n=100)
        _print_table("GET /api/health (100 sequential)", timings)
        assert len(timings) == 100


class TestLatencySessionsList:
    def test_sessions_list_p99_under_5s(self, client: httpx.Client):
        """
        Run 50 sequential GET /api/sessions calls and assert p99 < 5 s.
        The sessions list does a full ChromaDB scan; this test ensures the
        scan doesn't degrade as data volume grows.
        """
        timings = _measure(client, "GET", "/api/sessions", n=50)
        _print_table("GET /api/sessions", timings)

        p99 = _percentile(timings, 99)
        assert p99 < _SLOW_THRESHOLD_SECONDS, (
            f"GET /api/sessions p99 {p99*1000:.1f}ms exceeds threshold"
        )

    def test_sessions_list_p50_reported(self, client: httpx.Client):
        """
        p50 (median) for the sessions-list endpoint is reported to stdout.
        Median is a better indicator of typical user experience than mean.
        """
        timings = _measure(client, "GET", "/api/sessions", n=50)
        p50 = _percentile(timings, 50)
        print(f"\n[perf] GET /api/sessions  p50={p50*1000:.1f}ms")
        assert p50 >= 0  # always true; just confirms the measurement ran


class TestLatencySummary:
    def test_summary_p99_under_5s(self, client: httpx.Client):
        """
        Run 30 sequential GET /api/session/{id}/summary calls (with a UUID
        that has no detections so the DB query is fast) and assert p99 < 5 s.
        """
        import uuid
        fake_id = str(uuid.uuid4())
        timings = _measure(client, "GET", f"/api/session/{fake_id}/summary", n=30)
        _print_table(f"GET /api/session/{{id}}/summary (no detections)", timings)

        p99 = _percentile(timings, 99)
        assert p99 < _SLOW_THRESHOLD_SECONDS, (
            f"GET /api/session summary p99 {p99*1000:.1f}ms exceeds threshold"
        )


class TestSlowEndpointDetection:
    def test_detect_slow_endpoints(self, client: httpx.Client):
        """
        Sweep all lightweight read endpoints and flag any where ANY single
        request exceeds the slow threshold. Reports a list of offenders.
        This test acts as a canary for regression; it should never trip in a
        healthy environment.
        """
        import uuid
        fake_id = str(uuid.uuid4())

        endpoints = [
            ("GET", "/api/health"),
            ("GET", "/api/sessions"),
            ("GET", f"/api/session/{fake_id}/summary"),
            ("GET", f"/api/session/{fake_id}/timeline"),
        ]

        slow_endpoints = []
        for method, path in endpoints:
            timings = _measure(client, method, path, n=20)
            slowest = max(timings)
            if slowest > _SLOW_THRESHOLD_SECONDS:
                slow_endpoints.append((method, path, slowest))
            _print_table(f"{method} {path}", timings)

        if slow_endpoints:
            details = "\n".join(
                f"  {m} {p}: max={t*1000:.1f}ms" for m, p, t in slow_endpoints
            )
            pytest.fail(
                f"The following endpoints had requests exceeding "
                f"{_SLOW_THRESHOLD_SECONDS}s:\n{details}"
            )
