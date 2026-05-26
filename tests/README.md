# Face Detection Backend — Test Suite

Integration tests for the FastAPI backend. **The server must be running before
you run any tests** — these are integration tests that hit a live HTTP endpoint,
not unit tests with mocked dependencies.

## Quick start

```bash
# 1. Install test dependencies
pip install pytest httpx pillow

# 2. Start the backend (in a separate terminal)
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000

# 3. Run all tests
pytest tests/ -v

# 4. Run only a specific suite
pytest tests/test_api_normal.py -v
pytest tests/test_api_negative.py -v
pytest tests/test_api_concurrent.py -v
pytest tests/test_api_performance.py -v -s   # -s prints latency tables
```

## Test suites

| File | What it tests |
|------|--------------|
| `test_api_normal.py` | Happy-path: health, session lifecycle, summary, timeline, feedback |
| `test_api_negative.py` | Bad inputs, path traversal, wrong HTTP methods, edge-case session IDs |
| `test_api_concurrent.py` | 20–50 parallel requests, UUID uniqueness under load, no server crashes |
| `test_api_performance.py` | p50/p95/p99 latency, 100 sequential health checks, slow-endpoint detection |

## Notes

- Tests that require a live camera or WebSocket stream are marked
  `@pytest.mark.skip(reason="requires live camera/WebSocket")` and are
  excluded from the default run.
- Tests that require existing frame files on disk call `pytest.skip()` at
  runtime if the `data/frames/` directory is empty.
- The default timeout per request is **15 seconds** (30 s for concurrent
  tests). If the backend is under heavy load you may see occasional timeouts
  rather than assertion failures.
