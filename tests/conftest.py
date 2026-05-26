"""
Shared pytest fixtures for the face-detection backend integration test suite.

All tests are integration tests that require the backend to be running at
http://localhost:8000 before pytest is invoked.
"""

import io
import os
import struct
import uuid
import zlib

import httpx
import pytest

BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# Basic HTTP client
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def base_url() -> str:
    """Return the base URL of the running backend."""
    return BASE_URL


@pytest.fixture(scope="session")
def client() -> httpx.Client:
    """
    Long-lived synchronous httpx client for the entire test session.
    Using session scope keeps connection overhead minimal.
    """
    with httpx.Client(base_url=BASE_URL, timeout=15.0) as c:
        yield c


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def new_session(client: httpx.Client) -> str:
    """
    Create a fresh session via POST /api/session/start and return its session_id.
    Each test that needs a session gets its own independent UUID.
    """
    resp = client.post("/api/session/start")
    resp.raise_for_status()
    return resp.json()["session_id"]


@pytest.fixture
def nonexistent_session_id() -> str:
    """Return a well-formed UUID that is guaranteed not to exist in the DB."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def make_valid_jpeg(width: int = 64, height: int = 64) -> bytes:
    """
    Generate a minimal but valid JPEG image in memory using only stdlib + Pillow.
    Returns raw bytes suitable for file uploads or base64 encoding.
    """
    try:
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (width, height), color=(128, 64, 32))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()
    except ImportError:
        # Fallback: build a tiny JPEG via raw bytes (1×1 red pixel)
        # Source: https://github.com/mathiasbynens/small/blob/master/jpeg.jpg
        return (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
            b"186 9=4 ;7554?\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
            b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
            b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04"
            b"\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa"
            b'\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n'
            b"\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJKLMNOPQRSTUVWXYZ"
            b"cdefghijklmnopqrstuvwxyz\xff\xda\x00\x08\x01\x01\x00\x00?\x00"
            b"\xfb\xd2\x8a(\x00\xff\xd9"
        )


def make_corrupted_file() -> bytes:
    """
    Return bytes that look like a JPEG header but have garbage payload.
    Useful for testing that endpoints handle corrupt uploads gracefully.
    """
    return b"\xff\xd8\xff\xe0" + b"\x00" * 10 + b"\xde\xad\xbe\xef" * 50


@pytest.fixture
def valid_jpeg_bytes() -> bytes:
    """Pytest fixture exposing make_valid_jpeg() as bytes."""
    return make_valid_jpeg()


@pytest.fixture
def corrupted_file_bytes() -> bytes:
    """Pytest fixture exposing make_corrupted_file() as bytes."""
    return make_corrupted_file()
