"""
server/security.py

Everything in project brief section 22 ("Security / privacy for the local
prototype") that isn't specific to the prediction flow itself:

  - LocalNetworkOnlyMiddleware: rejects any request whose client IP is not
    in a private/loopback range. This is a local-network prototype, not an
    internet-facing service -- it never claims stronger guarantees than that
    (a determined actor already on the LAN is out of scope, same as any
    other unauthenticated LAN demo tool).
  - request IDs (RG-YYYYMMDD-NNNN, matching the project brief's own example)
  - filename sanitization (never trust a client-supplied filename for a
    server-side path)
  - safe result-file resolution for GET /results/{request_id}/{filename},
    which must never let a client read anything outside its own request's
    temp directory (path traversal via ".." or an absolute path is rejected).
"""

from __future__ import annotations

import ipaddress
import itertools
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

_REQUEST_ID_RE = re.compile(r"^RG-\d{8}-\d{4,}$")
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

_PRIVATE_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]


def is_private_address(host: str) -> bool:
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False  # not a parseable IP (e.g. a hostname) -- reject rather than guess
    return any(addr in net for net in _PRIVATE_NETWORKS)


class LocalNetworkOnlyMiddleware(BaseHTTPMiddleware):
    """Rejects any request whose client address is not private/loopback.
    A phone and a PC on the same home/lab Wi-Fi both get private addresses
    (typically 192.168.x.x), which is the deployment this app is built for
    (project brief section 5: "local Wi-Fi" only, no cloud upload)."""

    async def dispatch(self, request: Request, call_next):
        client = request.client
        host = client.host if client else None
        if host is None or not is_private_address(host):
            return JSONResponse(
                {"status": "error", "message": "This server only accepts requests from the local network."},
                status_code=403,
            )
        return await call_next(request)


class RequestIdGenerator:
    """RG-YYYYMMDD-NNNN, monotonically increasing within the process for a
    given day. A local-prototype server is expected to run as a single
    process (project brief section 23: one GPU, one worker, no multi-process
    deployment), so an in-memory counter is sufficient -- it resets on
    restart, which for a demo/prototype is an acceptable, documented
    limitation rather than a correctness issue (IDs are unique for the
    lifetime of one run, which is all any single demo session needs)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._day = None
        self._counter = itertools.count(1)

    def next_id(self) -> str:
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        with self._lock:
            if today != self._day:
                self._day = today
                self._counter = itertools.count(1)
            n = next(self._counter)
        return f"RG-{today}-{n:04d}"


def is_valid_request_id(request_id: str) -> bool:
    return bool(_REQUEST_ID_RE.match(request_id))


def sanitize_upload_filename(original_name: str | None, fallback_ext: str = ".jpg") -> str:
    """The client's original filename is NEVER used to build a server-side
    path -- only its extension (if it's a recognized image extension) is
    kept, purely for cosmetic/debugging purposes; the actual on-disk name is
    always request-id-derived (see PipelineService)."""
    ext = fallback_ext
    if original_name:
        suffix = Path(original_name).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
            ext = suffix
    return f"upload{ext}"


def resolve_result_file(base_dir: Path, request_id: str, filename: str) -> Path | None:
    """Safe path resolution for GET /results/{request_id}/{filename}.
    Returns None (caller returns 404) rather than raising, for any of:
    a malformed request_id, a filename containing a path separator or '..',
    or a resolved path that escapes base_dir/request_id -- so a client can
    never be told anything about the server's filesystem beyond "found" or
    "not found" inside its own request's own result folder."""
    if not is_valid_request_id(request_id):
        return None
    if not filename or not _SAFE_FILENAME_RE.match(filename):
        return None
    request_dir = (base_dir / request_id).resolve()
    candidate = (request_dir / filename).resolve()
    try:
        candidate.relative_to(request_dir)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate
