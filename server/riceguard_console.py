"""
server/riceguard_console.py -- interactive control console for the RiceGuard
FastAPI server.

This is a CONTROLLER, not a second backend: it launches server/main.py's
existing uvicorn process as a child process (the same command
run_server.ps1 already used: the project venv's python -m uvicorn main:app
--host 0.0.0.0 --port <port>), and derives everything it shows (health,
models, discovery, live requests, clients) from that one real process --
either by parsing its own stdout log stream (two extra INFO-level log lines
were added around POST /predict in main.py purely for this; nothing about
the pipeline changed) or by asking it directly (GET /health). No inference
logic, no second HTTP server, no new persistent state.

DESIGN NOTE (read before changing the output/rendering code below): an
earlier version of this file tried to pin a live status bar to the
terminal's last row via a VT100 scroll region + cursor save/restore,
redrawn on a fixed timer. That is a GUI-panel trick wearing a terminal's
clothes, and it is genuinely fragile under Windows' legacy console host
(conhost) -- scroll-region support there is inconsistent across window
resizes and terminal profiles. It was deliberately removed. What replaced
it:
  - Plain shell commands (no leading '/'): start, stop, restart, status,
    health, logs, clients, models, info, clear, help, ls, quit, exit.
  - Arrow-key history, Left/Right cursor movement, Home/End, Backspace/
    Delete all come from the Windows console host's own native line-input
    editing (ReadConsole) for a plain `input()` call -- true on cmd.exe,
    conhost, AND Windows Terminal, with no library or custom key-handling
    code needed. This file does not reimplement a line editor.
  - Ctrl+C raises KeyboardInterrupt around the input loop and exits the
    CONSOLE only (never the server) -- standard Python behavior, not
    custom-built.
  - Realtime telemetry is a periodic ONE-LINE entry appended to normal
    scrollback (see _telemetry_loop), printed only when something
    meaningfully changed (or a heartbeat interval elapses) -- never a
    redrawn panel, never interferes with an in-progress input line beyond
    the same interleaving any real background process (docker logs -f,
    tail -f, a chat bot) would produce in a shared terminal.
  - Ctrl+L is NOT intercepted as a "clear screen" keystroke: doing so would
    require replacing input() with a raw low-level key reader (msvcrt),
    which is exactly the "custom text-entry system" this design avoids.
    Use the `clear` command instead, which does the same thing.

`quit`/`exit` stops the CONSOLE only. The server process, if running, is
left running in the background -- this console never stops the server
except when the user explicitly types `stop`.

Colors reuse the Android app's own brand palette (android/app/.../ui/theme/
Color.kt -- forest green primary, rice-gold accent, desaturated status hues)
via 24-bit ANSI, applied narrowly (a keyword here, a header there) rather
than as filled panels -- restrained terminal color, not a dashboard. Color
is applied only when stdout is a real interactive terminal with VT
processing confirmed enabled (COLOR_ENABLED); piped/redirected output (log
files, automated checks) always gets the plain-text form, byte-identical to
running with color support unavailable, so nothing downstream ever has to
parse escape codes.

Run: start_riceguard_server.bat (project root), or directly:
    .venv\\Scripts\\python.exe server\\riceguard_console.py
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_DIR = PROJECT_ROOT / "server"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_PORT = 8000
CLIENT_ACTIVE_WINDOW_SEC = 120  # a client is "ACTIVE" if seen within this window, else dropped from 'clients'
LOG_HISTORY = 500
GRACEFUL_STOP_TIMEOUT_SEC = 6
TELEMETRY_POLL_SEC = 5      # how often the background thread checks for a change worth reporting
TELEMETRY_HEARTBEAT_SEC = 60  # print a "still alive" line at least this often even if nothing changed

ACCESS_LOG_RE = re.compile(
    r'INFO:\s+(?P<ip>[\d.]+):\d+ - "(?P<method>\w+) (?P<path>[^\s"]+) HTTP/\S+" (?P<status>\d{3})'
)
PREDICT_START_RE = re.compile(r"predict_start ip=(?P<ip>\S+)")
PREDICT_DONE_RE = re.compile(
    r"predict_done ip=(?P<ip>\S+) status=(?P<status>\d+) duration=(?P<duration>[\d.]+)s(?: result=(?P<result>.+))?"
)
MDNS_UP_RE = re.compile(r"mDNS: advertising")
MDNS_DOWN_RE = re.compile(r"mDNS: RiceGuard Server advertisement stopped")
MODELS_READY_RE = re.compile(r"Models loaded\. RiceGuard AI server ready\.")


class ServerState:
    """Everything the console shows is either live-derived from the child
    process's own log stream (this object) or fetched fresh from /health on
    demand -- nothing here is a substitute for that real check."""

    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.start_time: float | None = None
        self.discovery_active = False
        self.models_loaded = False
        self.port = DEFAULT_PORT
        self.logs: deque[str] = deque(maxlen=LOG_HISTORY)
        self.clients: dict[str, float] = {}  # ip -> last-seen unix ts
        self.total = 0
        self.success = 0
        self.failed = 0
        self.active = 0
        self.last_request_time: str | None = None
        self.last_prediction: str | None = None
        self.lock = threading.Lock()

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def reset_counters(self) -> None:
        with self.lock:
            self.discovery_active = False
            self.models_loaded = False
            self.clients.clear()
            self.total = self.success = self.failed = self.active = 0
            self.last_request_time = None
            self.last_prediction = None


state = ServerState()
_print_lock = threading.Lock()

# ------------------------------------------------------------------- color

IS_TTY = sys.stdout.isatty()
COLOR_ENABLED = False  # flipped on in main(), only after VT processing is confirmed enabled


class Ansi:
    """24-bit ANSI codes lifted from the Android app's own brand palette
    (ui/theme/Color.kt), applied to individual words/labels only -- never as
    a filled block -- so the terminal reads as the same product without
    looking like a GUI panel pasted into a shell."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    # Forest green (brand primary) -- healthy / active / success
    GREEN = "\033[38;2;46;139;102m"      # ForestGreen60 #2E8B66
    # Rice gold (brand accent) -- headers / highlights
    GOLD = "\033[38;2;201;162;39m"       # RiceGold60 #C9A227
    # Amber -- warnings (distinct from GOLD, which is reserved for headers)
    AMBER = "\033[38;2;217;119;6m"       # warm amber #D97706
    # Status red -- errors only
    RED = "\033[38;2;176;57;42m"         # StatusError #B0392A
    # Neutrals
    GRAY = "\033[38;2;146;155;142m"      # Neutral60 #929B8E -- timestamps/secondary info


def c(text: str, *codes: str) -> str:
    """Wraps `text` in the given ANSI codes -- a no-op (returns `text`
    unchanged) whenever COLOR_ENABLED is False, so every plain-text
    assertion already exercised against this console's piped output stays
    valid whether or not color is on."""
    if not COLOR_ENABLED or not codes:
        return text
    return "".join(codes) + text + Ansi.RESET


def status_mark(ok: bool) -> str:
    """A colored checkmark/cross when color is on, the original bracketed
    ASCII form otherwise (never emitted in non-color mode, so a redirected
    log file never contains a glyph that might not round-trip)."""
    if COLOR_ENABLED:
        return c("✓", Ansi.GREEN, Ansi.BOLD) if ok else c("✗", Ansi.RED, Ansi.BOLD)
    return "[OK]" if ok else "[FAIL]"


def _header(text: str) -> None:
    """A single colored header line -- no border, no box, no repeated '='
    bar. Terminal telemetry/log tools (systemctl status, htop, etc.) use a
    plain colored label for this, not a drawn panel."""
    out(c(text, Ansi.BOLD, Ansi.GOLD))


def _kv(label: str, value: str, value_color: str | None = None, width: int = 13) -> str:
    """Formats one 'Label : value' line with the label muted and the value
    optionally colored -- the one formatting primitive every status-style
    command below uses, so the whole console has one consistent look."""
    rendered_value = c(value, value_color, Ansi.BOLD) if value_color else value
    return f"{c(label.ljust(width), Ansi.GRAY)}: {rendered_value}"


# ---------------------------------------------------------------- terminal

def _enable_vt_processing() -> bool:
    """Windows' real console host (conhost/Windows Terminal) does not
    interpret ANSI/VT100 escape sequences unless a process explicitly opts
    in via SetConsoleMode -- without this, the color codes below would
    print as raw escape characters instead of being rendered. No-op (and
    harmless) on any platform/terminal that already understands VT
    sequences natively."""
    if os.name != "nt":
        return True
    try:
        kernel32 = ctypes.windll.kernel32
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        STD_OUTPUT_HANDLE = -11
        handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        if not kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING):
            return False
        return True
    except Exception:  # noqa: BLE001
        return False


def out(line: str = "") -> None:
    """Prints one line to normal scrollback. Nothing in this file ever
    moves the cursor, clears a region, or redraws in place -- every line
    printed here stays in scrollback exactly like a real shell's output."""
    with _print_lock:
        print(line)


def local_ip() -> str:
    """The address a phone on the same Wi-Fi can actually reach, not just
    any address this machine happens to have (matches the routing-table
    trick server/discovery.py's advertiser already uses)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "unknown"


def http_get_json(path: str, timeout: float = 3.0) -> tuple[bool, dict | str]:
    """Returns (ok, body_dict) on a real 200 JSON response, or (False,
    reason_str) -- never fabricates a response. This is the ONLY thing
    'health' and 'models' trust; log-derived state is never substituted for
    it -- a real check, not a cached status."""
    url = f"http://127.0.0.1:{state.port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return True, json.loads(body)
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return False, f"unreachable ({exc.reason})"
    except (TimeoutError, socket.timeout):
        return False, "timed out"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


# ---------------------------------------------------------------- log stream

def _colorize_raw_log_line(line: str) -> str:
    """Light-touch coloring for the raw child-process log lines shown by
    `logs` -- these are the real, unmodified stdout of the uvicorn process,
    not reformatted, so this only tints (never rewrites) based on an
    obvious ERROR/WARNING marker already in the text."""
    upper = line.upper()
    if "ERROR" in upper or "TRACEBACK" in upper:
        return c(line, Ansi.RED)
    if "WARN" in upper:
        return c(line, Ansi.AMBER)
    return c(line, Ansi.GRAY)


def _parse_line(line: str) -> None:
    if MDNS_UP_RE.search(line):
        with state.lock:
            state.discovery_active = True
    elif MDNS_DOWN_RE.search(line):
        with state.lock:
            state.discovery_active = False
    if MODELS_READY_RE.search(line):
        with state.lock:
            state.models_loaded = True

    m = PREDICT_START_RE.search(line)
    if m:
        # Counted for the Active-requests telemetry, but deliberately not
        # printed here -- a real access log reports the OUTCOME of a
        # request, not a separate "started" line for every in-flight one.
        with state.lock:
            state.active += 1
        return

    m = PREDICT_DONE_RE.search(line)
    if m:
        status, duration, result = m.group("status"), m.group("duration"), m.group("result")
        now = datetime.now().strftime("%H:%M:%S")
        with state.lock:
            state.active = max(0, state.active - 1)
            state.total += 1
            if status == "200":
                state.success += 1
            else:
                state.failed += 1
            state.last_request_time = now
            state.clients[m.group("ip")] = time.time()
            if result and result != "n/a":
                state.last_prediction = result
        status_colored = c(status, Ansi.GREEN if status == "200" else Ansi.RED, Ansi.BOLD)
        tail = f"  {c(result, Ansi.GOLD, Ansi.BOLD)}" if result and result != "n/a" else ""
        out(
            f"{c(now, Ansi.GRAY)} {c('INFO', Ansi.GRAY)}  {c('/predict', Ansi.GOLD):<9}  "
            f"{m.group('ip'):<15}  {status_colored}  {duration}s{tail}"
        )
        return

    m = ACCESS_LOG_RE.search(line)
    if m and not m.group("path").startswith("/predict"):
        with state.lock:
            state.clients[m.group("ip")] = time.time()
        if m.group("path").startswith("/health"):
            status_colored = c(m.group("status"), Ansi.GREEN if m.group("status") == "200" else Ansi.RED, Ansi.BOLD)
            out(
                f"{c(datetime.now().strftime('%H:%M:%S'), Ansi.GRAY)} {c('INFO', Ansi.GRAY)}  "
                f"{c('/health', Ansi.GOLD):<9}  {m.group('ip'):<15}  {status_colored}"
            )


def _reader_thread(proc: subprocess.Popen) -> None:
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n")
        if not line:
            continue
        with state.lock:
            state.logs.append(line)
        try:
            _parse_line(line)
        except Exception:  # noqa: BLE001 -- a parse hiccup must never kill the reader thread
            pass
    state.reset_counters()


# -------------------------------------------------------------- server control

def cmd_start() -> None:
    if state.running():
        out("Server is already running.")
        return

    port = int(os.environ.get("RICEGUARD_PORT", str(DEFAULT_PORT)))
    if port_in_use(port):
        out("")
        out(c(f"ERROR: Port {port} is already in use.", Ansi.RED, Ansi.BOLD))
        out("Use 'status' or 'info' to investigate.")
        out("")
        return
    if not VENV_PYTHON.exists():
        out(c(f"ERROR: Could not find the project venv at {VENV_PYTHON}", Ansi.RED, Ansi.BOLD))
        out("Run this from the project's existing .venv setup.")
        return

    out("Starting RiceGuard server...")
    state.port = port
    _telemetry_reset()
    env = dict(os.environ)
    env["RICEGUARD_PORT"] = str(port)
    env["PYTHONUNBUFFERED"] = "1"  # otherwise uvicorn's log lines sit in a pipe buffer and the console lags

    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    try:
        proc = subprocess.Popen(
            [str(VENV_PYTHON), "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(port)],
            cwd=str(SERVER_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creationflags,
        )
    except OSError as exc:
        out(c(f"ERROR: Failed to launch server process: {exc}", Ansi.RED, Ansi.BOLD))
        return

    state.process = proc
    state.start_time = time.time()
    threading.Thread(target=_reader_thread, args=(proc,), daemon=True).start()

    # Wait (bounded) for the real startup markers rather than assuming success.
    deadline = time.time() + 90  # first CUDA/model load can take a while
    while time.time() < deadline:
        if proc.poll() is not None:
            out(c("ERROR: Server process exited during startup -- see 'logs' for details.", Ansi.RED, Ansi.BOLD))
            state.process = None
            return
        with state.lock:
            models_ready = state.models_loaded
        if models_ready:
            break
        time.sleep(0.3)
    else:
        out(c("Server did not report ready within 90s -- check 'logs'; it may still come up.", Ansi.AMBER))

    ok, body = http_get_json("/health", timeout=5)
    healthy = ok and isinstance(body, dict) and body.get("status") == "ok"

    out(f"{status_mark(True)} FastAPI started")
    out(f"{status_mark(state.models_loaded)} Models loaded")
    out(f"{status_mark(state.discovery_active)} Discovery active")
    out(f"{status_mark(healthy)} API healthy" + ("" if healthy else f" ({body if not ok else 'models not ready'})"))
    out("")
    out(c("Server is READY.", Ansi.GREEN, Ansi.BOLD) if healthy
        else c("Server started but is not fully healthy yet -- check 'health' shortly.", Ansi.AMBER))


def cmd_stop() -> None:
    if not state.running():
        out("Server is not running.")
        return

    out("Stopping RiceGuard server...")
    proc = state.process
    assert proc is not None
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
        proc.wait(timeout=GRACEFUL_STOP_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        out(c("Server did not exit gracefully in time -- forcing shutdown.", Ansi.AMBER))
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    except Exception as exc:  # noqa: BLE001
        out(c(f"ERROR while stopping server: {exc}", Ansi.RED, Ansi.BOLD))

    was_discovery_active = state.discovery_active
    state.process = None
    state.start_time = None
    _telemetry_reset()
    out(f"{status_mark(True)} Server stopped")
    out(f"{status_mark(not state.discovery_active)} Discovery stopped")
    if was_discovery_active and state.discovery_active:
        out(c("(Discovery may not have unregistered cleanly -- it will still expire from mDNS caches on its own TTL.)", Ansi.GRAY))


def cmd_restart() -> None:
    cmd_stop()
    out("")
    cmd_start()


# ------------------------------------------------------------------ commands

def _uptime_str() -> str:
    if not state.start_time:
        return "--:--:--"
    secs = int(time.time() - state.start_time)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _gpu_info() -> tuple[str, str]:
    """Shells out to nvidia-smi directly -- deliberately NOT asking the
    inference process for this, so it works even without touching the
    server, and never blocks on a busy GPU worker."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return "N/A (nvidia-smi not found)", ""
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return "N/A (nvidia-smi query failed)", ""
        name, used, total = [p.strip() for p in result.stdout.strip().splitlines()[0].split(",")]
        return name, f"{float(used) / 1024:.1f} / {float(total) / 1024:.1f} GB"
    except Exception:  # noqa: BLE001
        return "N/A", ""


def _gpu_used_gb() -> str | None:
    """Compact 'usedGB' form for the one-line telemetry entry only -- the
    full 'used / total GB' breakdown still appears in `status`/`models`."""
    _, mem = _gpu_info()
    if not mem or "/" not in mem:
        return None
    used = mem.split("/")[0].strip()
    return f"{used}GB"


def cmd_status() -> None:
    running = state.running()
    ok, body = http_get_json("/health", timeout=3) if running else (False, "server not running")
    healthy = ok and isinstance(body, dict) and body.get("status") == "ok"
    gpu_name, gpu_mem = _gpu_info() if running else ("--", "")

    with state.lock:
        clients = sorted(ip for ip, ts in state.clients.items() if time.time() - ts <= CLIENT_ACTIVE_WINDOW_SEC)
        total, success, failed, active = state.total, state.success, state.failed, state.active
        last_req, last_pred = state.last_request_time, state.last_prediction
        discovery = state.discovery_active

    _header("RICEGUARD SERVER")
    out(_kv("Status", "RUNNING" if running else "STOPPED", Ansi.GREEN if running else Ansi.GRAY))
    out(_kv("API Health", "HEALTHY" if healthy else (str(body) if not running else "UNHEALTHY"), Ansi.GREEN if healthy else Ansi.RED))
    out(_kv("Discovery", "ACTIVE" if discovery else "STOPPED", Ansi.GREEN if discovery else Ansi.GRAY))
    out(_kv("IP Address", local_ip()))
    out(_kv("Port", str(state.port)))
    out(_kv("Uptime", _uptime_str()))
    out("")
    _header("Models")
    if isinstance(body, dict):
        out(_kv("Disease", "LOADED" if body.get("resnet50") else "NOT LOADED", Ansi.GREEN if body.get("resnet50") else Ansi.RED))
        out(_kv("Detection", "LOADED" if body.get("yolo") else "NOT LOADED", Ansi.GREEN if body.get("yolo") else Ansi.RED))
        device = body.get("device")
        device_str = f"CUDA / {gpu_name}" if device and gpu_name not in ("--", "") and "N/A" not in gpu_name else str(device or "--")
        out(_kv("Device", device_str))
    else:
        out(_kv("Disease", "UNKNOWN (not running)", Ansi.GRAY))
        out(_kv("Detection", "UNKNOWN (not running)", Ansi.GRAY))
    if gpu_mem:
        out(_kv("GPU Memory", gpu_mem))
    out("")
    _header("Network")
    out(_kv("Clients", str(len(clients))))
    out(_kv("Active", str(active)))
    out("")
    _header("Requests")
    out(_kv("Total", str(total)))
    out(_kv("Successful", str(success), Ansi.GREEN if success else None))
    out(_kv("Failed", str(failed), Ansi.RED if failed else None))
    out(_kv("Last Request", last_req or "--"))
    out(_kv("Last Result", last_pred or "--", Ansi.GOLD if last_pred else None))


def cmd_health() -> None:
    if not state.running():
        out(f"{status_mark(False)} API responding -- server is not running.")
        out("Run 'start' first.")
        return
    ok, body = http_get_json("/health", timeout=5)
    if not ok:
        out(f"{status_mark(False)} API responding -- {body}")
        return
    out(f"{status_mark(True)} API responding")
    models_ready = bool(body.get("models_ready"))
    out(f"{status_mark(bool(body.get('resnet50')))} Disease model loaded")
    out(f"{status_mark(bool(body.get('yolo')))} Detection model loaded")
    device = body.get("device")
    out(f"{status_mark(bool(device))} Inference device available" + (f" ({device})" if device else ""))
    out(f"{status_mark(models_ready)} Required services ready")
    out(f"{status_mark(state.discovery_active)} Discovery active")
    if not models_ready:
        out("")
        out(c(f"Reason: {body.get('status')!r} -- models_ready is false. Check 'logs' for load errors.", Ansi.AMBER))


def cmd_models() -> None:
    if not state.running():
        out("Server is not running -- start it with 'start' to check model status.")
        return
    ok, body = http_get_json("/health", timeout=5)
    if not ok:
        out(c(f"Could not reach the server: {body}", Ansi.RED))
        return
    gpu_name, gpu_mem = _gpu_info()
    out(_kv("Disease (ResNet50)", "LOADED" if body.get("resnet50") else "NOT LOADED", Ansi.GREEN if body.get("resnet50") else Ansi.RED, width=19))
    out(_kv("Detection (YOLOv8n)", "LOADED" if body.get("yolo") else "NOT LOADED", Ansi.GREEN if body.get("yolo") else Ansi.RED, width=19))
    out(_kv("Inference device", str(body.get("device") or "unknown"), width=19))
    out(_kv("GPU", gpu_name, width=19))
    if gpu_mem:
        out(_kv("GPU memory", gpu_mem, width=19))
    out(_kv("Queue depth", str(body.get("queue_depth", 0)), width=19))
    out(_kv("Model status", "READY" if body.get("models_ready") else "NOT READY", Ansi.GREEN if body.get("models_ready") else Ansi.RED, width=19))


def cmd_clients() -> None:
    with state.lock:
        clients = sorted(state.clients.items(), key=lambda kv: -kv[1])
        active_count = state.active
    now = time.time()
    live = [(ip, ts) for ip, ts in clients if now - ts <= CLIENT_ACTIVE_WINDOW_SEC]
    _header("Connected Clients")
    if not live:
        out(c("No clients seen recently.", Ansi.GRAY))
    else:
        # Only the IP is something this server can actually know -- there is
        # no device-identifying header sent by the Android client today, so
        # this deliberately does not invent a phone model/name for it.
        for ip, ts in live:
            age = int(now - ts)
            is_active = age < 20
            label = "ACTIVE" if is_active else f"idle {age}s"
            out(f"{ip:<16}{c(label, Ansi.GREEN if is_active else Ansi.GRAY)}")
    out("")
    out(_kv("Active requests", str(active_count)))


def cmd_logs(count: int = 20) -> None:
    with state.lock:
        lines = list(state.logs)[-count:]
    if not lines:
        out(c("No log output yet.", Ansi.GRAY))
        return
    for line in lines:
        out(_colorize_raw_log_line(line))


def cmd_info() -> None:
    ok, body = http_get_json("/health", timeout=2) if state.running() else (False, None)
    fastapi_version = "unknown"
    try:
        import importlib.metadata
        fastapi_version = importlib.metadata.version("fastapi")
    except Exception:  # noqa: BLE001
        pass
    _header("Server Info")
    out(_kv("Server name", body.get("service") if ok and isinstance(body, dict) else "RiceGuard", width=19))
    out(_kv("Project path", str(PROJECT_ROOT), width=19))
    out(_kv("Python env", str(VENV_PYTHON if VENV_PYTHON.exists() else sys.executable), width=19))
    out(_kv("FastAPI version", fastapi_version, width=19))
    out(_kv("Local IP", local_ip(), width=19))
    out(_kv("Port", str(state.port), width=19))
    out(_kv("Discovery service", "RiceGuard Server (_riceguard._tcp)", width=19))
    out(_kv("Uptime", _uptime_str() if state.running() else "--", width=19))
    out(_kv("OS", platform.platform(), width=19))


_HELP_ROWS = [
    ("start", "Start server"),
    ("stop", "Stop server"),
    ("restart", "Restart server"),
    ("status", "Server status"),
    ("health", "Real health check"),
    ("logs [n]", "Recent logs"),
    ("clients", "Connected clients"),
    ("models", "Model/GPU status"),
    ("info", "Server/network info"),
    ("clear", "Clear terminal"),
    ("help", "Command help"),
    ("exit", "Exit console"),
]


def cmd_help() -> None:
    out("Commands:")
    for name, desc in _HELP_ROWS:
        out(f"  {c(name.ljust(10), Ansi.GOLD)}{desc}")


def cmd_clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _banner() -> str:
    """A small, compact branded banner."""
    width = 40
    top = "╔" + "═" * width + "╗"
    bottom = "╚" + "═" * width + "╝"
    title = "RICEGUARD SERVER".center(width)
    lines = [
        c(top, Ansi.GOLD),
        c("║", Ansi.GOLD) + c(title, Ansi.BOLD, Ansi.GREEN) + c("║", Ansi.GOLD),
        c(bottom, Ansi.GOLD),
        "",
        f"Type {c(chr(39) + 'help' + chr(39), Ansi.GOLD)} for commands. No automatic startup -- type {c(chr(39) + 'start' + chr(39), Ansi.GOLD)} to launch the server.",
    ]
    return "\n".join(lines)


def dispatch(raw: str) -> bool:
    """Returns False to exit the console loop."""
    cmd = raw.strip()
    if not cmd:
        return True
    parts = cmd.split()
    head = parts[0].lower()

    if head in ("quit", "exit"):
        # Deliberately does NOT stop the server: the console only stops the
        # server when the user explicitly types 'stop'. If a server is
        # running, it is left running as a detached background process.
        if state.running():
            out(c(f"Leaving the server running on port {state.port}.", Ansi.AMBER))
        out("Goodbye.")
        return False
    if head in ("ls", "help"):
        cmd_help()
    elif head == "start":
        cmd_start()
    elif head == "stop":
        cmd_stop()
    elif head == "restart":
        cmd_restart()
    elif head == "status":
        cmd_status()
    elif head == "health":
        cmd_health()
    elif head == "models":
        cmd_models()
    elif head == "clients":
        cmd_clients()
    elif head == "info":
        cmd_info()
    elif head == "clear":
        cmd_clear()
    elif head == "logs":
        n = 20
        if len(parts) > 1 and parts[1].isdigit():
            n = int(parts[1])
        cmd_logs(n)
    else:
        out(c(f"Unknown command: {raw.strip()}", Ansi.RED))
        out("Type 'help' or 'ls' for available commands.")
    return True


# ------------------------------------------------------------- telemetry

# Snapshot of the fields that TRIGGER a printed telemetry line -- GPU memory
# and total-scan-count are deliberately excluded from this comparison (see
# _telemetry_loop) even though they appear IN the printed line, because they
# drift on their own and would otherwise defeat "don't repeat unchanged
# status."
_telemetry_last_key: tuple | None = None
_telemetry_last_emit: float = 0.0
_telemetry_lock = threading.Lock()


def _telemetry_reset() -> None:
    """Called on start/stop so a fresh server run gets its own baseline --
    otherwise the first poll after `start` could compare against
    whatever was true the last time the server ran and stay silent."""
    global _telemetry_last_key, _telemetry_last_emit
    with _telemetry_lock:
        _telemetry_last_key = None
        _telemetry_last_emit = 0.0


def _telemetry_line() -> tuple[tuple, str] | None:
    """Returns (comparison_key, rendered_text), or None if the server isn't
    running (no telemetry is printed while stopped -- `start`/`stop`
    already announce those transitions themselves)."""
    if not state.running():
        return None
    ok, body = http_get_json("/health", timeout=2)
    healthy = ok and isinstance(body, dict) and body.get("status") == "ok"
    with state.lock:
        clients = sum(1 for ts in state.clients.values() if time.time() - ts <= CLIENT_ACTIVE_WINDOW_SEC)
        active, total = state.active, state.total
        discovery = state.discovery_active

    key = (healthy, discovery, clients, active)

    running_part = c("RUNNING", Ansi.GREEN, Ansi.BOLD)
    api_part = c("API OK", Ansi.GREEN) if healthy else c("API DOWN", Ansi.RED, Ansi.BOLD)
    mdns_part = c("mDNS ON", Ansi.GREEN) if discovery else c("mDNS OFF", Ansi.AMBER)
    gpu = _gpu_used_gb()
    gpu_part = f" | {c('GPU', Ansi.GRAY)} {gpu}" if gpu else ""
    text = (
        f"{c('RiceGuard', Ansi.BOLD, Ansi.GREEN)} | {running_part} | {api_part} | {mdns_part} "
        f"| Clients {clients} | Active {active} | {total} scans{gpu_part}"
    )
    return key, text


def _telemetry_loop() -> None:
    global _telemetry_last_key, _telemetry_last_emit
    while not _telemetry_stop.wait(TELEMETRY_POLL_SEC):
        try:
            result = _telemetry_line()
        except Exception:  # noqa: BLE001 -- a telemetry hiccup must never kill the console
            continue
        if result is None:
            continue
        key, text = result
        now = time.time()
        with _telemetry_lock:
            changed = key != _telemetry_last_key
            stale = (now - _telemetry_last_emit) >= TELEMETRY_HEARTBEAT_SEC
            if not (changed or stale):
                continue
            _telemetry_last_key = key
            _telemetry_last_emit = now
        out(f"{c(datetime.now().strftime('%H:%M:%S'), Ansi.GRAY)} {c('INFO', Ansi.GRAY)}  {text}")


_telemetry_stop = threading.Event()
_telemetry_thread: threading.Thread | None = None


def _start_telemetry() -> None:
    global _telemetry_thread
    _telemetry_thread = threading.Thread(target=_telemetry_loop, daemon=True)
    _telemetry_thread.start()


def _stop_telemetry() -> None:
    _telemetry_stop.set()


def _prompt_str() -> str:
    return f"{c('RiceGuard', Ansi.BOLD, Ansi.GREEN)}> "


def main() -> None:
    global COLOR_ENABLED

    # Windows' console default codepage (cp1252 etc.) can't encode arbitrary
    # glyphs and would otherwise crash on the first status line; force UTF-8
    # and never raise on an unrenderable glyph instead of losing the whole
    # session over one character (see also `chcp 65001` in
    # start_riceguard_server.bat, which gets the terminal itself to actually
    # *display* these glyphs correctly rather than just not-crash on them).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    # Must happen BEFORE the first colored line is printed (the banner) --
    # enabling VT processing after already emitting escape codes would leave
    # the first frame's worth of output as raw, un-rendered characters.
    COLOR_ENABLED = IS_TTY and _enable_vt_processing()

    out(_banner())
    out("")
    _start_telemetry()
    try:
        while True:
            try:
                raw = input(_prompt_str())
            except EOFError:
                break
            if not dispatch(raw):
                break
    except KeyboardInterrupt:
        # Ctrl+C exits the console but -- same rule as 'quit'/'exit' -- never
        # stops the server on its own.
        out("")
        if state.running():
            out(c(f"Leaving the server running on port {state.port}.", Ansi.AMBER))
        out("Goodbye.")
    finally:
        _stop_telemetry()


if __name__ == "__main__":
    main()
