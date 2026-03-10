#!/usr/bin/env python3
"""
Lightweight process monitor HTTP server for FADA eval jobs.
Single file, minimal deps (flask only), Linux /proc scanning.

Endpoints:
    GET  /health      - server liveness
    GET  /processes   - monitored processes with CPU/mem/runtime
    GET  /events      - unprocessed completion events
    POST /events/ack  - acknowledge events (body: {"ids": [...]})
    GET  /gpu         - nvidia-smi summary

All endpoints require Authorization: Bearer <token> header.
Token is read from MONITOR_API_TOKEN environment variable.
"""
import functools
import hmac
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from flask import Flask, jsonify, request

logging.basicConfig(
    level=logging.INFO,
    format="[monitor] %(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent / "monitor_config.json"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


CFG = load_config()
EVENTS_FILE = Path(CFG["events_file"])
TRACKED_FILE = EVENTS_FILE.parent / "tracked.json"
POLL_INTERVAL = CFG.get("poll_interval_seconds", 15)
WATCH_PATTERNS = CFG["watch_patterns"]
PORT = CFG.get("port", 9731)
_EVENT_MAX_AGE_DAYS = 7

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
API_TOKEN = os.environ.get("MONITOR_API_TOKEN", "")
if not API_TOKEN:
    raise RuntimeError(
        "MONITOR_API_TOKEN environment variable must be set. "
        "Refusing to start without authentication."
    )


def require_token(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "missing bearer token"}), 401
        token = auth[7:]
        if not hmac.compare_digest(token, API_TOKEN):
            return jsonify({"error": "invalid token"}), 403
        return fn(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Cmdline sanitisation
# ---------------------------------------------------------------------------
_SECRET_PATTERNS = re.compile(
    r"(--(?:hf-token|api-key|password|secret|wandb-key|token|auth))"
    r"[\s=](\S+)",
    re.IGNORECASE,
)


def _safe_cmdline(cmdline: str) -> str:
    return _SECRET_PATTERNS.sub(r"\1 <REDACTED>", cmdline)


# ---------------------------------------------------------------------------
# Event store (in-memory + file-backed)
# ---------------------------------------------------------------------------
_events: OrderedDict[str, dict] = OrderedDict()
_events_lock = threading.Lock()


def _save_events():
    EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Trim acked events older than 7 days
    cutoff = time.time() - _EVENT_MAX_AGE_DAYS * 86400
    stale = [
        eid for eid, ev in _events.items()
        if ev["acked"] and ev["ts"] < cutoff
    ]
    for eid in stale:
        del _events[eid]
    tmp = EVENTS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(list(_events.values()), f, indent=2)
    tmp.rename(EVENTS_FILE)


def _load_events():
    if EVENTS_FILE.exists():
        try:
            with open(EVENTS_FILE) as f:
                for ev in json.load(f):
                    _events[ev["id"]] = ev
        except (json.JSONDecodeError, KeyError):
            log.warning("Corrupt events file %s, backing up to .bak", EVENTS_FILE)
            bak = EVENTS_FILE.with_suffix(".bak")
            shutil.copy2(EVENTS_FILE, bak)


def push_event(kind: str, data: dict):
    ev = {
        "id": str(uuid.uuid4()),
        "kind": kind,
        "ts": time.time(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": data,
        "acked": False,
    }
    with _events_lock:
        _events[ev["id"]] = ev
        _save_events()


def get_pending_events() -> list[dict]:
    with _events_lock:
        return [e for e in _events.values() if not e["acked"]]


def ack_events(ids: list[str]):
    acked = 0
    with _events_lock:
        for eid in ids:
            if eid in _events and not _events[eid]["acked"]:
                _events[eid]["acked"] = True
                acked += 1
        _save_events()
    return acked


# ---------------------------------------------------------------------------
# /proc scanning
# ---------------------------------------------------------------------------
def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.replace(b"\x00", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


_boot_time_cache: float | None = None


def _boot_time() -> float:
    global _boot_time_cache
    if _boot_time_cache is None:
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime"):
                _boot_time_cache = float(line.split()[1])
                break
        if _boot_time_cache is None:
            raise RuntimeError(
                "Could not determine boot time: no btime line in /proc/stat"
            )
    return _boot_time_cache


def _proc_stat(pid: int) -> dict | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().split()
        status_text = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None

    utime = int(stat[13])
    stime = int(stat[14])
    starttime_ticks = int(stat[21])

    clk = os.sysconf("SC_CLK_TCK")
    start_epoch = _boot_time() + (starttime_ticks / clk)
    runtime_s = time.time() - start_epoch

    vmrss_kb = 0
    for line in status_text.splitlines():
        if line.startswith("VmRSS:"):
            vmrss_kb = int(line.split()[1])
            break

    total_ticks = utime + stime
    cpu_pct_avg = round((total_ticks / clk) / max(runtime_s, 1) * 100, 1)

    return {
        "pid": pid,
        "state": stat[2],
        "cpu_pct_avg": cpu_pct_avg,
        "mem_rss_mb": round(vmrss_kb / 1024, 1),
        "runtime_s": int(runtime_s),
        "runtime_human": _fmt_duration(int(runtime_s)),
    }


def _fmt_duration(s: int) -> str:
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def scan_processes(patterns: list[str]) -> dict[int, dict]:
    result = {}
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        pid = int(pid_dir.name)
        cmd = _cmdline(pid)
        if not cmd:
            continue
        for pat in patterns:
            if pat in cmd:
                stats = _proc_stat(pid)
                if stats:
                    stats["cmdline"] = _safe_cmdline(cmd[:300])
                    stats["pattern"] = pat
                    result[pid] = stats
                break
    return result


# ---------------------------------------------------------------------------
# Tracked state persistence
# ---------------------------------------------------------------------------
def _save_tracked(tracked: dict[int, dict]):
    TRACKED_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = TRACKED_FILE.with_suffix(".tmp")
    # Convert int keys to str for JSON
    with open(tmp, "w") as f:
        json.dump({str(k): v for k, v in tracked.items()}, f, indent=2)
    tmp.rename(TRACKED_FILE)


def _load_tracked() -> dict[int, dict]:
    if not TRACKED_FILE.exists():
        return {}
    try:
        with open(TRACKED_FILE) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    except (json.JSONDecodeError, KeyError, ValueError):
        log.warning("Corrupt tracked file %s, backing up to .bak", TRACKED_FILE)
        bak = TRACKED_FILE.with_suffix(".bak")
        shutil.copy2(TRACKED_FILE, bak)
        return {}


# ---------------------------------------------------------------------------
# Background poller
# ---------------------------------------------------------------------------
_tracked: dict[int, dict] = {}
_poller_lock = threading.Lock()
_hostname = socket.gethostname()


def _poll_loop():
    global _tracked
    while True:
        try:
            current = scan_processes(WATCH_PATTERNS)

            # Collect events to emit while holding the lock, emit after release
            with _poller_lock:
                prev_pids = set(_tracked.keys())
                curr_pids = set(current.keys())
                started = [(pid, current[pid]) for pid in curr_pids - prev_pids]
                completed = [(pid, _tracked[pid]) for pid in prev_pids - curr_pids]
                _tracked = current

            # Persist tracked state (outside lock)
            _save_tracked(current)

            # Emit events outside the poller lock
            for pid, info in started:
                push_event("process_started", {
                    "pid": pid,
                    "hostname": _hostname,
                    "cmdline": info["cmdline"],
                    "pattern": info["pattern"],
                })

            for pid, prev in completed:
                push_event("process_completed", {
                    "pid": pid,
                    "hostname": _hostname,
                    "cmdline": prev["cmdline"],
                    "pattern": prev["pattern"],
                    "runtime_s": prev["runtime_s"],
                    "runtime_human": prev.get("runtime_human", ""),
                })
        except Exception as e:
            log.error("Poller error: %s", e)
        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# GPU query
# ---------------------------------------------------------------------------
_gpu_lock = threading.Lock()


def _gpu_info() -> list[dict]:
    if not _gpu_lock.acquire(blocking=False):
        return [{"error": "nvidia-smi query already in progress"}]
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            timeout=10, text=True,
        )
    except FileNotFoundError:
        return [{"error": "nvidia-smi not found"}]
    except subprocess.CalledProcessError:
        return [{"error": "nvidia-smi returned an error"}]
    except subprocess.TimeoutExpired:
        return [{"error": "nvidia-smi timed out"}]
    finally:
        _gpu_lock.release()

    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 6:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "util_pct": int(parts[2]),
                "mem_used_mb": int(parts[3]),
                "mem_total_mb": int(parts[4]),
                "temp_c": int(parts[5]),
            })
    return gpus


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.get("/health")
@require_token
def health():
    with _poller_lock:
        proc_count = len(_tracked)
    return jsonify({
        "status": "ok",
        "ts": time.time(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "watched_processes": proc_count,
        "pending_events": len(get_pending_events()),
    })


@app.get("/processes")
@require_token
def processes():
    with _poller_lock:
        procs = list(_tracked.values())
    return jsonify({"count": len(procs), "processes": procs})


@app.get("/events")
@require_token
def events():
    pending = get_pending_events()
    return jsonify({"count": len(pending), "events": pending})


@app.post("/events/ack")
@require_token
def events_ack():
    body = request.get_json(force=True, silent=True) or {}
    ids = body.get("ids", [])
    if not isinstance(ids, list):
        return jsonify({"error": "ids must be a list"}), 400
    acked = ack_events(ids)
    return jsonify({"acked": acked})


@app.get("/gpu")
@require_token
def gpu():
    return jsonify({"gpus": _gpu_info()})


# ---------------------------------------------------------------------------
# Module-level initialization (works with gunicorn and direct execution)
# ---------------------------------------------------------------------------
_load_events()
_tracked = _load_tracked()
log.info("Watching patterns: %s", WATCH_PATTERNS)
log.info("Poll interval: %ds", POLL_INTERVAL)

_poller_thread = threading.Thread(target=_poll_loop, daemon=True)
_poller_thread.start()

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("Starting on port %d (bound to 127.0.0.1)", PORT)
    app.run(host="127.0.0.1", port=PORT, threaded=True)
