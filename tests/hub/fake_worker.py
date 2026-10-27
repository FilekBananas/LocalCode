"""A stand-in for ``localcode.hub.download_worker`` used by the download tests.

It speaks the same stdin/stdout protocol but downloads nothing: it writes the
requested number of bytes into ``<dest>/.cache/localcode/<file>.part`` and
finally into ``<dest>/<file>``, so the manager's progress accounting, cancel
handling and partial-file cleanup can be tested deterministically and offline.

Behaviour is driven by environment variables:

``LC_FAKE_MODE``   ``ok`` (default) · ``error`` · ``crash`` · ``garbage`` · ``ignore_term``
``LC_FAKE_STEPS``  progress events per file (default 4)
``LC_FAKE_DELAY``  seconds between progress events (default 0.02)
``LC_FAKE_PID_FILE``  when set, the worker's pid is written there
``LC_FAKE_SPEC_FILE`` when set, the job spec received on stdin is written there
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

PART_DIR = Path(".cache") / "localcode"


def emit(event: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    mode = os.environ.get("LC_FAKE_MODE", "ok")
    steps = int(os.environ.get("LC_FAKE_STEPS", "4"))
    delay = float(os.environ.get("LC_FAKE_DELAY", "0.02"))
    pid_file = os.environ.get("LC_FAKE_PID_FILE")
    if pid_file:
        Path(pid_file).write_text(str(os.getpid()))
    if mode == "ignore_term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    spec = json.loads(sys.stdin.readline())
    spec_file = os.environ.get("LC_FAKE_SPEC_FILE")
    if spec_file:
        Path(spec_file).write_text(json.dumps(spec, ensure_ascii=False))
    dest = Path(spec["dest"])
    files = spec["files"]
    total = sum(int(f.get("size") or 0) for f in files)
    if mode == "garbage":
        sys.stdout.write("not json at all\n")
        sys.stdout.flush()
    emit({"event": "start", "done": 0, "total": total})
    done = 0
    for f in files:
        name = str(f["name"])
        size = int(f.get("size") or 0)
        part = dest / PART_DIR / (name + ".part")
        part.parent.mkdir(parents=True, exist_ok=True)
        with open(part, "wb") as fh:
            for step in range(1, steps + 1):
                if mode == "error" and step == 2:
                    emit({"event": "error", "code": "network",
                          "message": "Brak połączenia z Hugging Face. Sprawdź połączenie z internetem.",
                          "file": name})
                    return 1
                if mode == "crash" and step == 2:
                    os._exit(3)
                chunk = size * step // steps - fh.tell()
                fh.write(b"\0" * max(0, chunk))
                fh.flush()
                emit({"event": "progress", "file": name, "done": done + fh.tell(), "total": total})
                time.sleep(delay)
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(part, target)
        done += size
        emit({"event": "progress", "file": name, "done": done, "total": total})
        emit({"event": "file_done", "file": name})
    emit({"event": "done"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
