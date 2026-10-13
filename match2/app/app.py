#!/usr/bin/env python3
"""What the Match 2 harness runs, once per image:

    python3 /app/app.py --input-image /app/input/image_01.png

This file never touches the model. The model lives in server.py, which
start.sh launched when the container came up. We pass the image path over a
unix socket and write down what comes back.

The one promise this file keeps no matter what: every image leaves a valid
/app/output/<name>_output.json behind, in time. A blank reading costs one
image. A missing file or a timeout can cost the whole run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wire  # noqa: E402

OUTPUT_DIR = Path(os.environ.get("MC2_OUTPUT_DIR", "/app/output"))

# the harness gives 30s per image. Reading one takes a second or two; the
# margin is for a first image that arrives while the model is still loading.
IMAGE_SECONDS = float(os.environ.get("IMAGE_SECONDS", "25"))


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)  # the harness never sees a half-written file


def server_running() -> bool:
    """Is server.py (or the start.sh loop that restarts it) alive?"""
    me = str(os.getpid())
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or p.name == me:
            continue
        try:
            argv = (p / "cmdline").read_bytes().split(b"\0")[:3]
        except OSError:
            continue
        if any(a.endswith((b"server.py", b"start.sh")) for a in argv):
            return True
    return False


def start_server() -> None:
    # The container's CMD normally does this. If whoever ran the container
    # replaced the CMD, start it ourselves, detached so it outlives this exec.
    print("no server running, starting one", file=sys.stderr)
    subprocess.Popen(
        ["/bin/sh", str(HERE / "start.sh")],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True, close_fds=True,
    )


def run_image(image: Path) -> int:
    started = time.time()
    # image_01.png -> image_01_output.json
    out = OUTPUT_DIR / f"{image.stem}_output.json"
    result = {"text": "", "confidence": 0.0}
    deadline = started + IMAGE_SECONDS
    try:
        if not os.path.exists(wire.SOCKET_PATH) and not server_running():
            start_server()
        while not os.path.exists(wire.SOCKET_PATH) and time.time() < deadline - 4:
            time.sleep(0.5)
        resp = wire.ask(
            {"op": "ocr", "path": str(image.resolve()), "deadline": deadline},
            timeout=max(1.0, deadline + 1 - time.time()),
        )
        if isinstance(resp.get("text"), str):
            result = {"text": resp["text"], "confidence": float(resp.get("confidence") or 0.0)}
    except Exception as e:  # noqa: BLE001
        print(f"{image.name}: answering blank ({e})", file=sys.stderr)
    finally:
        write_atomic(out, result)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-image", type=Path, required=True, help="the image to read")
    args = ap.parse_args()
    return run_image(args.input_image)


if __name__ == "__main__":
    raise SystemExit(main())
