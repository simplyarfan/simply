#!/usr/bin/env python3
"""Long-running process that holds the model for Match 2.

start.sh launches this when the container starts. app.py (what the harness
actually calls) is a thin client that talks to us over a unix socket, so the
model loads once instead of once per image.
"""

from __future__ import annotations

import json
import os
import socket
import time
import traceback
from pathlib import Path

import wire
from ocr import PLATE_SIGN_PROMPT, clean_reading, load_image

BOOT = time.time()
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/qwen3-vl-8b")
LOG_DIR = Path(os.environ.get("SIMPLY_LOG_DIR", "/app/logs"))

MODEL = None


def log(msg):
    print(f"[server {time.time() - BOOT:6.1f}s] {msg}", flush=True)


def write_log(entry):
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "images.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def warm_up():
    """The first generate() on a fresh process is slow while kernels get
    compiled and tuned. Pay for that now, not on the first graded image."""
    from PIL import Image
    t0 = time.time()
    MODEL.read_image(Image.new("RGB", (64, 64), "white"), "Reply with the word ready.",
                     max_new_tokens=4, max_time=120)
    log(f"warm-up took {time.time() - t0:.1f}s")


def do_ocr(path: str, deadline: float) -> dict:
    t0 = time.time()
    if MODEL is None:
        return {"text": "", "confidence": 0.0}
    raw = MODEL.read_image(load_image(path), PLATE_SIGN_PROMPT, max_new_tokens=48,
                           max_time=deadline - time.time() - 1.5)
    text = clean_reading(raw)
    write_log({"image": path, "raw": raw, "text": text, "seconds": round(time.time() - t0, 2)})
    return {"text": text, "confidence": 0.9 if text else 0.0}


def handle(req: dict) -> dict:
    op = req.get("op")
    if op == "ping":
        return {"ok": True, "model": MODEL is not None, "uptime": time.time() - BOOT}
    if op == "ocr":
        return do_ocr(req["path"], float(req["deadline"]))
    return {"ok": False, "error": f"unknown op {op!r}"}


def main():
    global MODEL
    log(f"starting, model dir {MODEL_DIR}")
    try:
        from llm import Model
        MODEL = Model(MODEL_DIR)
        warm_up()
    except Exception:  # noqa: BLE001
        # Keep serving anyway: app.py still gets valid (empty) answers instead
        # of hanging, and the log says exactly what went wrong.
        log("model failed to load or warm up:\n" + traceback.format_exc())

    try:
        os.unlink(wire.SOCKET_PATH)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(wire.SOCKET_PATH)
    srv.listen(16)
    log(f"listening on {wire.SOCKET_PATH}")

    while True:
        conn, _ = srv.accept()
        with conn:
            try:
                conn.settimeout(10)
                req = wire.recv(conn)
                conn.settimeout(None)
                resp = handle(req)
            except Exception:  # noqa: BLE001
                log("request failed:\n" + traceback.format_exc())
                resp = {"text": "", "confidence": 0.0}
            try:
                wire.send(conn, resp)
            except OSError:
                log("client went away before we answered")  # it already wrote its fallback


if __name__ == "__main__":
    main()
