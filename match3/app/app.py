#!/usr/bin/env python3
"""What the harness runs, both ways:

    python3 /app/app.py --index /app/corpus
    python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."

This file never touches the model. The model lives in server.py, which
start.sh launched when the container came up. We just pass the request over a
unix socket and write down what comes back.

The one promise this file keeps no matter what: every query leaves a valid
/app/output/<query-id>_output.json behind, in time, written exactly once. A
blank answer costs one question. A missing file or a timeout can cost the
whole submission.
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
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))

# the harness gives 30s per question; we give ourselves 24 and keep the rest
QUERY_SECONDS = float(os.environ.get("QUERY_SECONDS", "24"))
# how long --index waits for the server to finish loading the model
SERVER_WAIT_SECONDS = float(os.environ.get("SERVER_WAIT_SECONDS", "420"))


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)  # the harness never sees a half-written file


def server_running() -> bool:
    """Is server.py (or the start.sh loop that restarts it) alive?

    The socket only appears once the model has loaded, so "no socket" can
    mean "still loading". This tells the two apart.
    """
    me = str(os.getpid())
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or p.name == me:
            continue
        try:
            argv = (p / "cmdline").read_bytes().split(b"\0")[:3]
        except OSError:
            continue
        # "python3 /app/server.py" or "/bin/sh /app/start.sh"; matching whole
        # arguments so a shell that merely mentions the name doesn't count
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


def wait_for_server(seconds: float) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if os.path.exists(wire.SOCKET_PATH):
            try:
                if wire.ask({"op": "ping"}, timeout=5).get("ok"):
                    return True
            except OSError:
                pass
        time.sleep(1)
    return False


def run_index(corpus: Path) -> int:
    if not server_running():  # a leftover socket file doesn't mean anyone's listening
        start_server()
    if wait_for_server(SERVER_WAIT_SECONDS):
        try:
            resp = wire.ask({"op": "index", "corpus": str(corpus)}, timeout=560)
            print(json.dumps(resp))
            return 0
        except Exception as e:  # noqa: BLE001
            print(f"index via server failed: {e}", file=sys.stderr)

    # No server (or it died). Parse what we can on the CPU and save it, so a
    # server that comes back later at least has the text. Images get skipped.
    print("server unavailable, indexing text only", file=sys.stderr)
    try:
        from corpus import load_corpus
        docs, skipped = load_corpus(corpus)
        out = [{"rel": d.rel, "kind": d.kind, "text": d.text, "flags": d.flags,
                "image_files": []} for d in docs]
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        write_atomic(INDEX_DIR / "index.json", {
            "docs": out, "skipped": [{"rel": s.rel, "reason": s.reason} for s in skipped]})
    except Exception as e:  # noqa: BLE001
        print(f"fallback index failed too: {e}", file=sys.stderr)
    return 0


def run_query(query_id: str, question: str) -> int:
    started = time.time()
    out = OUTPUT_DIR / f"{query_id}_output.json"
    # Written once, at the end. We don't put a blank file down first: if the
    # harness picks the file up as soon as it appears, it would grade the blank.
    # The socket timeout is what guarantees we get to the write in time.
    result = {"answer": "", "citations": [], "confidence": 0.0}
    deadline = started + QUERY_SECONDS
    try:
        # A server that's restarting has no socket for a moment. Worth waiting
        # for it while there's still time to answer.
        while not os.path.exists(wire.SOCKET_PATH) and time.time() < deadline - 5:
            time.sleep(0.5)
        resp = wire.ask(
            {"op": "query", "question": question, "deadline": deadline},
            timeout=max(1.0, deadline + 1 - time.time()),
        )
        answer = resp.get("answer") or ""
        citations = [c for c in (resp.get("citations") or []) if isinstance(c, str)]
        if isinstance(answer, str) and answer and citations:
            # half an answer scores zero; a clean refusal might not
            result = {"answer": answer, "citations": citations,
                      "confidence": float(resp.get("confidence") or 0.0)}
    except Exception as e:  # noqa: BLE001
        print(f"query {query_id}: answering blank ({e})", file=sys.stderr)
        if not server_running():
            start_server()  # too late for this question, not for the next ones
    finally:
        write_atomic(out, result)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, help="build the index over this corpus, then exit")
    ap.add_argument("--corpus", type=Path, help="corpus root for a query")
    ap.add_argument("--query-id", help="output stem the harness assigns, e.g. query_01")
    ap.add_argument("--query", help="the question to answer")
    args = ap.parse_args()

    if args.index is not None:
        return run_index(args.index)
    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")
    return run_query(args.query_id, args.query)


if __name__ == "__main__":
    raise SystemExit(main())
