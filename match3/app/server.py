#!/usr/bin/env python3
"""Long-running process that holds the model and the index.

start.sh launches this when the container starts. app.py (what the harness
actually calls) is a thin client that talks to us over a unix socket, so the
8B model loads once instead of once per question.

Requests are handled one at a time, in order. The harness asks one question
per process, so there's never a reason to do two things at once.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
import traceback
from collections import Counter
from pathlib import Path

import wire
from answer import (IMAGE_QUESTION_PROMPT, SYSTEM_PROMPT, build_messages,
                    clean_answer, content_terms, finalize, official_norm)
from corpus import load_corpus
from retrieval import id_tokens, make_pieces, select

BOOT = time.time()
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/qwen3-vl-8b")
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))
LOG_DIR = Path(os.environ.get("SIMPLY_LOG_DIR", "/app/logs"))

# Startup (container start + model load + --index) has to fit in 600s.
# Stop doing optional work (reading images) once we're this far in.
STARTUP_BUDGET_S = float(os.environ.get("STARTUP_BUDGET_S", "520"))

# How much corpus text goes in front of the model per question. Everything,
# if it fits. Measured at startup (see calibrate); CONTEXT_TOKENS forces it.
CONTEXT_TOKENS = int(os.environ.get("CONTEXT_TOKENS", "0"))
# reading the prompt should take at most this long, leaving the rest of the
# question's time for writing the answer
PREFILL_TARGET_S = float(os.environ.get("PREFILL_TARGET_S", "9"))


def log(msg):
    print(f"[server {time.time() - BOOT:6.1f}s] {msg}", flush=True)


class State:
    def __init__(self):
        self.model = None
        self.docs = []           # dicts: rel, kind, text, flags, image_files
        self.pieces = []
        self.texts = {}
        self.meta = {}
        self.doc_freq = {}
        self.budget = 16000

    def load_docs(self, docs):
        self.docs = docs
        tok = self.model.tokenizer if self.model else None
        self.pieces = make_pieces(docs, tok)
        self.texts = {d["rel"]: d["text"] for d in docs if d["text"].strip()}
        self.meta = {d["rel"]: (d["kind"], d["flags"]) for d in docs}
        df = Counter()
        for text in self.texts.values():
            df.update(id_tokens(text))
        self.doc_freq = dict(df)
        total = sum(p.tokens for p in self.pieces)
        log(f"index ready: {len(self.texts)} readable files, {len(self.pieces)} pieces, "
            f"~{total} tokens, budget {self.budget}")


S = State()


# ---- index --------------------------------------------------------------------

def do_index(corpus: str) -> dict:
    t0 = time.time()
    docs, skipped = load_corpus(Path(corpus))
    for s in skipped:
        log(f"skipped {s.rel}: {s.reason}")

    # Keep picture files on disk so a question can look at one again later.
    img_dir = INDEX_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    image_files = {n: [] for n in range(len(docs))}
    queue = []
    for n, d in enumerate(docs):
        for label, im, prio in d.pending_images:
            queue.append((prio, n, label, im))
            if prio == 0:
                path = img_dir / f"{n:04d}_{len(image_files[n])}.png"
                try:
                    im.save(path)
                    image_files[n].append(str(path))
                except Exception as e:  # noqa: BLE001
                    log(f"couldn't save image from {d.rel}: {e}")

    # Read the images: picture files first, then scanned pages, then figures,
    # so if the clock runs out it's the least likely answers we skip.
    queue.sort(key=lambda q: (q[0], q[1]))
    for prio, n, label, im in queue:
        d = docs[n]
        if S.model is None:
            break
        left = STARTUP_BUDGET_S - (time.time() - BOOT)
        if left < 10:
            log(f"out of startup time, not reading {d.rel} ({label})")
            continue
        try:
            ti = time.time()
            text = S.model.read_image(im, max_time=min(45.0, left - 5))
            log(f"read {d.rel} ({label}) in {time.time() - ti:.1f}s, {len(text)} chars")
        except Exception as e:  # noqa: BLE001
            log(f"vision model failed on {d.rel}: {e}")
            S.model.free()
            continue
        if text.strip():
            d.text += f"\n[{label}, text read from the image]\n{text.strip()}\n"

    out = [{"rel": d.rel, "kind": d.kind, "text": d.text.strip(),
            "flags": d.flags, "image_files": image_files[n]} for n, d in enumerate(docs)]

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"docs": out, "skipped": [{"rel": s.rel, "reason": s.reason} for s in skipped]}
    tmp = INDEX_DIR / "index.json.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, INDEX_DIR / "index.json")

    S.load_docs(out)
    return {"ok": True, "files": len(out), "skipped": len(skipped),
            "seconds": round(time.time() - t0, 1)}


def load_saved_index():
    p = INDEX_DIR / "index.json"
    if p.exists():
        try:
            S.load_docs(json.loads(p.read_text(encoding="utf-8"))["docs"])
        except Exception as e:  # noqa: BLE001
            log(f"saved index unreadable, ignoring: {e}")


# ---- query --------------------------------------------------------------------

def refusal(notes):
    return {"answer": "", "citations": [], "confidence": 0.0, "notes": notes}


def do_query(question: str, deadline: float) -> dict:
    t0 = time.time()
    if S.model is None:
        return refusal(["no model loaded"])
    if not S.pieces:
        return refusal(["empty index"])

    budget = S.budget
    raw, n_in = "", 0
    for attempt in range(2):
        pieces = select(question, S.pieces, budget)
        msgs = build_messages(question, pieces, S.meta)
        time_left = deadline - time.time() - 1.5
        if time_left < 3:
            return refusal(["ran out of time before generating"])
        try:
            raw, n_in = S.model.chat(msgs, max_new_tokens=320, max_time=time_left)
            break
        except Exception as e:  # noqa: BLE001 - usually out of memory on a huge context
            log(f"generate failed with {len(pieces)} pieces / budget {budget}: {e}")
            S.model.free()
            budget //= 2
    else:
        return refusal(["generation failed twice"])

    answer, cites, conf, notes = finalize(question, raw, S.texts, S.doc_freq, S.meta)

    # If the answer came out of a picture, look at the picture again with the
    # question in hand. Reading a pin map "for everything" and then searching
    # the transcript is less reliable than reading it for one thing.
    if answer and len(cites) == 1 and S.meta.get(cites[0], ("",))[0] == "image":
        answer = recheck_image(question, cites[0], answer, deadline, notes)

    # Nothing found, but the question is plainly about one of the pictures
    # ("printed on the asset label"): the transcript may have missed it, so
    # ask the picture directly.
    if not answer:
        answer, cites = ask_image(question, deadline, notes)
        if answer:
            conf = 0.5

    result = {"answer": answer, "citations": cites, "confidence": conf, "notes": notes}
    write_log({"question": question, "raw": raw, "input_tokens": n_in,
               "seconds": round(time.time() - t0, 2), **result})
    return result


def recheck_image(question, rel, answer, deadline, notes):
    files = next((d["image_files"] for d in S.docs if d["rel"] == rel), [])
    if not files or deadline - time.time() < 8:
        return answer
    try:
        direct = ask_picture(question, files[0], deadline)
    except Exception as e:  # noqa: BLE001
        notes.append(f"image recheck failed: {e}")
        return answer
    direct = clean_answer(direct, S.texts[rel], question)
    if direct and official_norm(direct) != official_norm(answer):
        notes.append(f"image recheck changed {answer!r} -> {direct!r}")
        return direct
    return answer


def image_docs_for(question):
    """Picture files the question names, best match first.

    A picture counts when its file name shares a word with the question, or
    its transcript shares two. Words with digits (product names, part
    numbers) don't count: they're in half the corpus.
    """
    q = {w for w in content_terms(question) if not any(c.isdigit() for c in w)}
    found = []
    for d in S.docs:
        if d["kind"] != "image" or not d["image_files"]:
            continue
        by_name = q & content_terms(d["rel"].replace("/", " "))
        by_text = q & content_terms(d["text"])
        if by_name or len(by_text) >= 2:
            found.append((2 * len(by_name) + len(by_text), d))
    found.sort(key=lambda f: -f[0])
    return [d for _, d in found]


REFUSAL_WORDS = re.compile(r"\b(not|no|none|cannot|can't|unable|unknown|n/a|unclear)\b", re.I)


def ask_picture(question, path, deadline):
    from PIL import Image
    with Image.open(path) as im:
        reply = S.model.read_image(
            im.convert("RGB"), IMAGE_QUESTION_PROMPT.format(question=question),
            max_new_tokens=32, max_time=deadline - time.time() - 2,
        )
    reply = reply.strip().splitlines()[0].strip() if reply.strip() else ""
    # a value is a few tokens; anything chattier is the model explaining itself
    if not reply or len(reply.split()) > 4 or REFUSAL_WORDS.search(reply):
        return ""
    return reply


def ask_image(question, deadline, notes):
    for d in image_docs_for(question)[:1]:
        if deadline - time.time() < 8:
            break
        try:
            direct = ask_picture(question, d["image_files"][0], deadline)
        except Exception as e:  # noqa: BLE001
            notes.append(f"asking {d['rel']} directly failed: {e}")
            break
        direct = clean_answer(direct, d["text"], question)
        if direct:
            notes.append(f"answered by looking at {d['rel']} directly")
            return direct, [d["rel"]]
    return "", []


def write_log(entry):
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "queries.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


# ---- main loop -------------------------------------------------------------------

def handle(req: dict) -> dict:
    op = req.get("op")
    if op == "ping":
        return {"ok": True, "model": S.model is not None, "uptime": time.time() - BOOT}
    if op == "index":
        return do_index(req["corpus"])
    if op == "query":
        return do_query(req["question"], float(req["deadline"]))
    return {"ok": False, "error": f"unknown op {op!r}"}


def pick_budget(vram_gib: float) -> int:
    # weights take ~18 GB; the rest is KV cache and prefill scratch.
    # 24 GB card -> 12k, 44 GB (our cap on MI300X) -> 32k.
    return 32000 if vram_gib >= 40 else 12000


def calibrate():
    """Warm the GPU up and measure how much text we can read per question.

    The first generate() on a fresh process is slow (kernels get compiled and
    tuned), and we'd rather pay that now than on question one. Then we time a
    full-size prompt: if it's too slow or doesn't fit in memory, shrink the
    budget until it does. Better to retrieve a bit harder than to time out.
    """
    m = S.model
    hello = [{"role": "user", "content": [{"type": "text", "text": "Reply with the word ready."}]}]
    t0 = time.time()
    m.chat(hello, max_new_tokens=4, max_time=120)
    log(f"warm-up took {time.time() - t0:.1f}s")

    budget = CONTEXT_TOKENS or pick_budget(m.vram_gib)
    line = ("2026-09-02T03:11:04Z node-07 meridian[2211]: INFO scheduler: batch 88213 "
            "accepted, 64 requests, queue depth 3, die 0 at 71C\n")
    per_line = max(1, len(m.tokenizer(line, add_special_tokens=False)["input_ids"]))
    started = time.time()
    for _ in range(5):
        # One read can't be interrupted, but we can stop trying new sizes.
        # Startup has a hard limit, and a slightly small budget beats missing it.
        if time.time() - started > 120:
            log(f"calibration is taking too long, settling for {budget} tokens")
            break
        log(f"trying {budget} tokens")
        filler = line * (budget // per_line)
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "text", "text": filler + "\nQUESTION: ?"}]}]
        t0 = time.time()
        try:
            m.chat(msgs, max_new_tokens=1, max_time=120)
        except Exception as e:  # noqa: BLE001 - out of memory, most likely
            log(f"{budget} tokens didn't fit ({type(e).__name__}), trying less")
            m.free()
            budget = int(budget * 0.6)
            continue
        dt = time.time() - t0
        log(f"read {budget} tokens in {dt:.1f}s")
        if dt <= PREFILL_TARGET_S or CONTEXT_TOKENS:
            break
        budget = int(budget * PREFILL_TARGET_S / dt * 0.9)
    m.free()
    S.budget = max(3000, budget)

    count = [{"role": "user", "content": [{"type": "text", "text":
             "Count from 1 to 100, numbers separated by spaces."}]}]
    t0 = time.time()
    text, _ = m.chat(count, max_new_tokens=64, max_time=60)
    n = len(m.tokenizer(text, add_special_tokens=False)["input_ids"])
    log(f"context budget {S.budget} tokens; writing speed ~{n / (time.time() - t0):.0f} tokens/s")


def main():
    log(f"starting, model dir {MODEL_DIR}")
    try:
        from llm import Model
        S.model = Model(MODEL_DIR)
        S.budget = pick_budget(S.model.vram_gib)
    except Exception:  # noqa: BLE001
        # Keep serving anyway: app.py still gets valid (empty) answers instead
        # of hanging, and the log says exactly what went wrong.
        log("MODEL FAILED TO LOAD:\n" + traceback.format_exc())
    if S.model is not None:
        try:
            calibrate()
        except Exception:  # noqa: BLE001
            log("calibration failed, keeping the default budget:\n" + traceback.format_exc())

    load_saved_index()

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
                resp = refusal(["server error"])
            try:
                wire.send(conn, resp)
            except OSError:
                log("client went away before we answered")  # it already wrote its fallback


if __name__ == "__main__":
    main()
