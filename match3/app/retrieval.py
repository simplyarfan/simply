"""Decide which parts of the corpus go in front of the model.

The default is: all of it. The sample corpus is a few thousand tokens and an
8B model reads 30k+ comfortably, so for a corpus that fits there is nothing to
retrieve and nothing to miss. Only when the corpus is bigger than the budget
do we rank pieces and drop the least useful ones.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# rough chars-per-token for this kind of text; only used when we have no tokenizer
CHARS_PER_TOKEN = 3.2
PIECE_TOKENS = 1500          # files bigger than this get split into windows

WORD_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", re.I)
STOP = set("""
a an and are as at be by for from has have how in is it its of on or that the
this to was were what when where which who why with does do did per into than
""".split())


def words(text: str) -> list[str]:
    out = []
    for w in WORD_RE.findall(text.lower()):
        out.append(w)
        # 'orr-1847' should also match plain 'orr' / '1847'
        if any(ch in w for ch in "-_."):
            out.extend(p for p in re.split(r"[-_.]", w) if p)
    return [w for w in out if w not in STOP]


def id_tokens(text: str) -> set[str]:
    """Things that look like identifiers: ORR-1847, E7731, TQ-40, 4.3.2, B14.

    These are what links one document to another, so they drive both the
    multi-hop expansion here and the citation check in answer.py.
    """
    found = set()
    for w in WORD_RE.findall(text):
        lw = w.lower()
        has_digit = any(c.isdigit() for c in lw)
        has_alpha = any(c.isalpha() for c in lw)
        if has_digit and has_alpha and len(lw) >= 3:
            found.add(lw)
        elif re.fullmatch(r"\d+(\.\d+){2,}", lw):  # versions like 4.3.2
            found.add(lw)
    return found


class Piece:
    def __init__(self, rel, part, text, tokens):
        self.rel, self.part, self.text, self.tokens = rel, part, text, tokens

    def header(self, doc_meta) -> str:
        kind, flags = doc_meta
        bits = [kind]
        if self.part:
            bits.append(self.part)
        bits.extend(flags)
        return f"===== FILE: {self.rel}  ({', '.join(bits)}) ====="


def count_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return int(len(text) / CHARS_PER_TOKEN) + 1


def make_pieces(docs, tokenizer=None) -> list[Piece]:
    pieces = []
    for d in docs:
        text = d["text"].strip()
        if not text:
            continue
        n = count_tokens(text, tokenizer)
        if n <= PIECE_TOKENS:
            pieces.append(Piece(d["rel"], "", text, n))
            continue
        # big file: cut on line boundaries into windows that overlap a little,
        # so a table row or log line is never sliced in half
        lines = text.split("\n")
        buf, buf_tok, idx = [], 0, 1
        for line in lines:
            lt = max(1, int(len(line) / CHARS_PER_TOKEN))
            if buf and buf_tok + lt > PIECE_TOKENS:
                chunk = "\n".join(buf)
                pieces.append(Piece(d["rel"], f"part {idx}", chunk, count_tokens(chunk, tokenizer)))
                idx += 1
                keep = buf[-5:]  # small overlap
                buf, buf_tok = list(keep), sum(max(1, int(len(x) / CHARS_PER_TOKEN)) for x in keep)
            buf.append(line)
            buf_tok += lt
        if buf:
            chunk = "\n".join(buf)
            pieces.append(Piece(d["rel"], f"part {idx}", chunk, count_tokens(chunk, tokenizer)))
    return pieces


def bm25_scores(question: str, pieces: list[Piece]) -> list[float]:
    q = Counter(words(question))
    # the path counts as text too: "asset label" should find support/asset_label.jpg
    docs = [Counter(words(p.rel.replace("/", " ") + "\n" + p.text)) for p in pieces]
    n = len(docs)
    avg = sum(sum(d.values()) for d in docs) / max(n, 1)
    df = Counter()
    for d in docs:
        df.update(d.keys())
    k1, b = 1.4, 0.75
    scores = []
    for d in docs:
        length = sum(d.values())
        s = 0.0
        for term in q:
            tf = d.get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg))
        scores.append(s)
    return scores


def select(question: str, pieces: list[Piece], budget_tokens: int) -> list[Piece]:
    total = sum(p.tokens for p in pieces)
    if total <= budget_tokens:
        return pieces

    scores = bm25_scores(question, pieces)
    order = sorted(range(len(pieces)), key=lambda i: -scores[i])
    chosen, used = set(), 0

    # first pass: best matches, but leave a third of the room for hop two
    for i in order:
        if used + pieces[i].tokens > budget_tokens * 0.66:
            continue
        chosen.add(i)
        used += pieces[i].tokens

    # second pass: follow identifiers. If a chosen piece mentions ORR-1847 and
    # the question doesn't, whatever else mentions ORR-1847 is probably hop two.
    q_ids = id_tokens(question)
    links = set()
    for i in chosen:
        links |= id_tokens(pieces[i].text) - q_ids
    if links:
        linked = [i for i in order if i not in chosen and id_tokens(pieces[i].text) & links]
        for i in linked:
            if used + pieces[i].tokens <= budget_tokens:
                chosen.add(i)
                used += pieces[i].tokens

    # then top up with whatever ranks next
    for i in order:
        if i not in chosen and used + pieces[i].tokens <= budget_tokens:
            chosen.add(i)
            used += pieces[i].tokens

    # keep corpus order in the prompt; it reads more naturally than score order
    return [pieces[i] for i in sorted(chosen)]
