"""Prompt, parse, and then check the model's work before we trust it.

The model is asked for an answer plus an evidence chain: which file each step
came from and a line quoted from it. Then plain code decides what we actually
submit:

  * a quote that isn't really in the file it names doesn't count
  * the answer has to be visible in one of the files that survive
  * any other file stays cited only if it hands an identifier (ticket, error
    code, part number, version) to the step after it

That last rule is the "would removing it make the answer impossible" test
from the brief, done in code instead of hoping the model gets it right.
"""

from __future__ import annotations

import json
import re

from retrieval import STOP, id_tokens, words

SYSTEM_PROMPT = """You answer questions about a set of company files. Use ONLY those files. They describe fictional products, so anything you remember from outside is wrong here.

How to answer:
1. Give the value only: a number, part number, version, code, quarter, pin, name. No sentence and no units. Copy identifiers exactly as the file writes them.
2. Keep any qualifier that is part of the value: a quarter keeps its fiscal year ("Q2 FY26", not "Q2"), a revision keeps its prefix ("REV-B1", not "B1").
3. If a document is marked withdrawn, superseded or obsolete, ignore it whenever a current version covers the same thing. If two revisions of a document disagree, the newer revision wins.
4. In source code, the value actually assigned wins over a value mentioned in a comment.
5. Some questions need two or more files: one gives an identifier (ticket, error code, part number, version) and another gives the value for that identifier. Follow the chain until you reach the value.
6. If the files do not contain the answer, the answer is an empty string. Do not guess. A value for something similar (a different product, part, volume, date or condition) is not the answer. A file you were not shown does not count.

Evidence: list the steps you used, in order, ending with the step that holds the answer. Each step is a file path exactly as written after "FILE:" and one short line copied word for word from that file. For a chain, include the line that gave you the identifier, then the line that gives the value for it. Only list files you needed. For each one ask: could I still answer without this file? If yes, leave it out.

Reply with JSON only, no other text:
{"answer": "...", "evidence": [{"file": "...", "quote": "..."}]}"""


def build_messages(question: str, pieces, doc_meta) -> list[dict]:
    blocks = []
    for p in pieces:
        blocks.append(p.header(doc_meta[p.rel]) + "\n" + p.text)
    files = "\n\n".join(blocks) if blocks else "(no readable files)"
    user = f"FILES\n\n{files}\n\n===== END OF FILES =====\n\nQUESTION: {question}"
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]


IMAGE_QUESTION_PROMPT = """Look at the image and answer the question using only what is printed in it.
Give the value only, no sentence. Keep any prefix or qualifier printed as part of the value.
If the image does not show the answer, reply with nothing at all.

Question: {question}"""


# ---- parsing ---------------------------------------------------------------

def parse_model_json(raw: str) -> dict:
    """Pull the JSON object out of whatever the model wrote."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(raw[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
    # last resort: at least get the answer string out
    m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    return {"answer": m.group(1) if m else "", "evidence": []}


# ---- comparisons -------------------------------------------------------------

def official_norm(s: str) -> str:
    # same normalisation the graders use: uppercase, drop whitespace and - . · _
    return re.sub(r"[\s\-.·_]", "", s.upper())


def loose(s: str) -> str:
    s = s.lower()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[‐-―]", "-", s)
    return " ".join(s.split())


def quote_in_text(quote: str, text: str) -> bool:
    q, t = loose(quote), loose(text)
    if not q:
        return False
    if q in t:
        return True
    # models tidy up whitespace and punctuation when they copy, so also accept
    # a quote whose words are nearly all in the file
    qw = [w for w in words(quote) if len(w) > 1]
    if len(qw) < 2:
        return False
    tw = set(words(text))
    return sum(w in tw for w in qw) / len(qw) >= 0.8


def answer_in_text(answer: str, text: str) -> bool:
    a = official_norm(answer)
    if not a:
        return False
    if a in official_norm(text):
        return True
    # "Q3 FY27" vs a table that says "Q3 | FY27" or "Q3 of FY27"
    parts = [p for p in re.split(r"[^a-z0-9]+", answer.lower()) if p]
    tw = set(re.split(r"[^a-z0-9]+", text.lower()))
    return bool(parts) and all(p in tw for p in parts)


# words that show up in every question and say nothing about which line answers it
GENERIC = set("""
value number name version code file files document shown show give given
listed list say says printed current which what
""".split())


def content_terms(text: str) -> set[str]:
    return {w for w in words(text) if len(w) > 2 and w not in STOP and w not in GENERIC}


def quote_window(quote: str, text: str, answer: str = "") -> str:
    """The line(s) of the file the quote came from.

    Only the quoted line itself, not its neighbours: in a table, a log or a
    list of releases the next line is a different record, and borrowing its
    identifiers is exactly how a decoy file sneaks into the citations. A
    quote that runs over a line break gets the lines it covers. If the
    quoted line doesn't hold the answer, the line that does is added.
    """
    lines = text.split("\n")
    qw = {w for w in words(quote) if len(w) > 1}
    lq = loose(quote)

    def best_line(candidates):
        best, best_i = 0, None
        for i in candidates:
            s = len(qw & set(words(lines[i])))
            if s > best:
                best, best_i = s, i
        return best_i

    def covered(i):
        return 0 <= i < len(lines) and loose(lines[i]) != "" and loose(lines[i]) in lq

    picked = []
    i = best_line(range(len(lines)))
    if i is not None:
        lo, hi = i, i
        while covered(lo - 1):
            lo -= 1
        while covered(hi + 1):
            hi += 1
        picked = list(range(lo, hi + 1))
    if answer and not answer_in_text(answer, "\n".join(lines[j] for j in picked)):
        holders = [j for j, ln in enumerate(lines) if answer_in_text(answer, ln)]
        if holders:
            j = best_line(holders)
            picked.append(holders[0] if j is None else j)
    return "\n".join(lines[j] for j in sorted(set(picked)))


# ---- answer clean-up -------------------------------------------------------

UNIT_NUMBER = re.compile(
    r"^[$€£]?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*(?:°\s*[CFK]?|%|[A-Za-zµ][A-Za-z/µ·\-]{0,11}(?:\s?[A-Za-z/]{1,8})?)?$"
)


# "What board revision ...", "Which revision ...", "What is the revision ..."
# but not "Which pin on revision B ..."
ASKS_FOR_REVISION = re.compile(r"\b(what|which)\s+(is\s+the\s+|\w+\s+)?rev(ision)?\b", re.I)


def clean_answer(answer: str, support: str, question: str = "") -> str:
    a = str(answer).strip().strip("`\"' ").rstrip(".").strip()
    if not a:
        return ""

    # "180 seconds", "94 °C", "$6,412" -> the number. The graders strip units
    # for us on their side, but commas aren't part of their normalisation.
    m = UNIT_NUMBER.match(a)
    if m and not re.fullmatch(r"\d+(\.\d+){2,}", a):
        a = m.group(1).replace(",", "")

    # Q3 -> Q3 FY27 when the evidence says so
    if re.fullmatch(r"Q[1-4]", a, re.I):
        m = re.search(rf"\b{a}\s*(?:of\s+)?(FY\s?\d{{2,4}}|\d{{4}})\b", support, re.I)
        if m:
            a = f"{a.upper()} {m.group(1)}"

    # C2 -> REV-C2 when the label prints it that way
    if re.fullmatch(r"[A-Z]?\d+[A-Z]?\d*|[A-Z]\d*", a):
        m = re.search(rf"\b(REV(?:ISION)?[-\s.:]*){re.escape(a)}\b", support, re.I)
        if m:
            a = "REV-" + a
        elif ASKS_FOR_REVISION.search(question) and re.fullmatch(r"[A-Z]{1,2}\d{1,2}", a):
            # The brief: "a revision includes its prefix (REV-C2, not C2)".
            # Graders strip the dash, so REV-C2, REV C2 and Rev.C2 all match.
            a = "REV-" + a
    return a


# ---- the main check ----------------------------------------------------------

def resolve_file(name: str, known: dict[str, str]) -> str | None:
    n = str(name).strip().replace("\\", "/")
    for prefix in ("/app/corpus/", "app/corpus/", "./", "/"):
        if n.startswith(prefix):
            n = n[len(prefix):]
    if n in known:
        return n
    base = n.rsplit("/", 1)[-1]
    hits = [k for k in known if k.rsplit("/", 1)[-1] == base]
    return hits[0] if len(hits) == 1 else None


def finalize(question: str, raw: str, texts: dict[str, str], doc_freq: dict[str, int],
             meta: dict | None = None):
    """Return (answer, citations, confidence, notes)."""
    meta = meta or {}
    notes = []
    obj = parse_model_json(raw)
    answer = str(obj.get("answer") or "").strip()
    evidence = obj.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = []

    if not answer:
        return "", [], 0.0, ["model refused"]

    # keep only steps whose quote really is in the file they name
    steps = []
    for e in evidence:
        if not isinstance(e, dict):
            continue
        rel = resolve_file(e.get("file", ""), texts)
        quote = str(e.get("quote") or "")
        if rel is None:
            notes.append(f"dropped unknown file {e.get('file')!r}")
            continue
        if not quote_in_text(quote, texts[rel]):
            notes.append(f"dropped {rel}: quote not found")
            continue
        steps.append((rel, quote))

    if not steps:
        return "", [], 0.0, notes + ["no verifiable evidence -> refuse"]

    # the answer has to be readable in one of the files we're about to cite.
    # Take the latest step that has it: that's where the chain ended.
    answer_idx = None
    for i in range(len(steps) - 1, -1, -1):
        if answer_in_text(answer, texts[steps[i][0]]):
            answer_idx = i
            break
    if answer_idx is None:
        return "", [], 0.0, notes + [f"answer {answer!r} not in any cited file -> refuse"]

    answer_file, answer_quote = steps[answer_idx]
    answer_win = quote_window(answer_quote, texts[answer_file], answer)
    answer = clean_answer(answer, answer_quote + "\n" + answer_win + "\n" + texts[answer_file], question)

    # Walk the chain back from the answer. An earlier file stays only if it
    # hands an identifier to a file we're already keeping, and that
    # identifier isn't just something the question itself told us.
    ignore = id_tokens(question) | set(words(question)) | id_tokens(answer)
    many = max(3, len(texts) // 4)

    def link_ids(text):
        return id_tokens(text) - ignore

    kept = {answer_file: answer_win}
    supplied = set()  # each identifier needs one source, not every file that mentions it
    pending = []
    # the steps before the answer first, nearest first; then any the model
    # listed after it (some models put the answer first, then how they got there)
    for rel, quote in [*reversed(steps[:answer_idx]), *steps[answer_idx + 1:]]:
        if rel != answer_file and rel not in (r for r, _ in pending):
            pending.append((rel, quote_window(quote, texts[rel])))

    def link(rel, win, strong):
        for k, kwin in kept.items():
            if strong:
                # this step's quoted line and the kept step's line name the
                # same thing. Allowed even for an id that's in a few files.
                ids = (link_ids(kwin) & link_ids(win)) - supplied
                ids = {t for t in ids if doc_freq.get(t, 0) <= many}
            else:
                # the model quoted the wrong line of the right file
                ids = (link_ids(kwin) & link_ids(texts[rel])) - supplied
                if k != answer_file:
                    # mid-chain the hand-off can run the other way too. Not
                    # against the answer file, where some other row's version
                    # number would let a decoy back in.
                    ids |= (link_ids(win) & link_ids(texts[k])) - supplied
                ids = {t for t in ids if doc_freq.get(t, 0) <= 3}
            if ids:
                return ids
        return set()

    # strong links first, so a file that actually quotes the identifier beats
    # one that merely mentions it somewhere
    while pending:
        for strong in (True, False):
            hit = next(((r, w, ids) for r, w in pending if (ids := link(r, w, strong))), None)
            if hit:
                break
        if not hit:
            break
        rel, win, ids = hit
        kept[rel] = win
        supplied |= ids
        pending = [(r, w) for r, w in pending if r != rel]
    for rel, _ in pending:
        notes.append(f"dropped {rel}: no identifier links it to the answer")

    if len(kept) == 1 and meta.get(answer_file, ("",))[0] != "image":
        hop = missing_hop(question, answer_file, answer_win, texts, doc_freq, ignore, meta)
        if hop:
            notes.append(f"added {hop}: it holds the identifier that leads to the answer")
            kept[hop] = ""

    # A number in the question is a condition ("at 10,000 units", "in
    # P-state 12"). If none of the files we're citing even mention it, the
    # model answered a neighbouring question (the 1,000-unit price, a part's
    # cost instead of the board's) and the honest answer is no answer.
    dropped = [rel for rel, _ in pending]
    for n in question_numbers(question):
        if any(number_in(n, texts[c]) for c in kept):
            continue
        # Unless the model did cite the file that has it and only quoted the
        # wrong line, so the chain check let it go ("node 12" in a log). The
        # question names that file's subject, so it was a hop after all.
        back = [r for r in dropped if number_in(n, texts[r])]
        if len(back) == 1:
            kept[back[0]] = ""
            notes.append(f"kept {back[0]} after all: it's the only cited file with {n}")
            continue
        return "", [], 0.0, notes + [f"question says {n} but no cited file does -> refuse"]

    # Same for a product or part tag ("KX-11"): if no cited file names it,
    # the model answered for a sister product. A picture is exempt, since
    # what we read off it may have missed the name; file names count.
    if not any(meta.get(c, ("",))[0] == "image" for c in kept):
        for tag in question_tags(question):
            if not any(tag_in(tag, c) or tag_in(tag, texts[c]) for c in kept):
                return "", [], 0.0, notes + [f"question is about {'-'.join(tag)} but no cited file is -> refuse"]

    citations = sorted(kept)
    confidence = 0.9 if len(steps) == len(evidence) else 0.6
    return answer, citations, confidence, notes


TAG_RE = re.compile(r"(?<![\w-])([A-Za-z]{2,5})-(\d{1,5})(?![\w-])")


def question_tags(question: str) -> list[tuple[str, str]]:
    # "KX-11", "TQ-40": letters, a hyphen, a number
    return [(m.group(1), m.group(2)) for m in TAG_RE.finditer(question)]


def tag_in(tag: tuple[str, str], text: str) -> bool:
    # written "KX-11", "KX 11", "KX_11" or "kx11", but not "KX-110"
    letters, digits = tag
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(letters)}[\s_-]?{digits}(?!\d)", text, re.I) is not None


NUMBER_RE = re.compile(
    r"(?<![\w\-.])(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)*)(?![\w\-]|[.,]\d)"
)


def question_numbers(question: str) -> list[str]:
    # standalone numbers only: the 40 in "TQ-40" or the 4 in "J4" is part of a name
    return [m.group(0).replace(",", "") for m in NUMBER_RE.finditer(question)]


def number_in(n: str, text: str) -> bool:
    flat = re.sub(r"(?<=\d)[, ](?=\d{3}\b)", "", text)  # 10,000 and 10 000 -> 10000
    forms = [n]
    if n.isdigit() and int(n) >= 1000 and int(n) % 1000 == 0:
        forms.append(f"{int(n) // 1000}k")  # 10000 -> 10k
    return any(re.search(rf"(?<![\d.]){re.escape(f)}(?![\d])", flat, re.I) for f in forms)


def missing_hop(question, answer_file, answer_win, texts, doc_freq, ignore, meta):
    """Catch a two-file chain where the model only listed the second file.

    The tell: the line holding the answer shares little with the question,
    so nobody could have found it from the question alone. If that line
    carries an identifier that one or two other files also have, and one of
    them is clearly the file the question is talking about, that file was
    the first hop.
    """
    q_terms = content_terms(question)
    own = len(q_terms & content_terms(answer_win))
    if own > 2:
        return None
    scores = {}
    for x in id_tokens(answer_win) - ignore:
        if not 2 <= doc_freq.get(x, 0) <= 3:
            continue
        for rel, t in texts.items():
            if rel == answer_file or x not in id_tokens(t):
                continue
            if meta.get(rel, ("", []))[1]:  # withdrawn/superseded files don't start chains
                continue
            scores[rel] = len(q_terms & (content_terms(t) | content_terms(rel.replace("/", " "))))
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not ranked or ranked[0][1] < own + 2:
        return None
    if len(ranked) > 1 and ranked[1][1] >= ranked[0][1]:
        return None  # two equally good candidates: don't guess
    return ranked[0][0]
