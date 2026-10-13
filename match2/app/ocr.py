"""Match 2: read the plate number or the sign text off one image.

Same model as the RAG side, just a different question. The graders compare
after uppercasing and dropping whitespace and - . · _ so what matters is the
right characters in the right order, and nothing extra around them.
"""

from __future__ import annotations

import re

import numpy as np
from PIL import Image, ImageOps

PLATE_SIGN_PROMPT = """This image shows a vehicle licence plate or a road sign. Write down its text, following these rules.

Licence plate: write only the registration number, exactly as printed. Leave out everything printed around it: the state or country name, slogans, websites, dealer names and registration stickers. On a Chinese plate the province character and the letter after it are part of the number, so keep them (for example 京A·12345).

Road sign: write every word and number printed on the sign, reading top to bottom, joined with single spaces (for example SPEED LIMIT 65 or ROAD WORK AHEAD). If the sign shows only a number, write only that number and add no unit.

Copy each character exactly as printed. Do not fix spelling, guess hidden characters or explain anything.
Reply with the text on a single line and nothing else."""

# Printed around the number on US plates, never part of it. The prompt already
# says to leave these out; this catches the times the model doesn't listen.
US_STATES = [
    "ALABAMA", "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO", "CONNECTICUT",
    "DELAWARE", "FLORIDA", "GEORGIA", "HAWAII", "IDAHO", "ILLINOIS", "INDIANA", "IOWA",
    "KANSAS", "KENTUCKY", "LOUISIANA", "MAINE", "MARYLAND", "MASSACHUSETTS", "MICHIGAN",
    "MINNESOTA", "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA", "NEVADA",
    "NEW HAMPSHIRE", "NEW JERSEY", "NEW MEXICO", "NEW YORK", "NORTH CAROLINA",
    "NORTH DAKOTA", "OHIO", "OKLAHOMA", "OREGON", "PENNSYLVANIA", "RHODE ISLAND",
    "SOUTH CAROLINA", "SOUTH DAKOTA", "TENNESSEE", "TEXAS", "UTAH", "VERMONT", "VIRGINIA",
    "WASHINGTON", "WEST VIRGINIA", "WISCONSIN", "WYOMING",
]
# longest first, so WEST VIRGINIA goes before VIRGINIA gets a chance
STATE_RE = re.compile(
    r"\b(" + "|".join(sorted((s.replace(" ", r"\s+") for s in US_STATES), key=len, reverse=True)) + r")\b",
    re.I,
)
LABEL_RE = re.compile(r"^(text|answer|plate(\s+number)?|sign(\s+text)?|reading)\s*[:：]\s*", re.I)


def load_image(path: str) -> Image.Image:
    """Open a PNG, JPEG or TIFF as plain 8-bit RGB."""
    with Image.open(path) as im:
        im.seek(0)  # multi-page TIFF: the first page is the picture
        im = ImageOps.exif_transpose(im)  # phone photos come in sideways otherwise
        if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            # 16-bit or float TIFF. convert("RGB") would clip most of it to
            # white, so stretch it into 8 bits first.
            a = np.asarray(im, dtype=np.float32)
            lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
            a = np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
            im = Image.fromarray(a)
        return im.convert("RGB")


def clean_reading(reply: str) -> str:
    text = reply.strip()
    if not text or text.upper() in ("NONE", "NO TEXT"):
        return ""
    text = LABEL_RE.sub("", text)
    text = text.strip("`\"'“”‘’ ")
    # several lines become one, top to bottom
    text = " ".join(line.strip() for line in text.splitlines() if line.strip())
    # a state name next to something with digits in it is a plate banner
    rest = STATE_RE.sub(" ", text)
    if rest != text and re.search(r"\d", rest):
        text = rest
    return re.sub(r"\s+", " ", text).strip()
