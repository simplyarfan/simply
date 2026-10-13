"""The one model we ship: a vision-language model that reads images at index
time and answers questions at query time. Only server.py imports this."""

from __future__ import annotations

import os
import time

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# The VRAM rule is 1-48 GiB, sampled the whole run. On a 192 GB MI300X the
# allocator would happily grow past 48, so we cap ourselves below the line.
VRAM_CAP_GIB = 44

OCR_PROMPT = (
    "Transcribe every piece of text printed in this image, exactly as written, "
    "including codes, part numbers, revisions and small print. "
    "Keep each label on the same line as the value it belongs to, written as 'label: value'. "
    "For a table, pin map or connector diagram, write one row or one pin per line with "
    "its name next to it. "
    "Do not describe the picture and do not add anything that is not printed. "
    "If there is no text at all, reply NONE."
)


class NoGPU(RuntimeError):
    pass


def home_test_quantization() -> dict:
    # Only the home test image sets SIMPLY_QUANT: the 8B model is 17.5 GB in
    # bf16, too big for a 16 GB RTX card, so there we squeeze the text layers
    # to 8 bits. The vision part stays bf16 so image reading is unchanged.
    # The submitted image never sets it and runs the full bf16 model.
    mode = os.environ.get("SIMPLY_QUANT", "")
    if not mode:
        return {}
    from transformers import BitsAndBytesConfig
    skip = ["visual", "lm_head"]
    if mode == "4bit":
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 llm_int8_skip_modules=skip)
    else:
        cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
    print(f"[llm] home test: loading the text layers in {mode}", flush=True)
    return {"quantization_config": cfg}


class Model:
    def __init__(self, model_dir: str):
        if not torch.cuda.is_available():
            raise NoGPU("torch can't see a GPU; refusing to run an 8B model on the CPU")

        total = torch.cuda.get_device_properties(0).total_memory
        cap = VRAM_CAP_GIB * 2**30
        if total > cap:
            torch.cuda.set_per_process_memory_fraction(cap / total)
        self.vram_gib = min(total, cap) / 2**30

        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(model_dir)
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda",
                **home_test_quantization(),
            )
        except (ImportError, ValueError):
            # older transformers wants accelerate for device_map; load then move
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_dir, dtype=torch.bfloat16, attn_implementation="sdpa"
            ).to("cuda")
        self.model.eval()
        self.tokenizer = self.processor.tokenizer
        print(f"[llm] loaded {model_dir} in {time.time() - t0:.1f}s, "
              f"vram budget {self.vram_gib:.0f} GiB", flush=True)

    def chat(self, messages: list[dict], max_new_tokens: int, max_time: float) -> tuple[str, int]:
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        n_in = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                max_time=max(1.0, max_time),
            )
        text = self.processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
        return text.strip(), n_in

    def read_image(self, image, prompt: str = OCR_PROMPT, max_new_tokens: int = 1280,
                   max_time: float = 45.0) -> str:
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": fit_image(image)},
            {"type": "text", "text": prompt},
        ]}]
        text, _ = self.chat(msgs, max_new_tokens=max_new_tokens, max_time=max_time)
        return "" if text.strip().upper() == "NONE" else drop_loops(text)

    def free(self):
        torch.cuda.empty_cache()


def fit_image(im):
    # Big photos cost thousands of vision tokens for no gain, and tiny labels
    # are easier to read after a 2x upscale. Aim for the sweet spot.
    w, h = im.size
    longest = max(w, h)
    if longest > 1792:
        s = 1792 / longest
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))))
    elif longest < 640:
        im = im.resize((w * 2, h * 2))
    return im


def drop_loops(text: str) -> str:
    # Greedy decoding sometimes gets stuck writing the same line over and
    # over until it hits the token limit. Three in a row is plenty.
    out = []
    for line in text.split("\n"):
        if len(out) >= 3 and line.strip() and all(x == line for x in out[-3:]):
            continue
        out.append(line)
    return "\n".join(out)
