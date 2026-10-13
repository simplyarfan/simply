# simply

Team simply's code for the AMD AI League mini-challenges. Each match has its
own folder and its own container image.

```
match2/   Match 2, OCR: read the plate number or sign text in one image
match3/   Match 3, RAG: answer questions about a folder of mixed files, with citations
```

Both use the same model, Qwen3-VL-8B-Instruct, on the mandated
`rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0` base. In both,
`start.sh` (the container's CMD) starts `server.py`, which loads the model onto
the GPU once and keeps it there. `app.py` is what the harness runs: a thin
client that talks to the server over a unix socket, so no image or question
pays for loading the model.

## Match 3: RAG

```
python3 /app/app.py --index /app/corpus
python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."
```

Each query writes `/app/output/<query-id>_output.json`:

```json
{"answer": "94", "citations": ["specs/tq40_datasheet_r2.pdf"], "confidence": 0.9}
```

- **Indexing:** every file becomes text: PDF (per page), Word (body and
  tables), Excel (every sheet), CSV, logs, Python and images. The same model
  reads the text printed in pictures and scanned pages. Unreadable, encrypted
  and unknown files are skipped one at a time and never stop the walk.
- **Retrieval:** if the whole corpus fits in the context budget, all of it
  goes to the model. Otherwise BM25 picks the pieces, and identifiers found in
  them (ticket numbers, error codes, part numbers) pull in the files they
  point to.
- **Answering:** the model answers in JSON with a quote from each file it
  used. `answer.py` then checks every quote against its file, checks the
  answer is actually in a cited file, and keeps an earlier file only if it
  hands an identifier to the next step. Anything that fails these checks
  becomes a refusal: an empty answer and no citations.

```
match3/Dockerfile        base image, Python packages, model weights, code
match3/pin_torch.py      pins torch to the ROCm build the base image ships
match3/build_check.py    fails the build if torch is no longer the ROCm build
match3/fetch_model.py    downloads the model weights into the image at build time
match3/requirements.txt  transformers, pypdf, python-docx, openpyxl
match3/app/app.py        what the harness runs; never loads the model
match3/app/server.py     holds the model and the index
match3/app/corpus.py     file parsing
match3/app/retrieval.py  picks what the model reads
match3/app/answer.py     prompt, then checks quotes and citations in code
match3/app/llm.py        the model wrapper
match3/app/wire.py       the socket protocol
match3/app/start.sh      container entrypoint
```

Build: `docker build -t <name> match3`. The weights are fetched from Hugging
Face during the build, so nothing is downloaded at evaluation time. Startup
(model load plus `--index`) is kept under 600 s, and GPU memory is capped at
44 GiB to stay inside the 48 GiB limit.

## Match 2: OCR

```
python3 /app/app.py --input-image /app/input/image_01.png
```

writes `/app/output/image_01_output.json`:

```json
{"text": "7ABC123", "confidence": 0.9}
```

- The image is opened as plain RGB (PNG, JPEG, or TIFF including 16-bit) and
  sized for the model.
- One prompt spells out the transcription rules: the plate number only,
  without the state name or slogans; the province character kept on Chinese
  plates; every word of a sign, top to bottom; a bare number with no unit.
- A small clean-up step removes labels, quotes and line breaks, and drops a
  US state name if the model wrote one next to the plate number.

```
match2/Dockerfile      the Match 3 image plus this folder's app code
match2/app/app.py      what the harness runs; never loads the model
match2/app/server.py   holds the model, reads one image per request
match2/app/ocr.py      prompt, image loading and clean-up
match2/app/llm.py      the model wrapper (same as Match 3)
match2/app/wire.py     the socket protocol (same as Match 3)
match2/app/start.sh    container entrypoint
```

Build: `docker build --build-arg BASE_IMAGE=<Match 3 image> -t <name> match2`.
Match 2 needs the same base, packages and weights as Match 3, so it is built
on top of that image and only the code layer changes. That way the 16 GB of
weights are not downloaded and stored twice.
