"""Download part of a Hugging Face model repo while the image builds.

    python3 fetch_model.py <repo> <revision> <dest> <pattern> [<pattern> ...]

There is no network when the image is graded, so the weights have to be baked
in. Fetching them here instead of COPYing them from the build machine means
the 17 GB never has to sit on that machine's disk or go through the build
context. Each Dockerfile step fetches one shard so no single layer is huge.
"""

import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo, revision, dest, *patterns = sys.argv[1:]
snapshot_download(repo, revision=revision, local_dir=dest, allow_patterns=patterns)
# snapshot_download leaves lock files and metadata in .cache; not needed at runtime
shutil.rmtree(Path(dest) / ".cache", ignore_errors=True)
got = sorted(p.name for p in Path(dest).iterdir())
print(f"{dest}: {', '.join(got)}")
