import hashlib
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "oxford-iiit-pet"
WEIGHTS = ROOT / "data" / "weights" / "vit-b16.safetensors"
SPLITS = ROOT / "splits"
RESULT_ROOT = (ROOT / os.environ.get("VIT_RESULT_ROOT", "")).resolve()
RUNS = RESULT_ROOT / "runs"
OUTPUTS = RESULT_ROOT / "outputs"
SEEDS = (42, 123, 2026)
NUM_CLASSES = 37


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(
            hashlib, "file_digest"
        ) else _sha256_stream(stream)


def _sha256_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_checkpoint(path, obj):
    import torch

    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_and_amp():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the experiment suite.")
    torch.set_float32_matmul_precision("high")
    return torch.device("cuda"), (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
