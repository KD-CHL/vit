import argparse
import csv
import hashlib
import os
import random
import shutil
import tarfile
import time
from collections import Counter
from pathlib import PurePosixPath
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from .common import DATA, NUM_CLASSES, OUTPUTS, SPLITS, WEIGHTS, save_json, sha256


ARCHIVES = {
    "images.tar.gz": "5c4f3ee8e5d25df40f4fd59a7f44e54c",
    "annotations.tar.gz": "95a8c909bbe2e81eed6a22bccdf3f68f",
}
DATA_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/"
MODEL_URL = (
    "https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k/"
    "resolve/2ec9fb3d7bb664aac471ac44582c94d18de33780/model.safetensors"
)
MODEL_HASH = "c401d219603ac3e20b6373c7b198c78d3a733f80b755d148bda3bc320ae69800"
SPLIT_SEED = 42


def md5(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, path, expected, digest):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and digest(path) == expected:
        return
    temporary = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            print(f"Downloading {path.name} ({attempt + 1}/3)", flush=True)
            with urlopen(url, timeout=60) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            if digest(temporary) != expected:
                raise OSError(f"Checksum mismatch: {path.name}")
            os.replace(temporary, path)
            return
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def extract_dataset():
    for name, checksum in ARCHIVES.items():
        archive = DATA / name
        download(DATA_URL + name, archive, checksum, md5)
        with tarfile.open(archive, "r:gz") as stream:
            for member in stream:
                path = PurePosixPath(member.name)
                if not member.isfile() or len(path.parts) != 2:
                    continue
                image = path.parent.name == "images" and path.suffix.lower() == ".jpg"
                annotation = str(path) in {"annotations/trainval.txt", "annotations/test.txt"}
                if image or annotation:
                    target = DATA / path.parent.name / path.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.is_file() and target.stat().st_size == member.size:
                        continue
                    with stream.extractfile(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)


def annotated_rows(source):
    rows = []
    annotation = DATA / "annotations" / f"{source}.txt"
    for line in annotation.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        stem, label, species, _ = line.split()
        rows.append({
            "path": f"images/{stem}.jpg",
            "label": int(label) - 1,
            "breed": stem.rsplit("_", 1)[0].replace("_", " ").title(),
            "species": {"1": "cat", "2": "dog"}[species],
        })
    return rows


def inspect_images(rows):
    clean, details, invalid = [], [], []
    for row in rows:
        try:
            with Image.open(DATA / row["path"]) as image:
                image.load()
                details.append((*image.size, image.mode))
        except (OSError, UnidentifiedImageError) as exc:
            invalid.append({"path": row["path"], "reason": str(exc)})
            continue
        clean.append(row)
    return clean, details, invalid


def create_splits(trainval, test):
    train, val = train_test_split(
        trainval, test_size=0.10, random_state=SPLIT_SEED,
        stratify=[row["label"] for row in trainval],
    )
    splits = {name: sorted(rows, key=lambda row: (row["label"], row["path"]))
              for name, rows in (("train", train), ("val", val), ("test", test))}
    paths = [row["path"] for rows in splits.values() for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate image paths in the split manifests.")
    return splits


def write_splits(splits):
    SPLITS.mkdir(exist_ok=True)
    classes = {}
    for split, rows in splits.items():
        path = SPLITS / f"{split}.csv"
        temporary = path.with_suffix(".csv.part")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, ["path", "label", "breed", "species"],
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        if path.exists() and path.read_bytes() != temporary.read_bytes():
            temporary.unlink()
            raise ValueError(f"The computed {split} split differs from the versioned manifest.")
        os.replace(temporary, path)
        for row in rows:
            label = row["label"]
            info = {"id": label, "breed": row["breed"], "species": row["species"]}
            if classes.setdefault(label, info) != info:
                raise ValueError(f"Inconsistent breed or species for label {label}")
    if sorted(classes) != list(range(NUM_CLASSES)):
        raise ValueError("The annotated dataset must contain 37 breeds.")
    ordered = [classes[label] for label in sorted(classes)]
    save_json(SPLITS / "classes.json", ordered)
    return ordered


def draw_dataset(splits, breeds, details):
    root = OUTPUTS / "data"
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SPLIT_SEED)
    fig, axes = plt.subplots(10, 16, figsize=(19, 13), layout="constrained")
    for ax in axes.ravel():
        ax.set_axis_off()
    for label, breed in enumerate(breeds):
        row_index, group = divmod(label, 4)
        candidates = [row for row in splits["train"] if row["label"] == label]
        for index, row in enumerate(rng.sample(candidates, 4)):
            ax = axes[row_index, 4 * group + index]
            with Image.open(DATA / row["path"]) as image:
                rgb = image.convert("RGB")
                rgb.thumbnail((145, 120))
            ax.imshow(rgb)
            if index == 0:
                ax.set_title(breed["breed"], fontsize=7.5, loc="left")
    fig.savefig(root / "train_examples.jpg", dpi=145)
    plt.close(fig)

    width, height = np.array([item[:2] for item in details]).T
    ratios = width / height
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
    for ax, values, title in zip(axes, (width, height, ratios),
                                  ("Width (px)", "Height (px)", "Width / height")):
        ax.hist(values, bins=35, color="#34699a")
        ax.set(xlabel=title, ylabel="Images")
    fig.savefig(root / "image_geometry.png", dpi=140)
    plt.close(fig)
    quantiles = [0, .05, .5, .95, 1]
    return {
        "quantiles": quantiles,
        "width": np.quantile(width, quantiles).tolist(),
        "height": np.quantile(height, quantiles).tolist(),
        "aspect_ratio": np.quantile(ratios, quantiles).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    extract_dataset()
    trainval, train_details, train_invalid = inspect_images(annotated_rows("trainval"))
    test, test_details, test_invalid = inspect_images(annotated_rows("test"))
    splits = create_splits(trainval, test)
    breeds = write_splits(splits)
    details = train_details + test_details
    geometry = draw_dataset(splits, breeds, details)
    counts = {split: Counter(row["label"] for row in rows) for split, rows in splits.items()}
    save_json(OUTPUTS / "data" / "summary.json", {
        "split_seed": SPLIT_SEED,
        "counts": {split: len(rows) for split, rows in splits.items()},
        "breed_counts": [
            {**breed, **{split: counter[breed["id"]] for split, counter in counts.items()}}
            for breed in breeds
        ],
        "geometry": geometry,
        "color_modes": Counter(item[2] for item in details),
        "invalid_images": train_invalid + test_invalid,
        "source_archive_md5": ARCHIVES,
        "split_sha256": {split: sha256(SPLITS / f"{split}.csv") for split in splits},
        "pretrained_sha256": MODEL_HASH,
    })
    print({split: len(rows) for split, rows in splits.items()}, flush=True)
    if not args.data_only:
        download(MODEL_URL, WEIGHTS, MODEL_HASH, sha256)


if __name__ == "__main__":
    main()
