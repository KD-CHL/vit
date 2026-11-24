import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .common import save_json
from .data import NormalizeBatch, classes, image_transform, load_rgb
from .models import load_model, vit_feature


def predict(checkpoint, path, device, merge=True):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    experiment = state["config"]["experiment"]
    model, config = load_model(state, device, merge=merge)
    transform = image_transform(config)
    norm = NormalizeBatch(config, device)
    breeds = classes()
    path = Path(path)
    Image.init()
    extensions = {suffix for suffix, fmt in Image.registered_extensions().items()
                  if fmt in Image.OPEN}
    if path.is_dir():
        paths = sorted(item for item in path.iterdir()
                       if item.is_file() and item.suffix.lower() in extensions)
    elif path.is_file():
        paths = [path]
    else:
        raise FileNotFoundError(path)
    if not paths:
        raise ValueError(f"No supported images: {path}")
    results = []
    with torch.inference_mode():
        for item in paths:
            images = norm(transform(load_rgb(item)).unsqueeze(0))
            if experiment == "v2":
                logits = model.head(vit_feature(model, images, state["config"]["pool"]))
            else:
                logits = model(images)
            scores, ids = logits.float().softmax(dim=1).topk(3)
            results.append({
                "image": str(item),
                "top3": [{"breed": breeds[index]["breed"], "confidence": score}
                         for index, score in zip(ids[0].tolist(), scores[0].tolist())],
            })
    return {"experiment": experiment, "lora_merged": experiment == "v4" and merge,
            "predictions": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-merge", action="store_true")
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    result = predict(args.checkpoint, args.input, torch.device(device), not args.no_merge)
    if args.output:
        save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
