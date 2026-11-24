import csv
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .common import DATA, OUTPUTS, RUNS, SEEDS, save_json
from .data import NormalizeBatch, classes, image_transform, load_rgb, split_rows
from .evaluate import read_predictions
from .models import load_model
from .predict import predict


def load_experiment(experiment, device):
    checkpoint = RUNS / f"{experiment}_s{SEEDS[0]}" / "best.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return load_model(state, device)


def explain_samples(entries, names, output, device):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    def reshape(tokens):
        patches = tokens[:, 1:]
        side = int(patches.shape[1] ** 0.5)
        return patches.reshape(patches.shape[0], side, side, -1).permute(0, 3, 1, 2)

    fig, axes = plt.subplots(5, 6, figsize=(15, 13), layout="constrained")
    for index, entry in enumerate(entries):
        row, group = divmod(index, 2)
        original = load_rgb(DATA / entry["path"])
        axes[row, group * 3].imshow(original)
        axes[row, group * 3].set_title(f"Label: {names[int(entry['label'])]}", fontsize=8)
    for column, experiment in enumerate(("v1", "v3"), start=1):
        model, data_config = load_experiment(experiment, device)
        transform = image_transform(data_config)
        norm = NormalizeBatch(data_config, device)
        if experiment == "v1":
            layers, reshape_transform = [model.layer4[-1]], None
        else:
            layers, reshape_transform = [model.blocks[-1].norm1], reshape
            model.set_grad_checkpointing(False)
            for block in model.blocks:
                block.attn.fused_attn = False
        prediction_key = f"{experiment}_prediction"
        with GradCAM(model=model, target_layers=layers, reshape_transform=reshape_transform) as cam:
            for index, entry in enumerate(entries):
                image = transform(load_rgb(DATA / entry["path"]))
                pixels = image.permute(1, 2, 0).numpy().astype(np.float32) / 255
                inputs = norm(image.unsqueeze(0))
                heatmap = cam(inputs, [ClassifierOutputTarget(entry[prediction_key])])[0]
                row, group = divmod(index, 2)
                ax = axes[row, group * 3 + column]
                ax.imshow(show_cam_on_image(pixels, heatmap, use_rgb=True))
                ax.set_title(f"{experiment.upper()}: {names[entry[prediction_key]]}", fontsize=8)
        del model
    for ax in axes.ravel():
        ax.set_axis_off()
    fig.savefig(output / "grad_cam_v1_v3.jpg", dpi=135)
    plt.close(fig)


def sample_errors(predictions, names, output):
    rng = random.Random(SEEDS[0])
    errors = [prediction for prediction in predictions
              if prediction["label"] != prediction["top1"]]
    selected = rng.sample(errors, min(20, len(errors)))
    fig, axes = plt.subplots(4, 5, figsize=(15, 11.5), layout="constrained")
    rows = []
    for ax in axes.ravel():
        ax.set_axis_off()
    for index, (ax, error) in enumerate(zip(axes.ravel(), selected), start=1):
        actual = names[int(error["label"])]
        predicted = names[int(error["top1"])]
        score = float(error["p1"])
        original = load_rgb(DATA / error["path"])
        original.thumbnail((380, 310))
        ax.imshow(original)
        ax.set_title(f"{index:02}  {actual}\n→ {predicted} ({score:.0%})", fontsize=8)
        rows.append({
            "id": index, "path": error["path"], "label": actual,
            "predicted": predicted, "confidence": score,
            "true_in_top5": error["label"] in [error[f"top{k}"] for k in range(2, 6)],
        })
    with (output / "error_samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, ["id", "path", "label", "predicted", "confidence",
                                         "true_in_top5"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    fig.savefig(output / "v3_errors.jpg", dpi=145)
    plt.close(fig)


def inference_demo(output, device):
    checkpoint = RUNS / f"v4_s{SEEDS[0]}" / "best.pt"
    row = random.Random(SEEDS[0]).choice(split_rows("val"))
    path = DATA / row["path"]
    merged = predict(checkpoint, path, device, merge=True)
    separate = predict(checkpoint, path, device, merge=False)
    top3 = merged["predictions"][0]["top3"]
    reference = separate["predictions"][0]["top3"]
    agreement = [item["breed"] for item in top3] == [item["breed"] for item in reference]
    delta = max(abs(a["confidence"] - b["confidence"]) for a, b in zip(top3, reference))
    if not agreement or delta > 0.001:
        raise ValueError(f"Merged LoRA output differs: agreement={agreement}, delta={delta}")
    save_json(output / "lora_merged_prediction.json", {
        "image": row["path"], "label": row["breed"],
        "lora_merged": merged["lora_merged"], "top3": top3,
        "unmerged_top3": reference, "top3_agreement": agreement, "max_score_delta": delta,
    })


def main():
    output = OUTPUTS / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    names = [entry["breed"] for entry in classes()]
    v1 = read_predictions(f"v1_s{SEEDS[0]}")
    v3 = read_predictions(f"v3_s{SEEDS[0]}")
    entries = []
    for left, right in zip(v1, v3):
        if left["path"] != right["path"] or left["label"] != right["label"]:
            raise ValueError("The prediction tables use different sample orders.")
        entries.append({"path": right["path"], "label": int(right["label"]),
                        "v1_prediction": int(left["top1"]), "v3_prediction": int(right["top1"])})
    picked = random.Random(SEEDS[0]).sample(entries, 10)
    save_json(output / "grad_cam_samples.json", picked)
    sample_errors(v3, names, output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    explain_samples(picked, names, output, device)
    inference_demo(output, device)


if __name__ == "__main__":
    main()
