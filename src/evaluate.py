import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from .common import OUTPUTS, RUNS, SEEDS, SPLITS, device_and_amp, read_json, save_json, sha256
from .data import NormalizeBatch, PetDataset, classes, image_transform, make_loader, split_rows
from .models import load_model, vit_feature


def calculate_metrics(topk, labels, breeds):
    predicted = topk[:, 0]
    class_ids = list(range(len(breeds)))
    matrix = confusion_matrix(labels, predicted, labels=class_ids)
    counts = matrix.sum(axis=1)
    species = np.array([entry["species"] for entry in breeds])
    switch = species[labels] != species[predicted]
    cross = {}
    for actual, wrong in (("cat", "dog"), ("dog", "cat")):
        mask = species[labels] == actual
        errors = int(np.sum(mask & switch))
        cross[f"{actual}_as_{wrong}"] = {
            "errors": errors,
            "total": int(mask.sum()),
            "rate": errors / int(mask.sum()) if mask.any() else None,
        }
    pairs = sorted(
        [(int(matrix[a, b] + matrix[b, a]), a, b) for a in class_ids for b in range(a + 1, len(breeds))],
        reverse=True,
    )
    return {
        "samples": len(labels),
        "top1": float(np.mean(predicted == labels)),
        "top5": float(np.mean(np.any(topk == labels[:, None], axis=1))),
        "macro_f1": float(f1_score(labels, predicted, labels=class_ids,
                                   average="macro", zero_division=0)),
        "species_errors": int(switch.sum()),
        "species_error_rate": float(switch.mean()),
        "species_directions": cross,
        "per_class": [
            {**breed, "count": int(counts[index]), "correct": int(matrix[index, index]),
             "accuracy": float(matrix[index, index] / counts[index]) if counts[index] else None}
            for index, breed in enumerate(breeds)
        ],
        "confusion_matrix": matrix.tolist(),
        "top_confusions": [
            {"class_a": breeds[a]["breed"], "class_b": breeds[b]["breed"], "errors": value,
             "a_as_b": int(matrix[a, b]), "b_as_a": int(matrix[b, a])}
            for value, a, b in pairs[:10] if value
        ],
    }


def draw_confusion(path, values, breeds, title):
    matrix = np.asarray(values)
    counts = matrix.sum(axis=1, keepdims=True)
    rates = np.divide(matrix, counts, out=np.zeros_like(matrix, dtype=float), where=counts != 0)
    names = [entry["breed"] for entry in breeds]
    fig, ax = plt.subplots(figsize=(11, 10), layout="constrained")
    image = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(names)), names, fontsize=6.5)
    ax.set(title=title, xlabel="Predicted breed", ylabel="Annotated breed")
    fig.colorbar(image, ax=ax, label="Fraction of annotated breed", shrink=.72)
    fig.savefig(path, dpi=135)
    plt.close(fig)


def predict_test(checkpoint, workers, batch_size):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    device, amp = device_and_amp()
    torch.set_num_threads(2)
    model, data_config = load_model(state, device)
    config = state["config"]
    norm = NormalizeBatch(data_config, device)
    data = PetDataset("test", image_transform(data_config))
    loader = make_loader(data, batch_size, workers)
    topk, probabilities = [], []
    with torch.inference_mode():
        for images, _ in loader:
            with torch.autocast(device.type, amp):
                inputs = norm(images)
                if config["experiment"] == "v2":
                    logits = model.head(vit_feature(model, inputs, config["pool"]).float())
                else:
                    logits = model(inputs)
            values, indices = logits.float().softmax(dim=1).topk(5)
            topk.append(indices.cpu().numpy())
            probabilities.append(values.cpu().numpy())
    return np.concatenate(topk), np.concatenate(probabilities)


def read_predictions(name):
    path = OUTPUTS / "evaluation" / name / "predictions.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def evaluate(name, workers=6, batch_size=32):
    checkpoint = RUNS / name / "best.pt"
    training = read_json(OUTPUTS / "training" / name / "summary.json")
    if not training["complete"]:
        raise ValueError(f"{name} has not finished training.")
    output = OUTPUTS / "evaluation" / name
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    provenance = {
        "checkpoint_sha256": sha256(checkpoint),
        "manifest_sha256": sha256(SPLITS / "test.csv"),
        "seed": training["seed"],
        "epoch": training["best_epoch"],
    }
    if metrics_path.exists():
        previous = read_json(metrics_path)
        if previous["provenance"] != provenance:
            raise ValueError(f"The final test results for {name} refer to a different checkpoint.")
        if (output / "predictions.csv").exists() and (output / "confusion.png").exists():
            print(f"Using recorded test results for {name}", flush=True)
            return previous

    test_rows = split_rows("test")
    labels = np.array([int(row["label"]) for row in test_rows])
    topk, scores = predict_test(checkpoint, workers, batch_size)
    breeds = classes()
    metrics = calculate_metrics(topk, labels, breeds)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["path", "label", "top1", "top2", "top3", "top4", "top5",
                         "p1", "p2", "p3", "p4", "p5"])
        for row, label, indices, probabilities in zip(test_rows, labels, topk, scores):
            writer.writerow([row["path"], label, *indices, *(f"{p:.7f}" for p in probabilities)])
    draw_confusion(output / "confusion.png", metrics["confusion_matrix"], breeds, name)
    result = {**metrics, "provenance": provenance}
    save_json(metrics_path, result)
    print(f"{name} top-1={metrics['top1']:.2%} top-5={metrics['top5']:.2%} "
          f"Macro-F1={metrics['macro_f1']:.4f}", flush=True)
    return result


def evaluate_all(workers):
    names = [f"{experiment}_s{seed}" for experiment in ("v1", "v2", "v3", "v4")
             for seed in SEEDS]
    for name in names:
        if not read_json(OUTPUTS / "training" / name / "summary.json")["complete"]:
            raise ValueError(f"Complete the suite before final evaluation: {name}")
    return {name: evaluate(name, workers) for name in names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    evaluate_all(args.workers)


if __name__ == "__main__":
    main()
