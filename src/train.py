import argparse
import gc
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter

from .common import (
    OUTPUTS, ROOT, RUNS, SPLITS, WEIGHTS, device_and_amp, read_json,
    save_checkpoint, save_json, seed_all, sha256,
)
from .data import NormalizeBatch, PetDataset, image_transform, make_loader
from .models import build_model, pool_tokens

CONFIG_FILES = {
    "v1": "v1_scratch", "v2": "v2_linear", "v3": "v3_full", "v4": "v4_lora",
}


def default_config(experiment):
    path = ROOT / "configs" / f"{CONFIG_FILES[experiment]}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def features(model, data_config, config, device, amp, workers):
    cache_dir = RUNS / "features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    norm = NormalizeBatch(data_config, device)
    model.eval()
    datasets = []
    elapsed = 0.0
    peak = 0
    for split in ("train", "val"):
        path = cache_dir / f"{split}.pt"
        key = {
            "split": sha256(SPLITS / f"{split}.csv"),
            "weights": sha256(WEIGHTS),
            "data_config": data_config,
            "precision": str(amp),
            "torch": str(torch.__version__),
        }
        content = torch.load(path, weights_only=True) if path.exists() else None
        if content is None or content["key"] != key:
            dataset = PetDataset(split, image_transform(data_config))
            loader = make_loader(dataset, config["feature_batch_size"], workers)
            vectors = {"cls": [], "mean": []}
            labels = []
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                for images, targets in loader:
                    with torch.autocast(device.type, amp):
                        all_tokens = model.forward_features(norm(images))
                    for pool in vectors:
                        vectors[pool].append(pool_tokens(model, all_tokens, pool).float().cpu())
                    labels.append(targets)
            torch.cuda.synchronize()
            content = {
                "key": key,
                "vectors": {pool: torch.cat(batches) for pool, batches in vectors.items()},
                "labels": torch.cat(labels),
                "seconds": time.perf_counter() - started,
                "peak_bytes": torch.cuda.max_memory_allocated(),
            }
            save_checkpoint(path, content)
            print(f"Cached {split}: {len(dataset)} in {content['seconds']:.1f}s", flush=True)
        elapsed += content["seconds"]
        peak = max(peak, content["peak_bytes"])
        datasets.append(TensorDataset(content["vectors"][config["pool"]], content["labels"]))
    return datasets, {"seconds": elapsed, "peak_bytes": peak}


def make_optimizer(model, config):
    if config["optimizer"] == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=config["lr"], momentum=0.9,
            weight_decay=config["weight_decay"], foreach=True,
        )
    groups = []
    if "head_lr" in config:
        head = list(model.head.parameters())
        head_ids = {id(parameter) for parameter in head}
        groups = [
            {"params": [parameter for parameter in model.parameters()
                        if id(parameter) not in head_ids]},
            {"params": head, "lr": config["head_lr"]},
        ]
    else:
        groups = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(
        groups, lr=config["lr"], weight_decay=config["weight_decay"], fused=True,
    )


def learning_rate_factor(step, warmup, total):
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup - 1, 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def run_epoch(model, loader, preprocess, amp, device, smoothing=0.0, optimizer=None,
              scheduler=None, scaler=None, accumulate=1):
    training = optimizer is not None
    model.train(training)
    seen = 0
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.int64, device=device)
    wait = 0.0
    window_items = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    iterator = iter(loader)
    for step in range(len(loader)):
        before_load = time.perf_counter()
        data, labels = next(iterator)
        wait += time.perf_counter() - before_load
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast(device.type, amp):
            logits = model(preprocess(data))
            loss = F.cross_entropy(logits, labels, label_smoothing=smoothing)
        if training:
            if step % accumulate == 0:
                window_items = min(
                    loader.batch_size * accumulate, len(loader.dataset) - seen,
                )
            scaler.scale(loss * (labels.size(0) / window_items)).backward()
            if (step + 1) % accumulate == 0 or step + 1 == len(loader):
                scale = scaler.get_scale() if scaler.is_enabled() else 1
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if not scaler.is_enabled() or scaler.get_scale() >= scale:
                    scheduler.step()
        seen += labels.size(0)
        total_loss += loss.detach() * labels.size(0)
        correct += (logits.detach().argmax(1) == labels).sum()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "seconds": elapsed,
        "images_per_second": seen / elapsed,
        "loader_wait_seconds": wait,
        "loss": total_loss.item() / seen,
        "top1": correct.item() / seen,
    }


def train(config, seed, name, workers=6, smoke_limit=None):
    device, amp = device_and_amp()
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = config["experiment"] == "v1"
    seed_all(seed)
    work = RUNS / name
    output = OUTPUTS / "training" / name
    if smoke_limit is not None:
        output = OUTPUTS / ".work" / name
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    best_path, last_path = work / "best.pt", work / "last.pt"
    summary_path = output / "summary.json"
    if summary_path.exists():
        previous = read_json(summary_path)
        if previous["config"] != config or previous["seed"] != seed:
            raise ValueError(f"Run {name} exists with a different configuration.")
        if previous["complete"] and best_path.exists():
            print(f"Already complete: {name}", flush=True)
            return previous

    linear = config["experiment"] == "v2"
    model, data_config = build_model(config)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    model = model.to(device)
    cache = None
    if linear:
        (train_data, val_data), cache = features(model, data_config, config, device, amp, workers)
        model = model.head
        preprocess = lambda data: data.to(device, non_blocking=True)
    else:
        norm = NormalizeBatch(data_config, device)
        preprocess = norm
        if config["experiment"] == "v1":
            model.to(memory_format=torch.channels_last)
            preprocess = lambda data: norm(data).contiguous(memory_format=torch.channels_last)
        train_data = PetDataset("train", image_transform(data_config, config["augmentation"]))
        val_data = PetDataset("val", image_transform(data_config))
    if smoke_limit:
        train_data = torch.utils.data.Subset(train_data, range(min(smoke_limit, len(train_data))))
        val_data = torch.utils.data.Subset(val_data, range(min(smoke_limit, len(val_data))))
    train_loader = make_loader(train_data, config["batch_size"], workers, shuffle=True, seed=seed)
    val_loader = make_loader(val_data, config.get("eval_batch_size", config["batch_size"]), workers)
    optimizer = make_optimizer(model, config)
    steps = math.ceil(len(train_loader) / config["accumulate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_factor(
            step, steps * config["warmup_epochs"], steps * config["epochs"]
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
    history = []
    best = -1.0
    best_epoch = 0
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        if state["config"] != config or state["seed"] != seed:
            raise ValueError(f"Checkpoint {last_path} uses a different configuration or seed.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, best, best_epoch = state["history"], state["best"], state["best_epoch"]
        del state
    start = len(history)
    print(f"Training {name} on {torch.cuda.get_device_name()} | {params:,} parameters "
          f"| {start}/{config['epochs']} epochs", flush=True)
    with SummaryWriter(log_dir=str(work / "tensorboard"), purge_step=start + 1) as writer:
        for epoch in range(start, config["epochs"]):
            train_loader.generator.manual_seed(seed + epoch)
            seed_all(seed + epoch)
            torch.cuda.reset_peak_memory_stats()
            train_result = run_epoch(
                model, train_loader, preprocess, amp, device,
                smoothing=config.get("label_smoothing", 0.0), optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, accumulate=config["accumulate"],
            )
            peak = torch.cuda.max_memory_allocated()
            validation = run_epoch(model, val_loader, preprocess, amp, device)
            precision = validation["top1"]
            log = {"epoch": epoch + 1, "train": train_result, "val": validation,
                   "peak_bytes": peak, "lr": scheduler.get_last_lr()}
            history.append(log)
            metadata = {"config": config, "seed": seed, "model": model.state_dict()}
            if precision > best:
                best, best_epoch = precision, epoch + 1
                save_checkpoint(best_path, {**metadata, "epoch": best_epoch, "val_top1": best})
            save_checkpoint(last_path, {
                **metadata, "history": history, "best": best, "best_epoch": best_epoch,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            })
            for split in ("train", "val"):
                for key, value in log[split].items():
                    writer.add_scalar(f"{split}/{key}", value, epoch + 1)
            writer.add_scalar("train/peak_bytes", peak, epoch + 1)
            save_json(output / "history.json", history)
            print(f"{name} {epoch + 1}/{config['epochs']} "
                  f"train={train_result['top1']:.2%} val={precision:.2%} "
                  f"{train_result['images_per_second']:.0f} img/s "
                  f"{peak / 2 ** 30:.2f} GiB", flush=True)
    summary = {
        "complete": len(history) == config["epochs"] and smoke_limit is None,
        "config": config, "seed": seed, "workers": workers, "pin_memory": True,
        "device": torch.cuda.get_device_name(), "amp": str(amp),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "parameters_trainable": params, "parameters_total": all_params,
        "best_val_top1": best, "best_epoch": best_epoch,
        "feature_extraction": cache,
        "peak_bytes": max(row["peak_bytes"] for row in history),
        "median_train_seconds": statistics.median(row["train"]["seconds"] for row in history),
        "median_train_ips": statistics.median(row["train"]["images_per_second"] for row in history),
        "total_training_seconds": sum(
            row["train"]["seconds"] + row["val"]["seconds"] for row in history
        ),
    }
    save_json(summary_path, summary)
    del train_loader, val_loader, model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--name")
    parser.add_argument("--smoke-images", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    name = args.name or f"{config['experiment']}_s{args.seed}"
    if args.smoke_images is not None:
        if args.smoke_images < 2:
            parser.error("--smoke-images must be at least 2")
        config.update(epochs=1, warmup_epochs=0)
        name = f"smoke_{name}"
    train(config, args.seed, name, args.workers, args.smoke_images)


if __name__ == "__main__":
    main()
