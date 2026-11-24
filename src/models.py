import timm
from timm.data import resolve_data_config

from .common import NUM_CLASSES, WEIGHTS


def build_model(config, initialize=True):
    experiment = config["experiment"]
    model = timm.create_model(
        config["model"], pretrained=False, num_classes=NUM_CLASSES,
        drop_path_rate=config.get("drop_path_rate", 0.0),
    )
    data_config = resolve_data_config({}, model=model)
    if experiment != "v1" and initialize:
        from safetensors.torch import load_file

        weights = load_file(WEIGHTS)
        for key in ("head.weight", "head.bias"):
            weights.pop(key)
        missing, extra = model.load_state_dict(weights, strict=False)
        if set(missing) != {"head.weight", "head.bias"} or extra:
            raise ValueError(f"Unexpected pretrained keys: missing={missing}, extra={extra}")
    if experiment == "v2":
        model.requires_grad_(False)
        model.head.requires_grad_(True)
    if config.get("gradient_checkpointing"):
        model.set_grad_checkpointing(True)
    if experiment == "v4":
        from peft import LoraConfig, get_peft_model

        rank = config["rank"]
        model = get_peft_model(
            model,
            LoraConfig(
                r=rank,
                lora_alpha=2 * rank,
                lora_dropout=config["lora_dropout"],
                target_modules=["qkv"],
                modules_to_save=["head"],
                bias="none",
            ),
        )
    return model, data_config


def vit_feature(model, images, pool):
    tokens = model.forward_features(images)
    return pool_tokens(model, tokens, pool)


def pool_tokens(model, tokens, pool):
    if pool == "cls":
        return tokens[:, 0]
    if pool == "mean":
        return tokens[:, model.num_prefix_tokens:].mean(dim=1)
    raise ValueError(f"Unknown feature pooling: {pool}")


def load_model(checkpoint, device, merge=False):
    config = checkpoint["config"]
    linear = config["experiment"] == "v2"
    model, data_config = build_model(config, initialize=linear)
    if linear:
        model.head.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval()
    if merge and config["experiment"] == "v4":
        model = model.merge_and_unload(safe_merge=True)
    return model, data_config
