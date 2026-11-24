import csv

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .common import DATA, SPLITS, read_json


def load_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def image_transform(data_config, augmentation="none"):
    size = data_config["input_size"][-1]
    interpolation = T.InterpolationMode(data_config["interpolation"])
    if augmentation == "none":
        resize = round(248 * size / 224)
        operations = [T.Resize(resize, interpolation), T.CenterCrop(size)]
    elif augmentation in {"basic", "rand"}:
        operations = [T.RandomResizedCrop(size, interpolation=interpolation),
                      T.RandomHorizontalFlip()]
        if augmentation == "rand":
            operations.append(T.RandAugment(interpolation=interpolation))
    else:
        raise ValueError(f"Unknown augmentation: {augmentation}")
    return T.Compose([*operations, T.PILToTensor()])


class NormalizeBatch:
    def __init__(self, data_config, device):
        self.device = device
        self.mean = torch.tensor(data_config["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(data_config["std"], device=device).view(1, 3, 1, 1)

    def __call__(self, images):
        images = images.to(self.device, non_blocking=True, dtype=torch.float32)
        return images.div_(255).sub_(self.mean).div_(self.std)


def split_rows(split):
    with (SPLITS / f"{split}.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def classes():
    return read_json(SPLITS / "classes.json")


class PetDataset(Dataset):
    def __init__(self, split, transform):
        self.rows = split_rows(split)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = load_rgb(DATA / row["path"])
        return self.transform(image), int(row["label"])


def make_loader(dataset, batch_size, workers, shuffle=False, seed=0, pin=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin,
        num_workers=workers,
        persistent_workers=bool(workers),
        generator=torch.Generator().manual_seed(seed),
    )
