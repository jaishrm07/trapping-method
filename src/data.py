"""Dataset loaders for the trapping paper's setting.

The paper uses ImageNet (primary) and Stanford Cars / Food101 / Country211
(restricted "harmful" datasets). All adversary fine-tuning is done via linear
probing on a frozen feature extractor, so we just need (image, label) tuples
with the standard ImageNet preprocessing.

Stanford Cars: torchvision.datasets.StanfordCars stopped working after
Stanford pulled the dataset URL (2023). We fall back to HuggingFace.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision import datasets as tv_datasets


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_train_transform(image_size: int = 224) -> Callable:
    return T.Compose([
        T.RandomResizedCrop(image_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_eval_transform(image_size: int = 224) -> Callable:
    return T.Compose([
        T.Resize(int(image_size * 256 / 224)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@dataclass
class DatasetSplits:
    train: Dataset
    test: Dataset
    num_classes: int


# -----------------------------------------------------------------------------
# Stanford Cars via HuggingFace `datasets`
# -----------------------------------------------------------------------------

class HFCarsWrapper(Dataset):
    """Wrap a HuggingFace `datasets` split into a (PIL→tensor, int) Dataset.

    HF returns dicts; PyTorch's ImageFolder-style code expects tuples. This
    bridge keeps the rest of the pipeline backbone-agnostic.
    """

    def __init__(self, hf_split, transform: Callable, image_key: str = "image", label_key: str = "label"):
        self.split = hf_split
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.split[idx]
        img = item[self.image_key]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img), int(item[self.label_key])


def load_stanford_cars(image_size: int = 224, cache_dir: str | None = None) -> DatasetSplits:
    """Stanford Cars (196 classes). 8144 train, 8041 test."""
    from datasets import load_dataset
    ds = load_dataset("tanganke/stanford_cars", cache_dir=cache_dir)
    train = HFCarsWrapper(ds["train"], make_train_transform(image_size))
    test = HFCarsWrapper(ds["test"], make_eval_transform(image_size))
    return DatasetSplits(train=train, test=test, num_classes=196)


# -----------------------------------------------------------------------------
# Food-101 (torchvision works fine for this one)
# -----------------------------------------------------------------------------

def load_food101(root: str, image_size: int = 224, download: bool = True) -> DatasetSplits:
    """Food-101 (101 classes). 75750 train, 25250 test."""
    Path(root).mkdir(parents=True, exist_ok=True)
    train = tv_datasets.Food101(root=root, split="train", transform=make_train_transform(image_size), download=download)
    test = tv_datasets.Food101(root=root, split="test", transform=make_eval_transform(image_size), download=download)
    return DatasetSplits(train=train, test=test, num_classes=101)


# -----------------------------------------------------------------------------
# Country-211 via HuggingFace
# -----------------------------------------------------------------------------

def load_country211(image_size: int = 224, cache_dir: str | None = None) -> DatasetSplits:
    """Country-211 (211 classes, 100 train + 50 valid + 100 test images per class)."""
    from datasets import load_dataset
    ds = load_dataset("clip-benchmark/wds_country211", cache_dir=cache_dir)
    train = HFCarsWrapper(ds["train"], make_train_transform(image_size), image_key="webp", label_key="cls")
    test = HFCarsWrapper(ds["test"], make_eval_transform(image_size), image_key="webp", label_key="cls")
    return DatasetSplits(train=train, test=test, num_classes=211)


# -----------------------------------------------------------------------------
# Single entry point for experiments
# -----------------------------------------------------------------------------

def load_dataset_by_name(name: str, root: str = "./data", image_size: int = 224, cache_dir: str | None = None) -> DatasetSplits:
    name = name.lower()
    if name == "cars":
        return load_stanford_cars(image_size=image_size, cache_dir=cache_dir)
    if name == "food101":
        return load_food101(root=root, image_size=image_size)
    if name == "country211":
        return load_country211(image_size=image_size, cache_dir=cache_dir)
    raise ValueError(f"Unknown dataset: {name}. Expected one of: cars, food101, country211.")


def make_loaders(splits: DatasetSplits, batch_size: int = 64, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(splits.train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(splits.test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
