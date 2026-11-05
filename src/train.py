from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import (
    densenet121,
    DenseNet121_Weights,
)


class ChestXrayDataset(Dataset):
    """Read images from a verified Day 1 manifest."""

    def __init__(self, csv_path: str, transform):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        required = {"path", "label", "sha256", "group_key"}
        missing = required - set(self.df.columns)

        if missing:
            raise ValueError(
                f"Missing manifest columns: {sorted(missing)}"
            )

        if not self.df["label"].isin([0, 1]).all():
            raise ValueError("Labels must be binary: 0 or 1.")

        self.paths = self.df["path"].astype(str).tolist()
        self.labels = self.df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        with Image.open(path) as image:
            # DenseNet pretrained weights expect three channels.
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            image = self.transform(image)

        label = torch.tensor(
            self.labels[index],
            dtype=torch.float32,
        )

        return image, label


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility matters more than peak speed for this project.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_transforms(image_size: int):
    # Use ImageNet normalization to match pretrained DenseNet weights.
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),

        # Small rotation and translation; no horizontal flips because
        # laterality and embedded medical markers can be meaningful.
        transforms.RandomAffine(
            degrees=5,
            translate=(0.02, 0.02),
            fill=0,
        ),

        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, val_transform


def build_model(pretrained: bool):
    weights = (
        DenseNet121_Weights.IMAGENET1K_V1
        if pretrained else None
    )

    model = densenet121(weights=weights)

    in_features = model.classifier.in_features

    # One logit represents the positive (pneumonia) class.
    model.classifier = nn.Linear(in_features, 1)

    return model


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    training,
    amp_enabled,
):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_labels = []
    all_probabilities = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                logits = model(images).flatten()
                loss = criterion(logits, labels)

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.detach().item() * batch_size

        probabilities = torch.sigmoid(
            logits.detach().float()
        )

        all_labels.extend(
            labels.detach().cpu().numpy().tolist()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )

    average_loss = total_loss / len(loader.dataset)

    auc = roc_auc_score(
        np.asarray(all_labels),
        np.asarray(all_probabilities),
    )

    return float(average_loss), float(auc)


def save_curves(history, path):
    df = pd.DataFrame(history)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        df["epoch"], df["train_loss"], label="Train"
    )
    axes[0].plot(
        df["epoch"], df["val_loss"], label="Validation"
    )
    axes[0].set_title("Binary cross-entropy loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(
        df["epoch"], df["train_auc"], label="Train"
    )
    axes[1].plot(
        df["epoch"], df["val_auc"], label="Validation"
    )
    axes[1].set_title("ROC-AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = int(config["seed"])
    set_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU unavailable. Enable a GPU in Colab."
        )

    device = torch.device("cuda")
    print("Device:", torch.cuda.get_device_name(0))

    data_config = config["data"]
    train_config = config["training"]
    output_config = config["output"]

    image_size = int(data_config["image_size"])
    batch_size = int(data_config["batch_size"])
    num_workers = int(data_config["num_workers"])

    train_transform, val_transform = build_transforms(
        image_size
    )

    train_dataset = ChestXrayDataset(
        data_config["train_csv"],
        train_transform,
    )
    val_dataset = ChestXrayDataset(
        data_config["val_csv"],
        val_transform,
    )

    # Defensive leakage checks before touching the model.
    train_df = train_dataset.df
    val_df = val_dataset.df

    if set(train_df["sha256"]) & set(val_df["sha256"]):
        raise RuntimeError(
            "Exact duplicate leakage between train and validation."
        )

    if set(train_df["group_key"]) & set(val_df["group_key"]):
        raise RuntimeError(
            "Group leakage between train and validation."
        )

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    train_labels = np.asarray(train_dataset.labels)
    n_normal = int((train_labels == 0).sum())
    n_pneumonia = int((train_labels == 1).sum())

    # Balances aggregate positive and negative loss contributions.
    positive_weight = n_normal / n_pneumonia

    print("Training NORMAL:", n_normal)
    print("Training PNEUMONIA:", n_pneumonia)
    print("Positive class weight:", round(positive_weight, 5))
    print("Validation images:", len(val_dataset))

    model = build_model(
        pretrained=bool(config["model"]["pretrained"])
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [positive_weight],
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )

    amp_enabled = bool(train_config["use_amp"])

    # CUDA GradScaler avoids underflow during mixed-precision training.
    scaler = (
        torch.amp.GradScaler("cuda")
        if amp_enabled else None
    )

    checkpoint_path = Path(output_config["checkpoint"])
    history_path = Path(output_config["history"])
    figure_path = Path(output_config["figure"])

    for path in [checkpoint_path, history_path, figure_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    best_auc = float("-inf")
    best_epoch = 0
    patience_counter = 0

    epochs = int(train_config["epochs"])
    patience = int(train_config["patience"])

    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss, train_auc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            training=True,
            amp_enabled=amp_enabled,
        )

        val_loss, val_auc = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            scaler=None,
            training=False,
            amp_enabled=amp_enabled,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_auc": train_auc,
            "val_loss": val_loss,
            "val_auc": val_auc,
        }

        history.append(record)

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_val_auc": best_auc,
                    "positive_weight": positive_weight,
                    "image_size": image_size,
                    "seed": seed,
                    "model_name": "densenet121",
                    "class_names": ["NORMAL", "PNEUMONIA"],
                    "config": config,
                },
                checkpoint_path,
            )
            status = "BEST"
        else:
            patience_counter += 1
            status = ""

        pd.DataFrame(history).to_csv(
            history_path, index=False
        )
        save_curves(history, figure_path)

        elapsed = time.time() - start

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"train_auc={train_auc:.4f} | "
            f"val_auc={val_auc:.4f} | "
            f"{elapsed:.0f}s {status}",
            flush=True,
        )

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    summary = {
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "epochs_completed": len(history),
        "train_images": len(train_dataset),
        "val_images": len(val_dataset),
        "positive_weight": positive_weight,
        "checkpoint": str(checkpoint_path),
    }

    summary_path = Path("reports/training_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nTRAINING COMPLETE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
