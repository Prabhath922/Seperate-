from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders(
    dataset_dir: Path,
    img_size: int,
    batch_size: int,
    val_split: float,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    augmentation = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    base_transform = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = datasets.ImageFolder(root=str(dataset_dir), transform=base_transform)
    class_names = dataset.classes

    val_len = int(len(dataset) * val_split)
    train_len = len(dataset) - val_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    # Apply augmentation to the training subset
    train_dataset.dataset = datasets.ImageFolder(root=str(dataset_dir), transform=augmentation)

    num_workers = min(4, max(1, torch.get_num_threads() // 2))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, class_names


def build_model(num_classes: int) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> Tuple[float, float]:
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    epoch_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    average_loss = epoch_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train a waste image classification model with PyTorch.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/Users/prabhathsundarapalli/Downloads/dataset-resized 2"),
        help="Path to the dataset directory that contains class subfolders.",
    )
    parser.add_argument("--img-size", type=int, default=224, help="Image size to resize the dataset to.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs to train.")
    parser.add_argument("--finetune-epochs", type=int, default=3, help="Number of epochs to fine-tune the entire model.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for warmup training.")
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-4, help="Learning rate for fine-tuning.")
    parser.add_argument("--model-output", type=Path, default=Path("model.pth"), help="Where to store the trained model state.")
    parser.add_argument("--labels-output", type=Path, default=Path("labels.json"), help="Where to store class labels.")

    args = parser.parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset directory not found: {args.dataset}")

    device = get_device()
    train_loader, val_loader, class_names = build_dataloaders(
        dataset_dir=args.dataset,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
    )

    model = build_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()

    history: Dict[str, List[Dict[str, float]]] = {"warmup": [], "finetune": []}

    warmup_epochs = max(args.epochs - args.finetune_epochs, 0)
    if warmup_epochs > 0:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.learning_rate)

        for epoch in range(1, warmup_epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, device, criterion, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, device, criterion, optimizer=None)
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            history["warmup"].append(metrics)
            print(f"[Warmup {epoch}/{warmup_epochs}] "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    if args.finetune_epochs > 0:
        for param in model.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_learning_rate)
        for epoch in range(1, args.finetune_epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, device, criterion, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, device, criterion, optimizer=None)
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            history["finetune"].append(metrics)
            print(f"[Finetune {epoch}/{args.finetune_epochs}] "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": class_names,
            "model_name": "resnet18",
            "num_classes": len(class_names),
        },
        args.model_output,
    )
    with open(args.labels_output, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    print(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()

