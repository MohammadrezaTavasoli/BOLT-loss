#!/usr/bin/env python3
"""
train_cifar10.py
=================
Train a CIFAR‑10 classifier with BOLT loss (Bayes Optimal Learning Threshold).

Usage
-----
$ python train_cifar10.py --epochs 100 --batch-size 128 --norm l2

By default the script uses Apple Silicon MPS if available, else CUDA, else CPU.
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ----------------------- BOLT loss ------------------------------------- #
def BOLT_loss(logits: torch.Tensor,
              targets: torch.Tensor,
              norm: str = "l2") -> torch.Tensor:
    """Batch‑averaged BOLT loss.

    Parameters
    ----------
    logits  : (B, K) raw network outputs for *K ≥ 2* classes
    targets : (B,)   ground‑truth labels in *0 … K‑1*
    norm    : "l1" | "l2" – absolute‑ or squared‑error variant

    Returns
    -------
    torch.Tensor – scalar loss (batch average)
    """
    # Convert logits → probabilities and drop class‑0 column
    probs = F.softmax(logits, dim=1)[:, 1:]           # (B, K‑1)

    B, C = probs.size()
    # Build a class‑index matrix  [[0,1,…,C−1], …]  shape: (B, C)
    class_idx = torch.arange(C, device=targets.device).expand(B, C)
    tgt = targets.unsqueeze(1).expand_as(class_idx)   # broadcast targets

    loss_mat = (class_idx >= tgt).float() * probs
    loss_mat += (class_idx == (tgt - 1)).float() * (1.0 - probs)

    if norm.lower() == "l2":
        return loss_mat.pow(2).sum() / B
    elif norm.lower() == "l1":
        return loss_mat.abs().sum() / B
    else:
        raise ValueError("norm must be 'l1' or 'l2'")


class BOLTLoss(nn.Module):
    """nn.Module wrapper so we can pass it as a Criterion."""
    def __init__(self, norm: str = "l2"):
        super().__init__()
        self.norm = norm

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return BOLT_loss(logits, targets, norm=self.norm)


# ----------------------- model ----------------------------------------- #
def get_resnet18(num_classes: int = 10) -> nn.Module:
    """ResNet‑18 customised for 32×32 CIFAR images."""
    model = models.resnet18(weights=None)
    # Replace first conv & max‑pool for small images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                            padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ----------------------- training utils -------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += inputs.size(0)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += inputs.size(0)

    return running_loss / total, 100.0 * correct / total


# ----------------------- main ------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="Train CIFAR‑10 with BOLT loss")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--norm", choices=["l1", "l2"], default="l2",
                        help="BOLT loss norm variant")
    parser.add_argument("--save-model", action="store_true",
                        help="Save best model weights to ./best_model.pt")
    args = parser.parse_args()

    # Select device in order: MPS → CUDA → CPU
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    # Data pipelines
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True,
                                 transform=transform_train)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True,
                                transform=transform_test)

    pin_memory = device.type == "cuda"   # only matters for CUDA
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=4,
                              pin_memory=pin_memory, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=4,
                             pin_memory=pin_memory, persistent_workers=True)

    # Model, criterion, optimiser
    model = get_resnet18().to(device)
    criterion = BOLTLoss(norm=args.norm)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader,
                                       criterion, device)
        dt = time.time() - t0

        print(f"[Epoch {epoch:03d}/{args.epochs}]  "
              f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.2f}%  "
              f"test_loss: {test_loss:.4f}  test_acc: {test_acc:.2f}%  "
              f"time: {dt:.1f}s")

        if args.save_model and test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_model.pt")

    if args.save_model:
        print(f"Best test accuracy: {best_acc:.2f}%  → model saved to best_model.pt")


if __name__ == "__main__":
    main()
