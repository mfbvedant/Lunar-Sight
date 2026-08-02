"""
Segmentation Training Loop
=============================
Colab-optimized training with self-training iterations.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

logger = logging.getLogger(__name__)


def train_segmentation(
    model: nn.Module,
    train_loader,
    val_loader,
    config: dict,
    checkpoint_dir: str | Path,
    device: str = "auto",
    resume: bool = True,
) -> dict:
    """Train the U-Net segmentation model with self-training.

    Self-training iterations:
        1. Train on sparse pseudo-labels
        2. Every N epochs, expand labels with high-confidence predictions
        3. Continue training on expanded label set

    Args:
        model: U-Net model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Training config dict.
        checkpoint_dir: Checkpoint directory.
        device: Device.
        resume: Whether to resume from checkpoint.

    Returns:
        Training history dict.
    """
    from agent4_segmentation.loss import CombinedSegmentationLoss

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seg_config = config.get("segmentation", {})
    epochs = seg_config.get("epochs", 80)
    lr = seg_config.get("learning_rate", 1e-4)
    weight_decay = seg_config.get("weight_decay", 1e-4)
    ckpt_every = seg_config.get("checkpoint_every_n_epochs", 5)
    patience = seg_config.get("early_stopping_patience", 15)
    alpha = seg_config.get("loss_bce_weight", 1.0)
    beta = seg_config.get("loss_dice_weight", 0.5)
    class_weights = seg_config.get("class_weights", [1.0, 3.0])

    model = model.to(device)
    criterion = CombinedSegmentationLoss(alpha, beta, class_weights)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0
    best_val_loss = float('inf')
    no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'epoch': []}

    if resume:
        start_epoch, best_val_loss = _load_checkpoint(
            model, optimizer, scheduler, checkpoint_dir, device
        )

    session_start = time.time()
    csv_path = checkpoint_dir / "training_log.csv"

    logger.info(f"Training: epochs {start_epoch+1}→{epochs}, device={device}")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # ---- Train ----
        model.train()
        train_losses = []

        for batch_idx, (features, labels, valid_mask) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)
            valid_mask = valid_mask.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss, components = criterion(logits, labels, valid_mask)
            loss.backward()
            optimizer.step()

            train_losses.append(components['total'])

        # ---- Validate ----
        model.eval()
        val_losses = []

        with torch.no_grad():
            for features, labels, valid_mask in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                valid_mask = valid_mask.to(device)
                logits = model(features)
                loss, components = criterion(logits, labels, valid_mask)
                val_losses.append(components['total'])

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses) if val_losses else float('inf')
        scheduler.step()

        epoch_time = time.time() - epoch_start
        session_elapsed = (time.time() - session_start) / 3600

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['epoch'].append(epoch + 1)

        logger.info(
            f"Epoch {epoch+1}/{epochs} | Train: {avg_train:.6f} | "
            f"Val: {avg_val:.6f} | Time: {epoch_time:.1f}s | "
            f"Session: {session_elapsed:.1f}h"
        )

        # Checkpoint
        if (epoch + 1) % ckpt_every == 0:
            _save_checkpoint(model, optimizer, scheduler, epoch + 1,
                            avg_val, best_val_loss, checkpoint_dir)

        # Best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            no_improve = 0
            _save_checkpoint(model, optimizer, scheduler, epoch + 1,
                            avg_val, best_val_loss, checkpoint_dir, is_best=True)
        else:
            no_improve += 1

        _log_csv(csv_path, epoch + 1, avg_train, avg_val)

        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

        if session_elapsed > 10.0:
            logger.warning("⚠ Session > 10h. Saving checkpoint.")
            _save_checkpoint(model, optimizer, scheduler, epoch + 1,
                            avg_val, best_val_loss, checkpoint_dir)

    return {
        'history': history,
        'best_val_loss': best_val_loss,
        'final_epoch': epoch + 1,
    }


def _save_checkpoint(model, optimizer, scheduler, epoch, val_loss,
                     best_val_loss, checkpoint_dir, is_best=False):
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
    }
    path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch:04d}.pt"
    torch.save(ckpt, str(path))
    if is_best:
        torch.save(ckpt, str(Path(checkpoint_dir) / "best_model.pt"))


def _load_checkpoint(model, optimizer, scheduler, checkpoint_dir, device):
    best = Path(checkpoint_dir) / "best_model.pt"
    if best.exists():
        ckpt = torch.load(str(best), map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        return ckpt['epoch'], ckpt.get('best_val_loss', float('inf'))
    return 0, float('inf')


def _log_csv(path, epoch, train_loss, val_loss):
    write_header = not path.exists()
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['epoch', 'train_loss', 'val_loss'])
        w.writerow([epoch, f'{train_loss:.6f}', f'{val_loss:.6f}'])
