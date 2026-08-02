"""
Despeckling Training Loop
===========================
Training loop optimized for Google Colab Free Tier with aggressive
checkpointing, resume support, and session time management.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

logger = logging.getLogger(__name__)


def train_despeckling(
    model: nn.Module,
    train_loader,
    val_loader,
    config: dict,
    checkpoint_dir: str | Path,
    device: str = "auto",
    resume: bool = True,
) -> dict:
    """Train the CV-CNN despeckling model.

    Features:
        - Checkpoint every N epochs to Google Drive
        - Auto-resume from latest checkpoint
        - Early stopping on validation loss
        - Session time tracking (warns at 10hr mark)
        - Loss logging to CSV

    Args:
        model: CV-CNN model instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Training config dict (from training_config.yaml).
        checkpoint_dir: Directory for saving checkpoints.
        device: Device ("cuda", "cpu", or "auto").
        resume: Whether to try resuming from checkpoint.

    Returns:
        Dict with training history and best metrics.
    """
    from agent2_despeckling.loss import CombinedDespecklingLoss

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    desp_config = config.get("despeckling", {})
    epochs = desp_config.get("epochs", 80)
    lr = desp_config.get("learning_rate", 1e-3)
    weight_decay = desp_config.get("weight_decay", 1e-4)
    ckpt_every = desp_config.get("checkpoint_every_n_epochs", 5)
    patience = desp_config.get("early_stopping_patience", 15)
    alpha = desp_config.get("loss_frobenius_weight", 1.0)
    beta = desp_config.get("loss_phase_weight", 0.5)

    model = model.to(device)
    criterion = CombinedDespecklingLoss(alpha=alpha, beta=beta)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    no_improve_count = 0
    history = {'train_loss': [], 'val_loss': [], 'epoch': []}

    if resume:
        start_epoch, best_val_loss = _load_checkpoint(
            model, optimizer, scheduler, checkpoint_dir, device
        )

    session_start = time.time()
    csv_path = checkpoint_dir / "training_log.csv"

    logger.info(f"Starting training: epochs {start_epoch+1}→{epochs}, device={device}")
    logger.info(f"Checkpoints → {checkpoint_dir}")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # ---- Training ----
        model.train()
        train_losses = []

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Convert to complex for the network
            inputs_c = torch.complex(inputs, torch.zeros_like(inputs))
            targets_c = torch.complex(targets, torch.zeros_like(targets))

            optimizer.zero_grad()
            outputs = model(inputs_c)
            loss, components = criterion(outputs, targets_c)
            loss.backward()
            optimizer.step()

            train_losses.append(components['total'])

            if batch_idx % 10 == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {components['total']:.6f} "
                    f"(Frob: {components['frobenius']:.6f}, Phase: {components['phase']:.6f})"
                )

        # ---- Validation ----
        model.eval()
        val_losses = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                inputs_c = torch.complex(inputs, torch.zeros_like(inputs))
                targets_c = torch.complex(targets, torch.zeros_like(targets))

                outputs = model(inputs_c)
                loss, components = criterion(outputs, targets_c)
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
            f"Epoch {epoch+1}/{epochs} | "
            f"Train: {avg_train:.6f} | Val: {avg_val:.6f} | "
            f"Time: {epoch_time:.1f}s | Session: {session_elapsed:.1f}h"
        )

        # ---- Checkpointing ----
        if (epoch + 1) % ckpt_every == 0:
            _save_checkpoint(
                model, optimizer, scheduler,
                epoch + 1, avg_val, best_val_loss,
                checkpoint_dir,
            )

        # ---- Best model ----
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            no_improve_count = 0
            _save_checkpoint(
                model, optimizer, scheduler,
                epoch + 1, avg_val, best_val_loss,
                checkpoint_dir, is_best=True,
            )
            logger.info(f"  ★ New best validation loss: {best_val_loss:.6f}")
        else:
            no_improve_count += 1

        # ---- Log to CSV ----
        _log_to_csv(csv_path, epoch + 1, avg_train, avg_val)

        # ---- Early stopping ----
        if no_improve_count >= patience:
            logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

        # ---- Session time warning ----
        if session_elapsed > 10.0:
            logger.warning(
                f"⚠ Session has been running for {session_elapsed:.1f}h. "
                "Colab Free may disconnect soon. Saving checkpoint..."
            )
            _save_checkpoint(
                model, optimizer, scheduler,
                epoch + 1, avg_val, best_val_loss,
                checkpoint_dir,
            )

    total_time = (time.time() - session_start) / 60
    logger.info(f"Training complete. Total time: {total_time:.1f} min")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")

    return {
        'history': history,
        'best_val_loss': best_val_loss,
        'final_epoch': epoch + 1,
        'total_time_min': total_time,
    }


def _save_checkpoint(
    model, optimizer, scheduler,
    epoch, val_loss, best_val_loss,
    checkpoint_dir, is_best=False,
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
    }

    path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch:04d}.pt"
    torch.save(checkpoint, str(path))
    logger.info(f"Saved checkpoint → {path}")

    if is_best:
        best_path = Path(checkpoint_dir) / "best_model.pt"
        torch.save(checkpoint, str(best_path))
        logger.info(f"Saved best model → {best_path}")


def _load_checkpoint(model, optimizer, scheduler, checkpoint_dir, device):
    """Load the latest checkpoint. Returns (start_epoch, best_val_loss)."""
    checkpoint_dir = Path(checkpoint_dir)

    # Try best model first, then latest checkpoint
    best_path = checkpoint_dir / "best_model.pt"
    if best_path.exists():
        ckpt = torch.load(str(best_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        logger.info(f"Resumed from best model at epoch {ckpt['epoch']}")
        return ckpt['epoch'], ckpt.get('best_val_loss', float('inf'))

    # Find latest numbered checkpoint
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if checkpoints:
        ckpt = torch.load(str(checkpoints[-1]), map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        logger.info(f"Resumed from {checkpoints[-1].name} at epoch {ckpt['epoch']}")
        return ckpt['epoch'], ckpt.get('best_val_loss', float('inf'))

    logger.info("No checkpoint found, starting from scratch")
    return 0, float('inf')


def _log_to_csv(csv_path, epoch, train_loss, val_loss):
    """Append training metrics to CSV log."""
    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['epoch', 'train_loss', 'val_loss'])
        writer.writerow([epoch, f'{train_loss:.6f}', f'{val_loss:.6f}'])
