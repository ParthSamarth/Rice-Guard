"""
classification/train_classifier.py

Trains the 6-class (Healthy + 5 diseases) CNN classifier via transfer
learning from ImageNet weights, on the combined full-image + YOLO-crop
dataset built by preprocessing/prepare_classification_dataset.py.

Implements (project brief section 7):
  - pretrained ImageNet weights, ImageNet normalization
  - the dataset's own train/val/test split (already fixed, seed 42)
  - class-distribution analysis -> class-weighted loss ONLY if the
    distribution is actually imbalanced beyond a configurable threshold
    (classification/dataset.py:compute_class_weights)
  - early stopping on validation macro-F1
  - ReduceLROnPlateau learning-rate scheduling
  - best-checkpoint saving (by validation macro-F1)
  - automatic batch size / AMP / worker count from utils.hardware

Usage:
    python classification/train_classifier.py --backbone resnet50
    python classification/train_classifier.py --backbone resnet18 --epochs 5   # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from classification.dataset import (  # noqa: E402
    ALL_CLASSES, CLASS_TO_IDX, RiceLeafClassificationDataset,
    build_transforms, compute_class_weights,
)
from classification.model import build_model, count_parameters, TARGET_LAYER_NAME  # noqa: E402
from utils.config import load_config, Config  # noqa: E402
from utils.hardware import detect_hardware, recommend_classifier_settings  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def evaluate(model, loader, criterion, device, amp: bool):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            n += images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(n, 1)
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels))) if n else 0.0
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if n else 0.0
    return avg_loss, acc, macro_f1, all_preds, all_labels


def train(backbone: str, config: Config, epochs_override: int | None = None,
          max_train_samples: int | None = None, run_tag: str | None = None,
          num_workers_override: int | None = None, resume_from: Path | None = None):
    set_seed(config["seed"])
    profile = detect_hardware()
    settings = recommend_classifier_settings(profile, backbone)
    if num_workers_override is not None:
        settings["num_workers"] = num_workers_override
    device = torch.device(settings["device"])
    print(f"Hardware: {profile.gpu_name or 'CPU'} | device={device} | settings={settings}")

    manifest_csv = config.path("paths", "classification_dataset") / "manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"{manifest_csv} not found. Run preprocessing/prepare_classification_dataset.py first.")

    image_size = settings["image_size"]
    train_ds = RiceLeafClassificationDataset(manifest_csv, "train", build_transforms(image_size, train=True))
    val_ds = RiceLeafClassificationDataset(manifest_csv, "val", build_transforms(image_size, train=False))
    test_ds = RiceLeafClassificationDataset(manifest_csv, "test", build_transforms(image_size, train=False))

    if max_train_samples is not None and max_train_samples < len(train_ds):
        # Smoke-test hook only: shrink to a fixed-seed random subset so the
        # training loop can be verified end-to-end in minutes, not hours.
        rng = np.random.default_rng(config["seed"])
        idx = rng.choice(len(train_ds), size=max_train_samples, replace=False)
        train_ds.df = train_ds.df.iloc[idx].reset_index(drop=True)

    print(f"Train/Val/Test samples: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    num_workers = settings["num_workers"]

    def build_loaders(bs: int):
        train_l = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"),
                              persistent_workers=(num_workers > 0), drop_last=True)
        val_l = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=num_workers, pin_memory=(device.type == "cuda"),
                            persistent_workers=(num_workers > 0))
        test_l = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=num_workers, pin_memory=(device.type == "cuda"),
                             persistent_workers=(num_workers > 0))
        return train_l, val_l, test_l

    batch_size = settings["batch_size"]
    train_loader, val_loader, test_loader = build_loaders(batch_size)

    class_counts = train_ds.class_counts()
    weights, weight_diag = compute_class_weights(class_counts)
    print(f"Class distribution (train): {weight_diag}")
    weights_t = weights.to(device) if weights is not None else None

    cls_cfg = config["classifier"]
    model = build_model(backbone=backbone, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_t, label_smoothing=cls_cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cls_cfg["lr"], weight_decay=cls_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    amp = settings["amp"]
    scaler = torch.amp.GradScaler(enabled=amp)

    max_epochs = epochs_override if epochs_override is not None else cls_cfg["max_epochs"]
    patience = cls_cfg["early_stopping_patience"]

    run_tag = run_tag or backbone
    models_dir = config.path("paths", "models") / f"classifier_{run_tag}"
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = config.path("paths", "outputs") / "classification" / run_tag
    outputs_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    history = []
    start_epoch = 1
    total_oom_events = 0     # cumulative, reported in the final summary (section 19)

    if resume_from is not None:
        # True resume (external-interruption recovery, mirrors
        # detection/train_yolo.py's --resume): restores model, optimizer,
        # and scheduler state plus every training-loop counter, so training
        # continues from the NEXT epoch rather than restarting the epoch
        # budget/early-stopping clock from 1. Falls back to a weights-only
        # warm start if this checkpoint predates these fields (e.g. one
        # saved before this resume mechanism existed).
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            best_val_f1 = ckpt["best_val_f1"]
            best_epoch = ckpt["best_epoch"]
            epochs_without_improvement = ckpt["epochs_without_improvement"]
            history = ckpt["history"]
            total_oom_events = ckpt.get("total_oom_events", 0)
            start_epoch = ckpt["epoch"] + 1
            print(f"Resumed FULL training state from {resume_from}: continuing at epoch {start_epoch} "
                  f"(best so far: epoch {best_epoch}, val_macro_f1={best_val_f1:.4f})")
        else:
            print(f"Resumed weights ONLY from {resume_from} (epoch {ckpt.get('epoch')}) -- "
                  f"no optimizer/scheduler state in this checkpoint, restarting epoch budget from 1.")
    param_info = count_parameters(model)
    print(f"Model: {backbone} | params={param_info}")

    t_start = time.time()
    oom_streak = 0           # consecutive OOMs at the CURRENT batch size
    OOM_STREAK_LIMIT = 3     # after this many in a row, shrink the batch size instead of keeps skipping
    MIN_BATCH_SIZE = 4

    if start_epoch > max_epochs:
        print(f"start_epoch={start_epoch} already exceeds max_epochs={max_epochs}; nothing left to train.")

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        t_epoch = time.time()
        running_loss, n_seen, n_correct = 0.0, 0, 0
        batch_size_changed_mid_epoch = False

        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                oom_streak = 0
            except torch.cuda.OutOfMemoryError:
                # Section 19: never silently eat an unbounded stream of OOMs.
                # The first couple are logged and skipped (transient memory
                # spikes happen); if they keep recurring at this batch size,
                # actually shrink the batch size and rebuild the loaders
                # rather than continuing to drop batches forever.
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                total_oom_events += 1
                oom_streak += 1
                print(f"  WARNING: CUDA OOM #{total_oom_events} on a batch of size {images.size(0)} "
                      f"(streak {oom_streak}/{OOM_STREAK_LIMIT} at batch_size={batch_size}) -- skipped it.")

                if oom_streak >= OOM_STREAK_LIMIT and batch_size > MIN_BATCH_SIZE:
                    new_batch_size = max(MIN_BATCH_SIZE, batch_size // 2)
                    print(f"  {oom_streak} consecutive OOMs at batch_size={batch_size} -- "
                          f"reducing to batch_size={new_batch_size} and rebuilding data loaders.")
                    batch_size = new_batch_size
                    train_loader, val_loader, test_loader = build_loaders(batch_size)
                    oom_streak = 0
                    batch_size_changed_mid_epoch = True
                    break  # this epoch's remaining batches came from the now-stale loader
                continue

            running_loss += loss.item() * images.size(0)
            n_seen += images.size(0)
            n_correct += (outputs.argmax(dim=1) == labels).sum().item()

        if batch_size_changed_mid_epoch:
            # This epoch's train_loss/train_acc below reflect only the batches
            # seen before the batch-size shrink (a partial pass), not a full
            # epoch -- accepted as a minor simplification; validation still
            # runs and this epoch is still checkpointed normally, and epoch
            # N+1 onward uses the new, smaller batch_size for a full pass.
            print(f"  Epoch {epoch} continuing with the reduced batch_size={batch_size} "
                  f"for its validation pass ({n_seen} training samples seen before the resize).")

        train_loss = running_loss / max(n_seen, 1)
        train_acc = n_correct / max(n_seen, 1)
        val_loss, val_acc, val_macro_f1, _, _ = evaluate(model, val_loader, criterion, device, amp)
        scheduler.step(val_macro_f1)

        epoch_time = time.time() - t_epoch
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[{run_tag}] epoch {epoch}/{max_epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f} | "
              f"lr={current_lr:.2e} | {epoch_time:.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": val_macro_f1,
            "lr": current_lr, "epoch_time_sec": round(epoch_time, 1),
        })

        improved = val_macro_f1 > best_val_f1
        if improved:
            best_val_f1 = val_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "backbone": backbone,
                "class_to_idx": CLASS_TO_IDX,
                "epoch": epoch,
                "val_macro_f1": val_macro_f1,
                "image_size": image_size,
            }, models_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        # last.pt carries the FULL resumable state (optimizer/scheduler/
        # counters), not just weights -- this is what --resume-style
        # interruption recovery reloads (see `resume_from` above). best.pt
        # stays weights-only/lean since evaluate_classifier.py and
        # predict_classifier.py only ever need inference from it.
        torch.save({
            "model_state_dict": model.state_dict(), "backbone": backbone,
            "class_to_idx": CLASS_TO_IDX, "epoch": epoch, "image_size": image_size,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_f1": best_val_f1, "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history, "total_oom_events": total_oom_events,
        }, models_dir / "last.pt")

        if epochs_without_improvement >= patience:
            print(f"Early stopping: no val_macro_f1 improvement for {patience} epochs "
                  f"(best={best_val_f1:.4f} at epoch {best_epoch}).")
            break

    total_time = time.time() - t_start

    # Final test-set evaluation using the BEST checkpoint (never the last).
    best_ckpt = torch.load(models_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_acc, test_macro_f1, test_preds, test_labels = evaluate(model, test_loader, criterion, device, amp)
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, test_preds, labels=list(range(len(ALL_CLASSES))), zero_division=0
    )
    per_class_test = {
        ALL_CLASSES[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                          "f1": float(f1[i]), "support": int(support[i])}
        for i in range(len(ALL_CLASSES))
    }
    print(f"\n[{run_tag}] TEST (best checkpoint, epoch {best_ckpt['epoch']}): "
          f"loss={test_loss:.4f} acc={test_acc:.4f} macro_f1={test_macro_f1:.4f}")

    if total_oom_events:
        print(f"Total CUDA OOM events during training: {total_oom_events} "
              f"(final batch_size={batch_size}, started at {settings['batch_size']})")

    summary = {
        "backbone": backbone, "run_tag": run_tag,
        "hardware": profile.to_dict(), "settings": settings,
        "param_info": param_info,
        "class_weight_diagnostics": weight_diag,
        "total_oom_events": total_oom_events,
        "initial_batch_size": settings["batch_size"], "final_batch_size": batch_size,
        "train_samples": len(train_ds), "val_samples": len(val_ds), "test_samples": len(test_ds),
        "epochs_trained": len(history), "best_epoch": best_epoch, "best_val_macro_f1": best_val_f1,
        "total_training_time_sec": round(total_time, 1),
        "test_loss": test_loss, "test_acc": test_acc, "test_macro_f1": test_macro_f1,
        "test_per_class": per_class_test,
        "history": history,
        "target_layer_for_gradcam": TARGET_LAYER_NAME[backbone],
        "checkpoint_best": str(models_dir / "best.pt"),
    }
    with open(outputs_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {outputs_dir / 'train_summary.json'}")
    print(f"Wrote {models_dir / 'best.pt'}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train the rice-leaf CNN classifier.")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet18", "resnet50"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override configs/config.yaml classifier.max_epochs")
    parser.add_argument("--max-train-samples", type=int, default=None,
                         help="Smoke-test only: cap the training set to N samples")
    parser.add_argument("--run-tag", type=str, default=None, help="Output subfolder name (defaults to --backbone)")
    parser.add_argument("--resume", action="store_true",
                         help="Resume an interrupted run from models/classifier_<run-tag>/last.pt (full "
                              "optimizer/scheduler/epoch-counter state, not just weights -- does NOT restart "
                              "the epoch budget or early-stopping clock from scratch).")
    parser.add_argument("--workers", type=int, default=None,
                         help="Override the DataLoader worker count (e.g. 0 for single-process loading, "
                              "the standard mitigation for repeated 'DataLoader worker exited unexpectedly' "
                              "crashes -- see detection/train_yolo.py for the same issue on this machine).")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    run_tag = args.run_tag or args.backbone
    last_ckpt = config.path("paths", "models") / f"classifier_{run_tag}" / "last.pt"
    resume_from = last_ckpt if (args.resume and last_ckpt.exists()) else None
    if args.resume and resume_from is None:
        print(f"WARNING: --resume given but no checkpoint at {last_ckpt} -- starting fresh instead.")

    try:
        train(args.backbone, config, epochs_override=args.epochs,
              max_train_samples=args.max_train_samples, run_tag=run_tag,
              resume_from=resume_from, num_workers_override=args.workers)
    except RuntimeError as exc:
        if "DataLoader worker" not in str(exc) and "exited unexpectedly" not in str(exc):
            raise
        # See detection/train_yolo.py for the same Windows spawn-multiprocessing
        # mitigation. Retry once, single-process, resuming FULL state from
        # last.pt if this run got far enough to write one.
        print(f"WARNING: {exc}\nRetrying with num_workers=0 (single-process data loading)...")
        train(args.backbone, config, epochs_override=args.epochs,
              max_train_samples=args.max_train_samples, run_tag=run_tag,
              num_workers_override=0, resume_from=last_ckpt if last_ckpt.exists() else None)


if __name__ == "__main__":
    main()
