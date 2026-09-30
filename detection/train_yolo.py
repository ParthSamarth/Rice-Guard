"""
detection/train_yolo.py

Trains YOLOv8 as a 5-class DISEASE detector (Blast, Brown Spot, Leaf Smut,
Rice Tungro, Sheath Blight -- Healthy is architecturally excluded, see
preprocessing/prepare_yolo_dataset.py). Thin wrapper around Ultralytics'
own trainer, which already provides transfer learning from COCO-pretrained
weights, built-in mosaic/HSV/flip augmentation, per-epoch validation,
last.pt/best.pt checkpointing, and early stopping via `patience`.

Usage:
    python detection/train_yolo.py --model yolov8n.pt
    python detection/train_yolo.py --model yolov8s.pt
    python detection/train_yolo.py --model yolov8n.pt --epochs 3   # smoke test
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config, Config  # noqa: E402
from utils.hardware import detect_hardware, recommend_yolo_settings  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def train(model_name: str, config: Config, epochs_override: int | None = None, run_tag: str | None = None,
          resume: bool = False, imgsz_override: int | None = None, patience_override: int | None = None):
    from ultralytics import YOLO  # imported lazily so --help doesn't require torch/ultralytics installed

    set_seed(config["seed"])
    profile = detect_hardware()
    settings = recommend_yolo_settings(profile)
    if imgsz_override is not None:
        settings = {**settings, "imgsz": imgsz_override}
    print(f"Hardware: {profile.gpu_name or 'CPU'} | cuda={profile.cuda_available} | settings={settings}")

    data_yaml = Path(__file__).resolve().parent.parent / "configs" / "yolo_data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found. Run preprocessing/prepare_yolo_dataset.py first.")

    run_tag = run_tag or Path(model_name).stem
    project_dir = config.path("paths", "outputs") / "yolo" / "runs"
    epochs = epochs_override if epochs_override is not None else config["yolo"]["epochs"]

    last_ckpt = project_dir / run_tag / "weights" / "last.pt"
    if resume:
        if not last_ckpt.exists():
            raise FileNotFoundError(f"--resume given but no checkpoint at {last_ckpt}")
        # Ultralytics reads the rest of the original training config (imgsz,
        # batch, patience, etc.) back out of the checkpoint's own train_args
        # when resume=True -- passing our own settings again would be
        # ignored/conflicting. `epochs` is the one exception: Ultralytics
        # explicitly supports overriding the total epoch count on a resumed
        # run (e.g. to cap a run shorter than its original budget), so it's
        # passed through when the caller gave one.
        resume_kwargs = {"resume": True}
        if epochs_override is not None:
            resume_kwargs["epochs"] = epochs_override
            print(f"Resuming from {last_ckpt} with epoch budget capped at {epochs_override} "
                  f"(external interruption recovery -- preserves all prior epochs, optimizer/EMA state, "
                  f"and best-fitness tracking; does NOT restart from scratch).")
        else:
            print(f"Resuming from {last_ckpt} (external interruption recovery -- preserves all prior epochs, "
                  f"optimizer/EMA state, and best-fitness tracking; does NOT restart from scratch).")
        model = YOLO(str(last_ckpt))
        t0 = time.time()
        try:
            model.train(**resume_kwargs)
        except RuntimeError as exc:
            if "DataLoader worker" not in str(exc) and "exited unexpectedly" not in str(exc):
                raise
            # Same protection as the fresh-training path below -- this bug
            # (missing here originally) let one resume attempt crash
            # uncaught instead of retrying single-process. Re-resume from
            # whatever last.pt exists NOW (the crashed attempt may itself
            # have completed and saved further epochs before dying).
            print(f"WARNING: {exc}\nRetrying resume with workers=0 (single-process data loading)...")
            model = YOLO(str(last_ckpt))
            resume_kwargs["workers"] = 0
            model.train(**resume_kwargs)
        total_time = time.time() - t0
        run_dir = Path(model.trainer.save_dir)
        return _finalize(model_name, config, run_tag, profile, settings, epochs, total_time, run_dir)

    patience = patience_override if patience_override is not None else config["yolo"]["patience"]
    settings = {**settings, "patience": patience}
    model = YOLO(model_name)
    t0 = time.time()
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=settings["imgsz"],
        batch=settings["batch"],
        device=settings["device"],
        workers=settings["workers"],
        amp=settings["amp"],
        patience=patience,
        seed=config["seed"],
        project=str(project_dir),
        name=run_tag,
        exist_ok=True,
        pretrained=True,
        plots=True,
        save=True,
        val=True,
        verbose=True,
    )
    try:
        model.train(**train_kwargs)
    except RuntimeError as exc:
        if "DataLoader worker" not in str(exc) and "exited unexpectedly" not in str(exc):
            raise
        # Windows spawn-multiprocessing DataLoader workers can die under RAM/
        # process pressure independent of anything wrong with the data itself
        # (see utils/hardware.py:_safe_workers). Retry once, single-process,
        # resuming from the last checkpoint if this run got far enough to
        # write one, rather than losing all prior progress.
        print(f"WARNING: {exc}\nRetrying with workers=0 (single-process data loading)...")
        if last_ckpt.exists():
            model = YOLO(str(last_ckpt))
            train_kwargs["resume"] = True
        train_kwargs["workers"] = 0
        model.train(**train_kwargs)
    total_time = time.time() - t0

    run_dir = Path(model.trainer.save_dir)
    return _finalize(model_name, config, run_tag, profile, settings, epochs, total_time, run_dir)


def _finalize(model_name: str, config: Config, run_tag: str, profile, settings: dict,
              epochs: int, total_time: float, run_dir: Path) -> dict:
    """Stage weights/metrics/plots to their stable, run-independent output
    paths and write the training summary JSON. Shared by both the normal
    training path and the --resume path (detection/train_yolo.py:train()).
    """
    best_weights = run_dir / "weights" / "best.pt"

    # Also stage a copy under models/ and outputs/yolo/{weights,metrics,plots}
    # so the rest of the pipeline (pipeline/inference_pipeline.py,
    # evaluation/) has one stable, run-independent path to load from,
    # regardless of Ultralytics' own auto-incrementing run-name folders.
    models_dir = config.path("paths", "models") / f"yolo_{run_tag}"
    models_dir.mkdir(parents=True, exist_ok=True)
    if best_weights.exists():
        shutil.copy2(best_weights, models_dir / "best.pt")

    out_weights = config.path("paths", "outputs") / "yolo" / "weights"
    out_metrics = config.path("paths", "outputs") / "yolo" / "metrics"
    out_plots = config.path("paths", "outputs") / "yolo" / "plots"
    for d in (out_weights, out_metrics, out_plots):
        d.mkdir(parents=True, exist_ok=True)
    if best_weights.exists():
        shutil.copy2(best_weights, out_weights / f"{run_tag}_best.pt")

    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        shutil.copy2(results_csv, out_metrics / f"{run_tag}_results.csv")

    for png in run_dir.glob("*.png"):
        try:
            shutil.copy2(png, out_plots / f"{run_tag}_{png.name}")
        except OSError:
            pass

    summary = {
        "model": model_name,
        "run_tag": run_tag,
        "hardware": profile.to_dict(),
        "settings": settings,
        "epochs_requested": epochs,
        "total_training_time_sec": round(total_time, 1),
        "run_dir": str(run_dir),
        "best_weights": str(models_dir / "best.pt") if best_weights.exists() else None,
        "results_csv": str(out_metrics / f"{run_tag}_results.csv") if results_csv.exists() else None,
    }
    summary_path = out_metrics / f"{run_tag}_train_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTraining complete in {total_time:.1f}s. Best weights: {summary['best_weights']}")
    print(f"Wrote {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 as a rice-disease detector.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="yolov8n.pt, yolov8s.pt, etc.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--resume", action="store_true",
                         help="Resume an interrupted run from outputs/yolo/runs/<run-tag>/weights/last.pt "
                              "(preserves all completed epochs, optimizer/EMA state, and best-fitness "
                              "tracking -- this is NOT a restart from scratch).")
    parser.add_argument("--imgsz", type=int, default=None,
                         help="Override the hardware-tier-recommended input resolution (e.g. for a "
                              "resolution ablation). Leaves utils/hardware.py's default untouched for "
                              "every other run.")
    parser.add_argument("--patience", type=int, default=None,
                         help="Override configs/config.yaml's yolo.patience for this run only.")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    train(args.model, config, epochs_override=args.epochs, run_tag=args.run_tag, resume=args.resume,
          imgsz_override=args.imgsz, patience_override=args.patience)


if __name__ == "__main__":
    main()
