"""
Hardware detection + conservative default selection (batch size, image size,
worker count, mixed precision) so training scripts don't have to guess and
don't CUDA-OOM on modest GPUs (this project was developed against a 6 GB
laptop GPU -- see outputs/experiment_summary.json for the actual profile
detected on the training machine).

Import and call `detect_hardware()` once per script; feed the result to
`recommend_yolo_settings` / `recommend_classifier_settings`.  Every
recommendation can be overridden from configs/config.yaml or the CLI --
these are defaults, not hard limits.
"""

from __future__ import annotations

import multiprocessing
import os
import platform
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class HardwareProfile:
    platform: str
    python_version: str
    torch_version: Optional[str]
    torchvision_version: Optional[str]
    ultralytics_version: Optional[str]
    cuda_available: bool
    cuda_version: Optional[str]
    gpu_name: Optional[str]
    total_vram_gb: Optional[float]
    cpu_count: int
    ram_gb: Optional[float]
    device: str  # "cuda:0" or "cpu"

    def to_dict(self) -> dict:
        return asdict(self)


def _ram_gb() -> Optional[float]:
    try:
        import psutil  # optional dependency; not required

        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except ImportError:
        pass
    # Fallback for Windows without psutil.
    try:
        if platform.system() == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / (1024 ** 3), 2)
    except Exception:
        pass
    return None


def detect_hardware() -> HardwareProfile:
    cpu_count = multiprocessing.cpu_count()
    ram_gb = _ram_gb()

    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else None
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            total_vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
            device = "cuda:0"
        else:
            gpu_name = None
            total_vram_gb = None
            device = "cpu"
    except ImportError:
        torch_version = None
        cuda_available = False
        cuda_version = None
        gpu_name = None
        total_vram_gb = None
        device = "cpu"

    try:
        import torchvision
        torchvision_version = torchvision.__version__
    except ImportError:
        torchvision_version = None

    try:
        import ultralytics
        ultralytics_version = ultralytics.__version__
    except ImportError:
        ultralytics_version = None

    return HardwareProfile(
        platform=platform.platform(),
        python_version=platform.python_version(),
        torch_version=torch_version,
        torchvision_version=torchvision_version,
        ultralytics_version=ultralytics_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        total_vram_gb=total_vram_gb,
        cpu_count=cpu_count,
        ram_gb=ram_gb,
        device=device,
    )


def _safe_workers(profile: HardwareProfile, cap: int = 4) -> int:
    # Windows DataLoader workers use the (slow) 'spawn' start method and each
    # worker re-imports the launching script, so we deliberately stay well
    # under cpu_count to avoid startup thrashing / RAM blowup, especially
    # since this project's classification dataset generates many small crops.
    # cap=4 (down from an earlier 6) after an actual training run on this
    # project's 16 GB / 6 GB-VRAM machine hit "DataLoader worker exited
    # unexpectedly" partway through epoch 1 -- a well-known Windows
    # spawn-multiprocessing failure mode under RAM/process pressure; fewer
    # workers is the standard mitigation. detection/train_yolo.py also
    # retries once with workers=0 if this happens again.
    return max(0, min(cap, profile.cpu_count // 2))


def recommend_yolo_settings(profile: HardwareProfile) -> dict:
    if not profile.cuda_available:
        return {
            "device": "cpu",
            "batch": 8,
            "imgsz": 416,
            "workers": _safe_workers(profile, cap=4),
            "amp": False,
            "note": "No CUDA GPU detected -- training YOLO on CPU will be slow. "
                    "Reduce epochs or use a subset for experimentation.",
        }

    vram = profile.total_vram_gb or 0
    if vram <= 4:
        imgsz = 512
    elif vram <= 16:
        imgsz = 640
    else:
        imgsz = 640

    # Use Ultralytics' own AutoBatch (batch=-1) instead of a static number:
    # it probes ACTUAL free VRAM at training start and targets ~60% memory
    # utilization, which is far more robust against OOM than any fixed
    # constant -- a smoke test on this project's own 6 GB laptop GPU hit a
    # CUDA allocator OOM warning at a hand-picked batch=16, which AutoBatch
    # is specifically designed to avoid (project brief section 19).
    return {
        "device": 0,
        "batch": -1,
        "imgsz": imgsz,
        "workers": _safe_workers(profile),
        "amp": True,
        "note": f"Selected for {profile.gpu_name} ({vram} GB VRAM); batch=-1 triggers Ultralytics AutoBatch.",
    }


def recommend_classifier_settings(profile: HardwareProfile, model_name: str = "resnet50") -> dict:
    if not profile.cuda_available:
        return {
            "device": "cpu",
            "batch_size": 16,
            "image_size": 224,
            "num_workers": _safe_workers(profile, cap=4),
            "amp": False,
            "note": "No CUDA GPU detected -- training on CPU will be slow.",
        }

    vram = profile.total_vram_gb or 0
    heavy = model_name.lower() in {"resnet50", "resnet101", "resnet152"}
    if vram <= 4:
        batch = 16 if heavy else 32
    elif vram <= 8:
        batch = 32 if heavy else 64
    elif vram <= 16:
        batch = 64 if heavy else 128
    else:
        batch = 128 if heavy else 256

    return {
        "device": "cuda:0",
        "batch_size": batch,
        "image_size": 224,
        "num_workers": _safe_workers(profile),
        "amp": True,
        "note": f"Selected for {model_name} on {profile.gpu_name} ({vram} GB VRAM).",
    }


if __name__ == "__main__":
    import json

    profile = detect_hardware()
    print(json.dumps(profile.to_dict(), indent=2))
    print("YOLO settings:", json.dumps(recommend_yolo_settings(profile), indent=2))
    print("Classifier (resnet50) settings:", json.dumps(recommend_classifier_settings(profile, "resnet50"), indent=2))
    print("Classifier (resnet18) settings:", json.dumps(recommend_classifier_settings(profile, "resnet18"), indent=2))
