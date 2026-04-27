#!/usr/bin/env python3
"""
Environment sanity check — run before experiments to verify setup.
"""
import platform

import torch
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlInit

print("=" * 50)
print("ENVIRONMENT CHECK")
print("=" * 50)

print(f"Python:         {platform.python_version()}")
print(f"Torch:          {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version:   {torch.version.cuda}")

assert torch.cuda.is_available(), "CUDA not available — wrong environment"

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
gpu_name = nvmlDeviceGetName(handle)
print(f"GPU:            {gpu_name}")

import vllm  # noqa: E402 — intentionally after basic checks

vllm_version = getattr(vllm, "__version__", "(dev/editable install)")
print(f"vLLM:           {vllm_version}")
print()
print("✓ All checks passed — environment is ready")
print("=" * 50)
