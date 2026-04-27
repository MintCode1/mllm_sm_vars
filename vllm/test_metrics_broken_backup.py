#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Controlled experiment script for vLLM metrics with GPU power measurement and SM masking.

Runs a strict closed-loop experiment:
- 10 requests total
- Batch size = 1 (one request at a time)
- Each request: ~512 input tokens, 10 output tokens
- Next request starts ONLY after previous finishes
- Measures GPU power consumption
- Applies SM/TPC masking for controlled resource availability
- Logs results to CSV
"""

import csv
import gc
import os
import subprocess
from typing import Iterable

import torch
# Must be set BEFORE vLLM is imported.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
os.environ["VLLM_USE_DEEP_GEMM"] = "0"

assert os.path.exists("test_image.png"), "Missing test image"

from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName,
    nvmlDeviceGetPowerUsage,
    nvmlInit,
)
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.v1.libsmctrl import build_sm_mask

# Experiment controls
DEBUG_MASK = False
RESUME = False
WARMUP_REQUESTS = 3
REQUESTS = 10
SMOKE_SINGLE_REQUEST = True

# H100 config used by the masking math.
TOTAL_SMS = 132
TOTAL_TPCS = 66

# Default full sweep for final data collection.
SM_SWEEP = [
    108,
    100,
    92,
    84,
    76,
    68,
    60,
    52,
    44,
    40,
    36,
    32,
    28,
    24,
    20,
    16,
    12,
    8,
    4,
]

MODEL_CANDIDATES = [
    "OpenGVLab/InternVL3-9B",
    "OpenGVLab/InternVL3-8B",
]

# Global power sampling
power_samples = []
handle = None


def create_long_prompt():
    """Create a prompt ~512 tokens long."""
    base = "Explain the theory of relativity in detail. "
    return base * 40


def _resolve_sm_sweep(default_sweep: list[int]) -> list[int]:
    raw = os.environ.get("SM_SWEEP_OVERRIDE", "").strip()
    if not raw:
        return default_sweep
    try:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(
            "SM_SWEEP_OVERRIDE must be a comma-separated list of integers"
        ) from exc
    if not values:
        raise ValueError("SM_SWEEP_OVERRIDE did not contain any integer values")
    return values


def _iter_models(models: Iterable[str]) -> Iterable[str]:
    for model_name in models:
        yield model_name


def create_llm_with_fallback() -> tuple[LLM, str]:
    last_exc: Exception | None = None
    for model_name in _iter_models(MODEL_CANDIDATES):
        try:
            llm = LLM(
                model=model_name,
                trust_remote_code=True,
                max_model_len=1024,
                gpu_memory_utilization=0.9,
                enforce_eager=True,
                max_num_seqs=1,
            )
            return llm, model_name
        except Exception as exc:
            print(f"Model init failed for {model_name}: {exc}")
            last_exc = exc

    raise RuntimeError("Failed to initialize any candidate model") from last_exc


def configure_stream_mask(active_sms: int) -> tuple[int, int, str]:
    if active_sms >= TOTAL_SMS:
        os.environ["VLLM_STREAM_MASK"] = "0"
        return TOTAL_TPCS, 0, "0x0"

    active_tpcs = max(1, active_sms // 2)
    disabled_tpcs = list(range(active_tpcs, TOTAL_TPCS))
    mask = build_sm_mask(TOTAL_TPCS, disabled_tpcs)
    os.environ["VLLM_STREAM_MASK"] = str(mask)
    return active_tpcs, mask, hex(mask)


def main():
    global handle

    print(">>> Starting script")
    mode_label = "vLLM stream_mask"
    sm_sweep = _resolve_sm_sweep(SM_SWEEP)
    if SMOKE_SINGLE_REQUEST:
        sm_sweep = [TOTAL_SMS]

    if not RESUME and os.path.exists("results.csv"):
        os.remove("results.csv")

    print("=" * 60)
    print(f"CLOSED-LOOP METRICS EXPERIMENT SWEEP — {mode_label}")
    print("=" * 60)

    # Initialize NVML for GPU power measurement
    print("\nInitializing NVML...")
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    print(f"GPU: {nvmlDeviceGetName(handle)}")
    print(f"Detected SMs: {torch.cuda.get_device_properties(0).multi_processor_count}")

    # Create long prompt (~512 tokens)
    long_prompt = create_long_prompt()
    test_image = Image.open("test_image.png").convert("RGB")
    request_payload = {
        "prompt": f"<image>\n{long_prompt}",
        "multi_modal_data": {
            "image": test_image,
        },
    }

    # Configure sampling
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    csv_exists = os.path.exists("results.csv")
    if not csv_exists:
        with open("results.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "active_sms",
                "active_tpcs",
                "mask_hex",
                "avg_latency",
                "throughput_tok_per_s",
                "encoder_avg",
                "prefill_avg",
                "decode_avg",
                "avg_power",
            ])

    print(f"\nSweep parameters:")
    print(f"  - SM values: {sm_sweep}")
    print(f"  - Total SMs (mask model): {TOTAL_SMS}")
    print(f"  - Total TPCs: {TOTAL_TPCS}")
    print(f"  - Batch size: 1 (closed-loop)")
    print(f"  - Warmup requests per run: {WARMUP_REQUESTS}")
    print(f"  - Number of measured requests per run: {REQUESTS}")
    print(f"  - Input tokens (approx): 512")
    print(f"  - Output tokens: 10")

    for active_sms in sm_sweep:
        print(f"\n=== Running with {active_sms} SMs ===")
        active_tpcs, mask, mask_hex = configure_stream_mask(active_sms)
        print(
            f"Mask config: active_tpcs={active_tpcs}, "
            f"VLLM_STREAM_MASK={os.environ.get('VLLM_STREAM_MASK')} ({mask_hex})"
        )

        print(">>> About to initialize LLM")
        print("Initializing vLLM...")
        llm, chosen_model = create_llm_with_fallback()
        print(">>> LLM initialized")
        print(f"Using model: {chosen_model}")

        # Warmup requests (excluded from metrics/power aggregation)
        print(f"  Warmup: {WARMUP_REQUESTS} requests...")
        for _ in range(WARMUP_REQUESTS):
            llm.generate([request_payload], sampling_params)

        engine_core_client = llm.llm_engine.engine_core

        def call_metric_rpc(method: str):
            return engine_core_client.collective_rpc(
                method=method,
                args=(),
                kwargs=None,
            )[0]

        call_metric_rpc("reset_metrics")
        call_metric_rpc("start_metrics_run")
        power_samples.clear()

        print(">>> About to run first generate")
        if SMOKE_SINGLE_REQUEST:
            llm.generate([request_payload], sampling_params)
            power_samples.append(nvmlDeviceGetPowerUsage(handle) / 1000.0)
        else:
            # Run measured requests in strict closed-loop (one at a time).
            for _ in range(REQUESTS):
                llm.generate([request_payload], sampling_params)
                if DEBUG_MASK:
                    print(f"Mask touches: {call_metric_rpc('get_stream_mask_touch_count')}")

                power_samples.append(nvmlDeviceGetPowerUsage(handle) / 1000.0)

        call_metric_rpc("end_metrics_run")

        avg_power = sum(power_samples) / len(power_samples)
        summary = call_metric_rpc("get_metrics_summary")
        util = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        ).decode().strip()

        row = [
            active_sms,
            active_tpcs,
            mask_hex,
            summary["avg_latency"],
            summary["throughput_tok_per_s"],
            summary["encoder_avg"],
            summary["prefill_avg"],
            summary["decode_avg"],
            avg_power,
        ]

        print("\n" + "-" * 60)
        print(f"RUN SUMMARY ({active_sms} SMs)")
        print("-" * 60)
        print(f"avg_latency: {summary['avg_latency']}")
        print(f"throughput_tok_per_s: {summary['throughput_tok_per_s']}")
        print(f"encoder_avg: {summary['encoder_avg']}")
        print(f"prefill_avg: {summary['prefill_avg']}")
        print(f"decode_avg: {summary['decode_avg']}")
        print(f"avg_power (watts): {avg_power}")
        print(f"GPU Utilization: {util}%")
        print(
            f"SM={active_sms}, latency={summary['avg_latency']}, "
            f"throughput={summary['throughput_tok_per_s']}, power={avg_power}"
        )

        with open("results.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(f"Saved row: {row}")

        # Release engine resources before creating the next LLM instance.
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
