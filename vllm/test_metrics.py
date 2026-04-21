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
import ctypes
import time

from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlInit
from vllm import LLM, SamplingParams

# TPC count for this run
TPC_COUNT = 54  # Run with all TPCs (54 on typical H100)

# Total TPCs on the GPU (adjust if needed)
TOTAL_TPCS = 54

# Global power sampling
power_samples = []
handle = None

# SM masking library
lib = None


def load_smctrl_library():
    """Attempt to load the libsmctrl.so library for SM masking."""
    global lib
    try:
        lib = ctypes.CDLL("libsmctrl.so")
        print("✓ libsmctrl.so loaded successfully")
        return True
    except OSError as e:
        print(f"⚠ Warning: Could not load libsmctrl.so: {e}")
        print("  SM masking will be disabled for this run")
        lib = None
        return False


def create_tpc_mask(total_tpcs, active_tpcs):
    """
    Create a TPC mask to disable TPCs beyond active_tpcs.
    
    Args:
        total_tpcs: Total number of TPCs on the GPU
        active_tpcs: Number of TPCs to keep active
        
    Returns:
        A 64-bit mask where disabled TPCs have bit=1, enabled TPCs have bit=0
    """
    mask = 0
    for i in range(active_tpcs, total_tpcs):
        mask |= (1 << i)
    return mask


def set_tpc_mask(active_tpcs):
    """Set the global TPC mask to restrict GPU execution."""
    if lib is None:
        print(f"⚠ SM masking not available, running with default GPU configuration")
        return False

    try:
        mask = create_tpc_mask(TOTAL_TPCS, active_tpcs)
        lib.libsmctrl_set_global_mask(mask)
        print(f"✓ Applied TPC mask: {active_tpcs}/{TOTAL_TPCS} TPCs active (mask={hex(mask)})")
        return True
    except Exception as e:
        print(f"⚠ Error setting TPC mask: {e}")
        return False


def sample_power():
    """Sample current GPU power usage in watts."""
    power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
    power_samples.append(power)


def create_long_prompt():
    """Create a prompt ~512 tokens long."""
    base = "Explain the theory of relativity in detail. "
    return base * 40


def main():
    global handle

    print("=" * 60)
    print("CLOSED-LOOP METRICS EXPERIMENT WITH SM MASKING")
    print("=" * 60)

    # Initialize NVML for GPU power measurement
    print("\nInitializing NVML...")
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)

    # Load SM masking library
    print("Loading SM masking library...")
    load_smctrl_library()

    # Apply TPC mask
    print(f"\nApplying SM/TPC mask...")
    set_tpc_mask(TPC_COUNT)

    # Initialize model
    print("Initializing vLLM...")
    llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)

    # Create long prompt (~512 tokens)
    long_prompt = create_long_prompt()

    # Configure sampling
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    # Get metrics tracker
    metrics = llm.llm_engine.model_executor.driver_worker.model_runner.metrics
    metrics.reset()

    print(f"\nExperiment parameters:")
    print(f"  - TPC count: {TPC_COUNT}/{TOTAL_TPCS}")
    print(f"  - Batch size: 1 (closed-loop)")
    print(f"  - Number of requests: 10")
    print(f"  - Input tokens (approx): 512")
    print(f"  - Output tokens: 10")
    print(f"\nStarting experiment...")

    # Clear power samples for this run
    power_samples.clear()

    # Start metrics tracking
    metrics.start_run()

    # Run 10 requests in strict closed-loop (one at a time)
    for i in range(10):
        print(f"  Request {i + 1}/10...")
        sample_power()
        llm.generate([long_prompt], sampling_params)

    # End metrics tracking
    metrics.end_run()

    # Compute average power
    avg_power = sum(power_samples) / len(power_samples) if power_samples else 0.0

    # Get metrics summary
    summary = metrics.summary()

    # Prepare result row
    row = [
        TPC_COUNT,
        summary["avg_latency"],
        summary["throughput_tok_per_s"],
        summary["encoder_avg"],
        summary["prefill_avg"],
        summary["decode_avg"],
        avg_power,
    ]

    # Print summary
    print("\n" + "=" * 60)
    print("FINAL METRICS")
    print("=" * 60)
    print()
    print(f"avg_latency: {summary['avg_latency']}")
    print(f"throughput_tok_per_s: {summary['throughput_tok_per_s']}")
    print(f"encoder_avg: {summary['encoder_avg']}")
    print(f"prefill_avg: {summary['prefill_avg']}")
    print(f"decode_avg: {summary['decode_avg']}")
    print(f"avg_power (watts): {avg_power}")
    print()

    # Write to CSV
    print("Writing results to CSV...")
    with open("results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print(f"Saved row: {row}")
    print("=" * 60)


if __name__ == "__main__":
    main()
