#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run original SM-variation experiment with phase-level metrics.

This script is intentionally standalone and does not modify existing experiment
scripts. It enforces:
- Closed-loop arrival
- 10 requests total
- Batch size 1
- 512 input tokens
- 10 output tokens
- Static SM/TPC mask per run
- Extreme validation (full vs very low SM) before data collection

It writes one CSV row per SM/TPC setting and a validation sidecar JSON used by
the plotting script.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# Must be set before importing vLLM.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

from PIL import Image
from pynvml import (  # type: ignore
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName,
    nvmlDeviceGetPowerUsage,
    nvmlInit,
)
from vllm import LLM, SamplingParams
from vllm.v1.libsmctrl import build_sm_mask

TOTAL_SMS = 132
TOTAL_TPCS = 66

DEFAULT_SWEEP = [108, 92, 76, 60, 44, 32, 24, 16, 8, 4]

REQUESTS = 10
WARMUP_REQUESTS = 2
INPUT_TOKENS = 512
OUTPUT_TOKENS = 10
POWER_SAMPLE_PERIOD_S = 0.005

CSV_FIELDS = [
    "active_sms",
    "active_tpcs",
    "encoder_latency",
    "prefill_latency",
    "decode_latency",
    "encoder_throughput",
    "prefill_throughput",
    "decode_throughput",
    "encoder_power",
    "prefill_power",
    "decode_power",
    "total_latency",
    "total_throughput",
    "total_power",
]


@dataclass
class PhaseEvent:
    phase: str
    start_ns: int
    end_ns: int

    @property
    def duration_s(self) -> float:
        return max(0, self.end_ns - self.start_ns) / 1e9


@dataclass
class PowerSample:
    ts_ns: int
    power_w: float


_ACTIVE_EVENTS: list[PhaseEvent] | None = None
_EVENT_LOCK = threading.Lock()
_HOOKS_INSTALLED = False


def _record_event(phase: str, start_ns: int, end_ns: int) -> None:
    global _ACTIVE_EVENTS
    with _EVENT_LOCK:
        if _ACTIVE_EVENTS is not None:
            _ACTIVE_EVENTS.append(PhaseEvent(phase=phase, start_ns=start_ns, end_ns=end_ns))


def install_phase_hooks() -> None:
    """Install lightweight runtime hooks for encoder/prefill/decode boundaries."""
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return

    # V2 model runner encoder hook.
    from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner

    if not hasattr(EncoderRunner, "_smphase_orig_execute_mm_encoder"):
        EncoderRunner._smphase_orig_execute_mm_encoder = EncoderRunner.execute_mm_encoder

        def _wrapped_execute_mm_encoder(self, mm_kwargs):
            start = time.perf_counter_ns()
            out = self._smphase_orig_execute_mm_encoder(mm_kwargs)
            end = time.perf_counter_ns()
            _record_event("encoder", start, end)
            return out

        EncoderRunner.execute_mm_encoder = _wrapped_execute_mm_encoder

    # V2 model runner prefill/decode hook.
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2

    if not hasattr(GPUModelRunnerV2, "_smphase_orig_execute_model"):
        GPUModelRunnerV2._smphase_orig_execute_model = GPUModelRunnerV2.execute_model

        def _wrapped_execute_model(self, scheduler_output, *args, **kwargs):
            phase = "prefill" if scheduler_output.scheduled_new_reqs else "decode"
            start = time.perf_counter_ns()
            out = self._smphase_orig_execute_model(scheduler_output, *args, **kwargs)
            end = time.perf_counter_ns()
            _record_event(phase, start, end)
            return out

        GPUModelRunnerV2.execute_model = _wrapped_execute_model

    _HOOKS_INSTALLED = True


class PowerSampler:
    def __init__(self, nvml_handle: Any, period_s: float = POWER_SAMPLE_PERIOD_S):
        self._handle = nvml_handle
        self._period = period_s
        self._samples: list[PowerSample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            ts = time.perf_counter_ns()
            power_w = nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            self._samples.append(PowerSample(ts_ns=ts, power_w=power_w))
            time.sleep(self._period)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> list[PowerSample]:
        self._stop.set()
        self._thread.join(timeout=5.0)
        return self._samples


def parse_sm_sweep(raw: str | None) -> list[int]:
    if not raw:
        return list(DEFAULT_SWEEP)
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("SM sweep override was empty")
    return vals


def configure_stream_mask(active_sms: int) -> tuple[int, int]:
    if active_sms >= TOTAL_SMS:
        os.environ["VLLM_STREAM_MASK"] = "0"
        return TOTAL_TPCS, 0

    active_tpcs = max(1, active_sms // 2)
    disabled_tpcs = list(range(active_tpcs, TOTAL_TPCS))
    mask = build_sm_mask(TOTAL_TPCS, disabled_tpcs)
    os.environ["VLLM_STREAM_MASK"] = str(mask)
    return active_tpcs, mask


def build_exact_token_prompt(tokenizer: Any, target_tokens: int) -> str:
    seed = "Explain key ideas in machine learning with clear examples. "
    text = seed * 100
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) < target_tokens:
        while len(token_ids) < target_tokens:
            text += seed
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    token_ids = token_ids[:target_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _phase_power_w(events: list[PhaseEvent], samples: list[PowerSample], phase: str) -> float:
    phase_events = [e for e in events if e.phase == phase]
    if not phase_events or not samples:
        return 0.0

    aligned: list[float] = []
    for event in phase_events:
        in_window = [
            s.power_w for s in samples if event.start_ns <= s.ts_ns <= event.end_ns
        ]
        if in_window:
            aligned.extend(in_window)
        else:
            mid = (event.start_ns + event.end_ns) // 2
            nearest = min(samples, key=lambda s: abs(s.ts_ns - mid))
            aligned.append(nearest.power_w)
    return average(aligned)


def build_metrics_row(
    active_sms: int,
    active_tpcs: int,
    phase_events: list[PhaseEvent],
    power_samples: list[PowerSample],
    summary: dict[str, float],
) -> dict[str, float | int]:
    encoder_total = sum(e.duration_s for e in phase_events if e.phase == "encoder")
    prefill_total = sum(e.duration_s for e in phase_events if e.phase == "prefill")
    decode_total = sum(e.duration_s for e in phase_events if e.phase == "decode")

    encoder_latency = encoder_total / REQUESTS if REQUESTS else 0.0
    prefill_latency = prefill_total / REQUESTS if REQUESTS else 0.0
    decode_latency = decode_total / REQUESTS if REQUESTS else 0.0

    encoder_throughput = REQUESTS / encoder_total if encoder_total > 0 else 0.0
    prefill_throughput = (REQUESTS * INPUT_TOKENS) / prefill_total if prefill_total > 0 else 0.0
    decode_throughput = (REQUESTS * OUTPUT_TOKENS) / decode_total if decode_total > 0 else 0.0

    encoder_power = _phase_power_w(phase_events, power_samples, "encoder")
    prefill_power = _phase_power_w(phase_events, power_samples, "prefill")
    decode_power = _phase_power_w(phase_events, power_samples, "decode")
    total_power = average([s.power_w for s in power_samples])

    return {
        "active_sms": active_sms,
        "active_tpcs": active_tpcs,
        "encoder_latency": encoder_latency,
        "prefill_latency": prefill_latency,
        "decode_latency": decode_latency,
        "encoder_throughput": encoder_throughput,
        "prefill_throughput": prefill_throughput,
        "decode_throughput": decode_throughput,
        "encoder_power": encoder_power,
        "prefill_power": prefill_power,
        "decode_power": decode_power,
        "total_latency": float(summary.get("avg_latency", 0.0)),
        "total_throughput": float(summary.get("throughput_tok_per_s", 0.0)),
        "total_power": total_power,
    }


def run_one_setting(
    *,
    model_name: str,
    image_path: Path,
    active_sms: int,
    nvml_handle: Any,
    gpu_memory_utilization: float,
) -> dict[str, float | int]:
    global _ACTIVE_EVENTS

    active_tpcs, _ = configure_stream_mask(active_sms)

    torch.cuda.empty_cache()

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        max_num_seqs=1,
    )

    tokenizer = llm.get_tokenizer()
    prompt_text = build_exact_token_prompt(tokenizer, INPUT_TOKENS)
    test_image = Image.open(image_path).convert("RGB")

    request_payload = {
        "prompt": "<image>\n" + prompt_text,
        "multi_modal_data": {"image": test_image},
    }
    sampling = SamplingParams(max_tokens=OUTPUT_TOKENS, temperature=0.0)

    for _ in range(WARMUP_REQUESTS):
        llm.generate([request_payload], sampling)

    core = llm.llm_engine.engine_core

    def rpc(method: str):
        return core.collective_rpc(method=method, args=(), kwargs=None)[0]

    rpc("reset_metrics")
    rpc("start_metrics_run")

    with _EVENT_LOCK:
        _ACTIVE_EVENTS = []

    sampler = PowerSampler(nvml_handle)
    sampler.start()
    for _ in range(REQUESTS):
        llm.generate([request_payload], sampling)
    power_samples = sampler.stop()

    rpc("end_metrics_run")
    summary = rpc("get_metrics_summary")
    phase_events = list(_ACTIVE_EVENTS or [])
    with _EVENT_LOCK:
        _ACTIVE_EVENTS = None

    row = build_metrics_row(
        active_sms=active_sms,
        active_tpcs=active_tpcs,
        phase_events=phase_events,
        power_samples=power_samples,
        summary=summary,
    )

    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return row


def validate_extreme_masking(
    model_name: str,
    image_path: Path,
    nvml_handle: Any,
    low_sms: int,
    gpu_memory_utilization: float,
) -> dict[str, Any]:
    full_row = run_one_setting(
        model_name=model_name,
        image_path=image_path,
        active_sms=TOTAL_SMS,
        nvml_handle=nvml_handle,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    low_row = run_one_setting(
        model_name=model_name,
        image_path=image_path,
        active_sms=low_sms,
        nvml_handle=nvml_handle,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    full_latency = float(full_row["total_latency"])
    low_latency = float(low_row["total_latency"])
    full_tp = float(full_row["total_throughput"])
    low_tp = float(low_row["total_throughput"])

    latency_ratio = (low_latency / full_latency) if full_latency > 0 else 0.0
    throughput_ratio = (low_tp / full_tp) if full_tp > 0 else 0.0

    passed = latency_ratio >= 1.20 and throughput_ratio <= 0.80
    return {
        "passed": passed,
        "full": full_row,
        "low": low_row,
        "latency_ratio_low_over_full": latency_ratio,
        "throughput_ratio_low_over_full": throughput_ratio,
    }


def write_csv(rows: list[dict[str, float | int]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="OpenGVLab/InternVL3-2B",
        help="InternVL3 model id to run",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("test_image.png"),
        help="Image path for multimodal input",
    )
    parser.add_argument(
        "--sm-sweep",
        default=None,
        help="Comma-separated active SM values",
    )
    parser.add_argument(
        "--extreme-low-sm",
        type=int,
        default=4,
        help="Low SM value for masking validation",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("experiment_outputs/sm_phase_results.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
        help="vLLM GPU memory utilization fraction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Missing image: {args.image}")

    sweep = parse_sm_sweep(args.sm_sweep)
    if args.extreme_low_sm < 1:
        raise ValueError("--extreme-low-sm must be >= 1")

    install_phase_hooks()

    nvmlInit()
    nvml_handle = nvmlDeviceGetHandleByIndex(0)
    gpu_name = nvmlDeviceGetName(nvml_handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode("utf-8", errors="replace")

    print(f"GPU: {gpu_name}")
    print(f"Detected SMs: {torch.cuda.get_device_properties(0).multi_processor_count}")
    print("Running extreme validation before sweep...")

    validation = validate_extreme_masking(
        model_name=args.model,
        image_path=args.image,
        nvml_handle=nvml_handle,
        low_sms=args.extreme_low_sm,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    validation_path = args.output_csv.with_suffix(".validation.json")
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")

    if not validation["passed"]:
        print("EXTREME VALIDATION FAILED")
        raise RuntimeError(
            "SM masking validation failed: low-SM did not clearly increase latency "
            "and reduce throughput. Sweep aborted; plots would be invalid."
        )

    print("EXTREME VALIDATION PASSED")
    print("Validation passed. Collecting SM sweep data...")
    rows: list[dict[str, float | int]] = []
    for active_sms in sweep:
        print(f"Collecting active_sms={active_sms}")
        row = run_one_setting(
            model_name=args.model,
            image_path=args.image,
            active_sms=active_sms,
            nvml_handle=nvml_handle,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        rows.append(row)

    write_csv(rows, args.output_csv)
    print(f"Wrote CSV: {args.output_csv}")
    print(f"Validation sidecar: {validation_path}")


if __name__ == "__main__":
    main()
