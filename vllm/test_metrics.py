#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Software-level spatial multiplexing sweep with repeated trials.

For each workload configuration, this script runs:
1) Sequential mode: prefill-heavy request then decode-heavy request.
2) Overlap mode: staggered asynchronous submission via separate threads.

Then it averages key metrics across trials, saves structured CSV results,
and generates comparison plots.
"""

import csv
import gc
import inspect
import math
import os
import subprocess
import threading
import time
from statistics import mean
from typing import Iterable

import torch

# Must be set BEFORE vLLM is imported.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
os.environ["VLLM_USE_DEEP_GEMM"] = "0"

from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName,
    nvmlDeviceGetPowerUsage,
    nvmlInit,
)
from vllm import LLM, SamplingParams

# Experiment controls
RESUME = False
WARMUP_REQUESTS = 3
STAGGER_SECONDS = 0.05
NUM_TRIALS = 3

CONFIGS = [
    {"prefill_len": 512, "decode_len": 100},
    {"prefill_len": 512, "decode_len": 300},
    {"prefill_len": 1024, "decode_len": 100},
    {"prefill_len": 1024, "decode_len": 300},
]

SHORT_PROMPT = "Hello"
LONG_MAX_TOKENS = 16

OUTPUT_CSV = "results_multiplexing_sweep.csv"
PLOT_THROUGHPUT = "throughput_vs_config.png"
PLOT_WALLTIME = "walltime_vs_config.png"
PLOT_POWER_EFF = "power_efficiency_vs_config.png"
PLOT_IMPROVEMENT_CFG = "improvement_vs_config.png"
PLOT_IMPROVEMENT_RATIO = "improvement_vs_ratio.png"
PLOT_IMPROVEMENT_SLACK = "improvement_vs_slack.png"
PLOT_IMPROVEMENT_REG = "improvement_regression.png"

ANALYSIS_DIR = "experiment_outputs/multiplexing_refined"
ANALYSIS_CSV = os.path.join(ANALYSIS_DIR, "results_multiplexing_sweep_refined.csv")
ANALYSIS_PLOT_THROUGHPUT = os.path.join(ANALYSIS_DIR, PLOT_THROUGHPUT)
ANALYSIS_PLOT_WALLTIME = os.path.join(ANALYSIS_DIR, PLOT_WALLTIME)
ANALYSIS_PLOT_POWER_EFF = os.path.join(ANALYSIS_DIR, PLOT_POWER_EFF)
ANALYSIS_PLOT_IMPROVEMENT_CFG = os.path.join(ANALYSIS_DIR, PLOT_IMPROVEMENT_CFG)
ANALYSIS_PLOT_IMPROVEMENT_RATIO = os.path.join(ANALYSIS_DIR, PLOT_IMPROVEMENT_RATIO)
ANALYSIS_PLOT_IMPROVEMENT_SLACK = os.path.join(ANALYSIS_DIR, PLOT_IMPROVEMENT_SLACK)
ANALYSIS_PLOT_IMPROVEMENT_REG = os.path.join(ANALYSIS_DIR, PLOT_IMPROVEMENT_REG)

MODEL_CANDIDATES = [
    "OpenGVLab/InternVL3-2B",
    "OpenGVLab/InternVL3-9B",
    "OpenGVLab/InternVL3-8B",
]

# Global power sampling
handle = None


def _iter_models(models: Iterable[str]) -> Iterable[str]:
    for model_name in models:
        yield model_name


def create_llm_with_fallback() -> tuple[LLM, str]:
    last_exc: Exception | None = None
    for model_name in _iter_models(MODEL_CANDIDATES):
        try:
            llm_kwargs = dict(
                model=model_name,
                trust_remote_code=True,
                max_model_len=4096,
                gpu_memory_utilization=0.9,
                enforce_eager=True,
                max_num_seqs=2,
            )
            if "disable_cuda_graph" in inspect.signature(LLM.__init__).parameters:
                llm_kwargs["disable_cuda_graph"] = True

            llm = LLM(**llm_kwargs)
            return llm, model_name
        except Exception as exc:
            print(f"Model init failed for {model_name}: {exc}")
            last_exc = exc

    raise RuntimeError("Failed to initialize any candidate model") from last_exc


def _sample_power() -> float:
    return nvmlDeviceGetPowerUsage(handle) / 1000.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def make_prefill_prompt(prefill_len: int) -> str:
    # Approximate token-count scaling by repeating short words.
    return ("relativity " * prefill_len).strip()


def _run_sequential(
    llm: LLM,
    long_prompt: str,
    long_sampling: SamplingParams,
    short_sampling: SamplingParams,
) -> int:
    llm.generate([long_prompt], long_sampling)
    llm.generate([SHORT_PROMPT], short_sampling)
    return 2


def _run_overlap_staggered(
    llm: LLM,
    long_prompt: str,
    long_sampling: SamplingParams,
    short_sampling: SamplingParams,
) -> int:
    def run_prefill() -> None:
        llm.enqueue([long_prompt], long_sampling, use_tqdm=False)

    def run_decode() -> None:
        llm.enqueue([SHORT_PROMPT], short_sampling, use_tqdm=False)

    t1 = threading.Thread(target=run_prefill)
    t2 = threading.Thread(target=run_decode)
    t1.start()
    time.sleep(STAGGER_SECONDS)
    t2.start()
    t1.join()
    t2.join()
    outputs = llm.wait_for_completion(use_tqdm=False)
    return len(outputs)


def _run_scenario(
    call_metric_rpc,
    run_fn,
) -> dict:
    power_samples: list[float] = [_sample_power()]
    call_metric_rpc("reset_metrics")
    call_metric_rpc("start_metrics_run")
    start = time.time()
    num_outputs = run_fn()
    torch.cuda.synchronize()
    wall_time = time.time() - start
    power_samples.append(_sample_power())
    call_metric_rpc("end_metrics_run")

    summary = call_metric_rpc("get_metrics_summary")
    avg_power = _mean(power_samples)
    gpu_util = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    ).decode().strip()

    return {
        "wall_time": wall_time,
        "num_requests": num_outputs,
        "avg_latency": summary["avg_latency"],
        "throughput_tok_per_s": summary["throughput_tok_per_s"],
        "encoder_avg": summary["encoder_avg"],
        "prefill_avg": summary["prefill_avg"],
        "decode_avg": summary["decode_avg"],
        "avg_power": avg_power,
        "gpu_utilization": float(gpu_util),
    }


def build_average_row(prefill_len: int, decode_len: int, mode: str,
                      trial_results: list[dict]) -> dict:
    return {
        "prefill_len": prefill_len,
        "decode_len": decode_len,
        "mode": mode,
        "wall_time": mean(r["wall_time"] for r in trial_results),
        "throughput": mean(r["throughput_tok_per_s"] for r in trial_results),
        "power": mean(r["avg_power"] for r in trial_results),
        "utilization": mean(r["gpu_utilization"] for r in trial_results),
        "encoder_avg": mean(r["encoder_avg"] for r in trial_results),
        "prefill_avg": mean(r["prefill_avg"] for r in trial_results),
        "decode_avg": mean(r["decode_avg"] for r in trial_results),
    }


def write_results_csv(df) -> None:
    df.to_csv(ANALYSIS_CSV, index=False)


def make_plots(df) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df["config"] = df.apply(
        lambda r: f"P{int(r['prefill_len'])}-D{int(r['decode_len'])}", axis=1)
    order = [
        f"P{cfg['prefill_len']}-D{cfg['decode_len']}" for cfg in CONFIGS
    ]
    df["config"] = pd.Categorical(df["config"], categories=order, ordered=True)
    df = df.sort_values("config")
    df["power_efficiency"] = df["throughput"] / df["power"]

    def _plot(metric: str, ylabel: str, title: str, output: str) -> None:
        plt.figure(figsize=(9, 5))
        for mode in ["sequential", "overlap"]:
            sub = df[df["mode"] == mode]
            plt.plot(sub["config"], sub[metric], marker="o", linewidth=2,
                     label=mode)
        plt.xlabel("Configuration (prefill/decode)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output, dpi=200)
        plt.close()

    _plot(
        metric="throughput",
        ylabel="Throughput (tok/s)",
        title="Throughput vs Workload Configuration",
        output=ANALYSIS_PLOT_THROUGHPUT,
    )
    _plot(
        metric="wall_time",
        ylabel="Wall Time (s)",
        title="Wall Time vs Workload Configuration",
        output=ANALYSIS_PLOT_WALLTIME,
    )
    _plot(
        metric="power_efficiency",
        ylabel="Throughput / Power (tok/s/W)",
        title="Power Efficiency vs Workload Configuration",
        output=ANALYSIS_PLOT_POWER_EFF,
    )

    benefit_df = (
        df[df["mode"] == "overlap"]
        .copy()
        .sort_values("config")
    )

    plt.figure(figsize=(9, 5))
    plt.plot(
        benefit_df["config"],
        benefit_df["throughput_improvement_pct"],
        marker="o",
        linewidth=2,
    )
    plt.xlabel("Workload (Prefill / Decode)")
    plt.ylabel("Throughput Improvement (%)")
    plt.title("Overlap Benefit vs Workload Balance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ANALYSIS_PLOT_IMPROVEMENT_CFG, dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.scatter(
        benefit_df["prefill_decode_ratio"],
        benefit_df["throughput_improvement_pct"],
    )
    plt.xlabel("Prefill / Decode Ratio")
    plt.ylabel("Throughput Improvement (%)")
    plt.title("Benefit vs Phase Balance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ANALYSIS_PLOT_IMPROVEMENT_RATIO, dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.scatter(
        benefit_df["slack"],
        benefit_df["throughput_improvement_pct"],
    )
    plt.xlabel("Compute Slack (prefill_avg - decode_avg)")
    plt.ylabel("Throughput Improvement (%)")
    plt.title("Overlap Benefit vs Compute Slack")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ANALYSIS_PLOT_IMPROVEMENT_SLACK, dpi=200)
    plt.close()

    x = benefit_df["balance"].to_numpy(dtype=float)
    y = benefit_df["throughput_improvement_pct"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept

    plt.figure(figsize=(9, 5))
    plt.scatter(x, y, label="configs")
    order_idx = np.argsort(x)
    plt.plot(x[order_idx], y_fit[order_idx], color="tab:red", label="linear fit")
    plt.xlabel("Balance = min(prefill, decode) / max(prefill, decode)")
    plt.ylabel("Throughput Improvement (%)")
    plt.title("Overlap Benefit vs Workload Balance (Regression)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ANALYSIS_PLOT_IMPROVEMENT_REG, dpi=200)
    plt.close()


def build_analysis_dataframe(rows: list[dict]):
    import pandas as pd

    df = pd.DataFrame(rows)

    pivot = (
        df.pivot_table(
            index=["prefill_len", "decode_len"],
            columns="mode",
            values=["throughput", "wall_time"],
            aggfunc="first",
        )
        .reset_index()
    )

    seq_thr = pivot[("throughput", "sequential")]
    ovl_thr = pivot[("throughput", "overlap")]
    seq_wall = pivot[("wall_time", "sequential")]
    ovl_wall = pivot[("wall_time", "overlap")]

    throughput_improvement_pct = (ovl_thr - seq_thr) / seq_thr * 100.0
    latency_improvement_pct = (seq_wall - ovl_wall) / seq_wall * 100.0

    improvement_df = pd.DataFrame(
        {
            "prefill_len": pivot[("prefill_len", "")],
            "decode_len": pivot[("decode_len", "")],
            "throughput_improvement_pct": throughput_improvement_pct,
            "latency_improvement_pct": latency_improvement_pct,
        }
    )

    df = df.merge(improvement_df, on=["prefill_len", "decode_len"], how="left")
    df["prefill_decode_ratio"] = df["prefill_len"] / df["decode_len"]
    df["balance"] = df.apply(
        lambda r: min(r["prefill_len"], r["decode_len"]) /
        max(r["prefill_len"], r["decode_len"]),
        axis=1,
    )
    df["slack"] = df["prefill_avg"] - df["decode_avg"]
    return df


def print_regression_stats(df) -> None:
    import numpy as np

    benefit_df = (
        df[df["mode"] == "overlap"]
        .copy()
        .sort_values(["prefill_len", "decode_len"])
    )
    x_ratio = benefit_df["prefill_decode_ratio"].to_numpy(dtype=float)
    x_log_ratio = np.log(x_ratio)
    y = benefit_df["throughput_improvement_pct"].to_numpy(dtype=float)

    def fit_and_r2(x, y_values):
        slope, intercept = np.polyfit(x, y_values, 1)
        y_pred = slope * x + intercept
        ss_res = np.sum((y_values - y_pred) ** 2)
        ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
        return slope, r2

    slope_ratio, r2_ratio = fit_and_r2(x_ratio, y)
    slope_log_ratio, r2_log_ratio = fit_and_r2(x_log_ratio, y)

    print("\nRegression: throughput_improvement_pct ~ prefill_decode_ratio")
    print(f"  slope={slope_ratio:.6f}")
    print(f"  R^2={r2_ratio:.6f}")

    print("Regression: throughput_improvement_pct ~ log(prefill_decode_ratio)")
    print(f"  slope={slope_log_ratio:.6f}")
    print(f"  R^2={r2_log_ratio:.6f}")


def main():
    global handle

    print(">>> Starting script")

    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    if not RESUME and os.path.exists(ANALYSIS_CSV):
        os.remove(ANALYSIS_CSV)

    print("=" * 60)
    print("SOFTWARE SPATIAL MULTIPLEXING EXPERIMENT")
    print("=" * 60)

    # Initialize NVML for GPU power measurement
    print("\nInitializing NVML...")
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    print(f"GPU: {nvmlDeviceGetName(handle)}")
    print(f"Detected SMs: {torch.cuda.get_device_properties(0).multi_processor_count}")

    print("\nWorkloads:")
    for cfg in CONFIGS:
        print(
            f"  - prefill_len={cfg['prefill_len']}, decode_len={cfg['decode_len']}"
        )
    print(f"  - Heavy decode prompt: {SHORT_PROMPT}")
    print(f"  - Warmup requests per scenario: {WARMUP_REQUESTS}")
    print(f"  - Trials per config: {NUM_TRIALS}")

    print(">>> About to initialize LLM")
    print("Initializing vLLM...")
    llm, chosen_model = create_llm_with_fallback()
    print(">>> LLM initialized")
    print(f"Using model: {chosen_model}")

    engine_core_client = llm.llm_engine.engine_core

    def call_metric_rpc(method: str):
        return engine_core_client.collective_rpc(
            method=method,
            args=(),
            kwargs=None,
        )[0]

    averaged_rows: list[dict] = []

    for cfg in CONFIGS:
        prefill_len = cfg["prefill_len"]
        decode_len = cfg["decode_len"]
        long_prompt = make_prefill_prompt(prefill_len)
        long_sampling = SamplingParams(max_tokens=LONG_MAX_TOKENS, temperature=0.0)
        short_sampling = SamplingParams(max_tokens=decode_len, temperature=0.0)

        print("\n" + "=" * 60)
        print(
            f"CONFIG prefill_len={prefill_len}, decode_len={decode_len}"
        )
        print("=" * 60)

        sequential_trials: list[dict] = []
        overlap_trials: list[dict] = []

        for trial_idx in range(NUM_TRIALS):
            print(f"\nTrial {trial_idx + 1}/{NUM_TRIALS} - sequential")
            for _ in range(WARMUP_REQUESTS):
                _run_sequential(llm, long_prompt, long_sampling, short_sampling)

            sequential_result = _run_scenario(
                call_metric_rpc,
                lambda: _run_sequential(llm, long_prompt, long_sampling, short_sampling),
            )
            sequential_trials.append(sequential_result)

            print(f"\nTrial {trial_idx + 1}/{NUM_TRIALS} - overlap")
            for _ in range(WARMUP_REQUESTS):
                _run_overlap_staggered(llm, long_prompt, long_sampling, short_sampling)

            overlap_result = _run_scenario(
                call_metric_rpc,
                lambda: _run_overlap_staggered(
                    llm, long_prompt, long_sampling, short_sampling),
            )
            overlap_trials.append(overlap_result)

            print(
                f"Phase validation (overlap trial {trial_idx + 1}): "
                f"encoder_avg={overlap_result['encoder_avg']}, "
                f"prefill_avg={overlap_result['prefill_avg']}, "
                f"decode_avg={overlap_result['decode_avg']}"
            )

        seq_avg = build_average_row(prefill_len, decode_len, "sequential",
                                    sequential_trials)
        ovl_avg = build_average_row(prefill_len, decode_len, "overlap",
                                    overlap_trials)
        averaged_rows.extend([seq_avg, ovl_avg])

        seq_eff = seq_avg["throughput"] / seq_avg["power"] if seq_avg["power"] else 0.0
        ovl_eff = ovl_avg["throughput"] / ovl_avg["power"] if ovl_avg["power"] else 0.0
        improvement = ((ovl_avg["throughput"] - seq_avg["throughput"]) /
                       seq_avg["throughput"] * 100.0) if seq_avg["throughput"] else 0.0
        print("\nConfig summary:")
        print(f"  sequential throughput={seq_avg['throughput']:.3f}, wall_time={seq_avg['wall_time']:.3f}, util={seq_avg['utilization']:.1f}")
        print(f"  overlap throughput={ovl_avg['throughput']:.3f}, wall_time={ovl_avg['wall_time']:.3f}, util={ovl_avg['utilization']:.1f}")
        print(f"  throughput improvement={improvement:.2f}%")
        print(f"  power efficiency seq={seq_eff:.4f}, overlap={ovl_eff:.4f}")

    analysis_df = build_analysis_dataframe(averaged_rows)
    write_results_csv(analysis_df)
    make_plots(analysis_df)
    print_regression_stats(analysis_df)

    print("Observation:")
    print("Overlap provides maximum benefit when workload balance enables compute slack, allowing memory-bound decode to be hidden under compute-bound prefill.")

    print("\n" + "-" * 60)
    print("SWEEP COMPLETE")
    print("-" * 60)
    print(f"Saved averaged CSV: {ANALYSIS_CSV}")
    print(f"Saved plot: {ANALYSIS_PLOT_THROUGHPUT}")
    print(f"Saved plot: {ANALYSIS_PLOT_WALLTIME}")
    print(f"Saved plot: {ANALYSIS_PLOT_POWER_EFF}")
    print(f"Saved plot: {ANALYSIS_PLOT_IMPROVEMENT_CFG}")
    print(f"Saved plot: {ANALYSIS_PLOT_IMPROVEMENT_RATIO}")
    print(f"Saved plot: {ANALYSIS_PLOT_IMPROVEMENT_SLACK}")
    print(f"Saved plot: {ANALYSIS_PLOT_IMPROVEMENT_REG}")

    # Release engine resources after the full sweep.
    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
