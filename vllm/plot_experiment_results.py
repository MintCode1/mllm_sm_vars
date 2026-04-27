#!/usr/bin/env python3
"""Generate plots from SM sweep experiment results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_CSV = Path("results.csv")


def main() -> None:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {RESULTS_CSV}")

    df = pd.read_csv(RESULTS_CSV)
    df = df.sort_values(by="active_sms")

    x = df["active_sms"]

    # Throughput vs SM
    plt.figure(figsize=(8, 5))
    plt.plot(x, df["throughput_tok_per_s"], marker="o", linewidth=2)
    plt.xlabel("Active SMs")
    plt.ylabel("Throughput (tokens/s)")
    plt.title("Throughput vs SM Allocation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("throughput_vs_sm.png", dpi=200)
    plt.close()

    # Encoder phase vs SM
    plt.figure(figsize=(8, 5))
    plt.plot(x, df["encoder_avg"], marker="o", linewidth=2, color="tab:blue")
    plt.xlabel("Active SMs")
    plt.ylabel("Encoder Avg Time (s)")
    plt.title("Encoder Time vs SM Allocation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("encoder_vs_sm.png", dpi=200)
    plt.close()

    # Prefill phase vs SM
    plt.figure(figsize=(8, 5))
    plt.plot(x, df["prefill_avg"], marker="o", linewidth=2, color="tab:orange")
    plt.xlabel("Active SMs")
    plt.ylabel("Prefill Avg Time (s)")
    plt.title("Prefill Time vs SM Allocation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("prefill_vs_sm.png", dpi=200)
    plt.close()

    # Decode phase vs SM
    plt.figure(figsize=(8, 5))
    plt.plot(x, df["decode_avg"], marker="o", linewidth=2, color="tab:green")
    plt.xlabel("Active SMs")
    plt.ylabel("Decode Avg Time (s)")
    plt.title("Decode Time vs SM Allocation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("decode_vs_sm.png", dpi=200)
    plt.close()

    # Normalized phase comparison to expose relative scaling behavior.
    enc_norm = df["encoder_avg"] / df["encoder_avg"].max()
    pre_norm = df["prefill_avg"] / df["prefill_avg"].max()
    dec_norm = df["decode_avg"] / df["decode_avg"].max()

    plt.figure(figsize=(9, 5.5))
    plt.plot(x, enc_norm, marker="o", linewidth=2, label="Encoder (normalized)")
    plt.plot(x, pre_norm, marker="o", linewidth=2, label="Prefill (normalized)")
    plt.plot(x, dec_norm, marker="o", linewidth=2, label="Decode (normalized)")
    plt.xlabel("Active SMs")
    plt.ylabel("Normalized Phase Time")
    plt.title("Normalized Encoder/Prefill/Decode Scaling vs SM Allocation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("phases_comparison.png", dpi=200)
    plt.close()

    print("Generated plots:")
    print("- throughput_vs_sm.png")
    print("- encoder_vs_sm.png")
    print("- prefill_vs_sm.png")
    print("- decode_vs_sm.png")
    print("- phases_comparison.png")


if __name__ == "__main__":
    main()
