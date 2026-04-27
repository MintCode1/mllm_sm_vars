import os
import time
import torch

from vllm.v1.libsmctrl import build_sm_mask, apply_configured_stream_mask

TOTAL_TPCS = 66


def run_case(label, active_tpcs):
    if active_tpcs >= TOTAL_TPCS:
        mask = 0
    else:
        disabled = list(range(active_tpcs, TOTAL_TPCS))
        mask = build_sm_mask(TOTAL_TPCS, disabled)

    os.environ["VLLM_STREAM_MASK"] = str(mask)

    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    print(f"\n=== {label} active_tpcs={active_tpcs} mask={hex(mask)} ===", flush=True)

    with torch.cuda.stream(stream):
        applied = apply_configured_stream_mask(stream)
        print(f"mask_applied={applied}, stream_ptr={int(stream.cuda_stream)}", flush=True)

        a = torch.randn((8192, 8192), device="cuda", dtype=torch.float16)
        b = torch.randn((8192, 8192), device="cuda", dtype=torch.float16)

        # warmup
        for _ in range(5):
            c = a @ b
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(20):
            c = a @ b
        torch.cuda.synchronize()
        elapsed = time.time() - start

    print(f"elapsed={elapsed:.6f}s", flush=True)
    return elapsed


if __name__ == "__main__":
    torch.cuda.init()
    print(torch.cuda.get_device_name(0), flush=True)

    full = run_case("FULL", 66)
    small = run_case("LIMITED", 4)

    print("\nRESULT")
    print(f"full={full:.6f}")
    print(f"limited={small:.6f}")
    print(f"ratio_limited_over_full={small / full:.3f}")
