import sys
import time

active_sms = int(sys.argv[1])

from vllm.v1.libsmctrl import create_sm_limited_context

print(f"Creating SM-limited CUDA context with active_sms={active_sms}", flush=True)
create_sm_limited_context(active_sms)

import torch

print("CUDA device:", torch.cuda.get_device_name(0), flush=True)
print("Reported SMs:", torch.cuda.get_device_properties(0).multi_processor_count, flush=True)

a = torch.randn((8192, 8192), device="cuda", dtype=torch.float16)
b = torch.randn((8192, 8192), device="cuda", dtype=torch.float16)

for _ in range(5):
    c = a @ b
torch.cuda.synchronize()

start = time.time()
for _ in range(20):
    c = a @ b
torch.cuda.synchronize()

elapsed = time.time() - start
print(f"elapsed={elapsed:.6f}", flush=True)
