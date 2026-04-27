#!/usr/bin/env bash
set -euo pipefail

cd /workspace/mllm_sm_vars/vllm

export LIBSMCTRL_PATH=/workspace/mllm_sm_vars/vllm/vllm/v1/.libs/libsmctrl.so
export LD_LIBRARY_PATH=/workspace/mllm_sm_vars/vllm/vllm/v1/.libs:${LD_LIBRARY_PATH:-}
export VLLM_STREAM_MASK_TOUCH_LOG=stream_mask_touches.log

python -u test_metrics.py | tee -a screen_experiment.log
