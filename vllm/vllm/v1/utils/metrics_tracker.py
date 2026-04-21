# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight metrics tracker for measuring request latency, throughput, and phase timings."""

import time


class MetricsTracker:
    """Tracks metrics for controlled experiments.
    
    Measures:
    - Average latency per request
    - Throughput (tokens/sec)
    - Phase-level timing (encoder, prefill, decode)
    """

    def __init__(self):
        """Initialize the metrics tracker."""
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.request_latencies = []
        self.total_tokens = 0
        self.start_time = None
        self.end_time = None

        self.encoder_times = []
        self.prefill_times = []
        self.decode_times = []

    def start_run(self):
        """Mark the start of the measurement run."""
        self.start_time = time.time()

    def end_run(self):
        """Mark the end of the measurement run."""
        self.end_time = time.time()

    def record_request_latency(self, latency):
        """Record the latency of a single request in seconds."""
        self.request_latencies.append(latency)

    def add_tokens(self, n):
        """Add n tokens to the total token count."""
        self.total_tokens += n

    def record_phase(self, phase, duration):
        """Record the duration of a phase in seconds.
        
        Args:
            phase: Phase name ("encoder", "prefill", or "decode")
            duration: Duration in seconds
        """
        if phase == "encoder":
            self.encoder_times.append(duration)
        elif phase == "prefill":
            self.prefill_times.append(duration)
        elif phase == "decode":
            self.decode_times.append(duration)

    def summary(self):
        """Return a summary of collected metrics.
        
        Returns:
            dict: Summary containing average latency, throughput, and phase timings
        """
        total_time = self.end_time - self.start_time if self.end_time else 0

        return {
            "avg_latency": sum(self.request_latencies) / len(self.request_latencies)
            if self.request_latencies
            else 0,
            "throughput_tok_per_s": self.total_tokens / total_time if total_time > 0 else 0,
            "encoder_avg": sum(self.encoder_times) / len(self.encoder_times)
            if self.encoder_times
            else 0,
            "prefill_avg": sum(self.prefill_times) / len(self.prefill_times)
            if self.prefill_times
            else 0,
            "decode_avg": sum(self.decode_times) / len(self.decode_times)
            if self.decode_times
            else 0,
        }
