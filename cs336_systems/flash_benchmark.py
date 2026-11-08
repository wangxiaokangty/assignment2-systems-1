"""Full problem grid, using triton.testing.do_bench for each measured phase."""

import argparse
import gc
import itertools
import json
from pathlib import Path
import torch
import torch._functorch.config
import triton.testing
from cs336_systems.attention import attention, FlashAttentionTriton


def benchmark_case(n, d, dtype, implementation):
    q, k, v = [torch.randn(1, n, d, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
    do = torch.randn_like(q)
    fn = attention if implementation == "pytorch" else FlashAttentionTriton.apply

    def forward():
        return fn(q, k, v, True)

    def combined():
        return torch.autograd.grad(forward(), (q, k, v), do)

    result = {}
    # Record forward independently, even when dense backward cannot fit.
    result["forward_ms"] = triton.testing.do_bench(forward, warmup=10, rep=30)
    try:
        output = forward()

        def backward():
            return torch.autograd.grad(output, (q, k, v), do, retain_graph=True)

        result["backward_ms"] = triton.testing.do_bench(backward, warmup=10, rep=30)
        output = None
        result["end_to_end_ms"] = triton.testing.do_bench(combined, warmup=10, rep=30)
        result["status"] = "ok"
    except torch.OutOfMemoryError:
        result["status"] = "backward OOM"
    return result


def benchmark_flash(output):
    torch._functorch.config.donated_buffer = False
    rows = []
    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, batch=1, causal=True, warmup_ms=10, rep_ms=30, rows=rows)
    for n, d, dtype, impl in itertools.product([2**i for i in range(7, 17)], [16, 32, 64, 128], [torch.bfloat16, torch.float32], ["pytorch", "flash"]):
        gc.collect()
        torch.cuda.empty_cache()
        row = dict(n=n, d=d, dtype=str(dtype), implementation=impl)
        try:
            row.update(benchmark_case(n, d, dtype, impl))
        except torch.OutOfMemoryError:
            row["status"] = "forward OOM"
        rows.append(row)
        Path(output).write_text(json.dumps(result, indent=2))
        print(row, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/flash_benchmarking.json")
    args = parser.parse_args()
    benchmark_flash(args.output)
