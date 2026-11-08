"""Reproducible synchronized attention benchmarks; execute with python -m."""
import argparse
import gc
import itertools
import json
from pathlib import Path
import time

import torch
import torch._functorch.config
torch._functorch.config.donated_buffer = False
from cs336_systems.attention import attention


def benchmark_attention(compiled=False, iterations=100):
    rows = []
    for d, n in itertools.product([16, 32, 64, 128], [256, 1024, 4096, 8192, 16384]):
        torch.compiler.reset()
        gc.collect()
        torch.cuda.empty_cache()
        row = dict(d=d, n=n, batch=8, compiled=compiled, iterations=iterations)
        try:
            q, k, v = [torch.randn(8, n, d, device='cuda', requires_grad=True) for _ in range(3)]
            do = torch.randn_like(q)
            fn = torch.compile(attention) if compiled else attention
            for _ in range(5):
                torch.autograd.grad(fn(q, k, v), (q, k, v), do)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                o = fn(q, k, v)
                torch.cuda.synchronize()
                del o
            row['forward_ms'] = (time.perf_counter() - start) * 1000 / iterations
            o = fn(q, k, v)
            torch.cuda.synchronize()
            row['before_backward_bytes'] = torch.cuda.memory_allocated()
            row['saved_activation_bytes'] = row['before_backward_bytes'] - sum(t.numel()*t.element_size() for t in (q,k,v,do))
            start = time.perf_counter()
            for _ in range(iterations):
                torch.autograd.grad(o, (q, k, v), do, retain_graph=True)
                torch.cuda.synchronize()
            row['backward_ms'] = (time.perf_counter() - start) * 1000 / iterations
            del o, q, k, v, do
            row['status'] = 'ok'
        except torch.OutOfMemoryError:
            row['status'] = 'OOM'
        rows.append(row)
        print(row, flush=True)
    return dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, rows=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiled', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = benchmark_attention(args.compiled)
    Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
