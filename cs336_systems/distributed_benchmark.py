"""Reproducible single-node collective, DDP, and FSDP measurements.

Example: CUDA_VISIBLE_DEVICES=4,5 uv run python -m
cs336_systems.distributed_benchmark train --mode naive --size xl
"""
import argparse
import json
import os
from pathlib import Path
import socket
import statistics
import time

import torch
from torch import distributed as dist, multiprocessing as mp

from cs336_systems.distributed import DDP, NaiveDDP
from cs336_systems.fsdp import FSDP
from cs336_systems.sharded_optimizer import ShardedOptimizer


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _worker(rank, world, port, config):
    torch.set_num_threads(4)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    device = torch.device(f"cuda:{rank}" if config["backend"] == "nccl" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(config["backend"], rank=rank, world_size=world)
    try:
        if config["command"] == "collective":
            rows = _collective(device, config)
        else:
            rows = _train(device, config)
        gathered = [None] * world
        dist.all_gather_object(gathered, rows)
        if rank == 0:
            output = {"world_size": world, "backend": config["backend"],
                      "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                      "config": config, "ranks": gathered}
            Path(config["output"]).parent.mkdir(parents=True, exist_ok=True)
            Path(config["output"]).write_text(json.dumps(output, indent=2))
            print(json.dumps(output, indent=2), flush=True)
    finally:
        dist.destroy_process_group()


def _collective(device, config):
    rows = []
    for megabytes in (1, 10, 100, 1000):
        # Decimal MB/GB, as reported explicitly in the result.
        tensor = torch.ones(megabytes * 1_000_000 // 4, device=device, dtype=torch.float32)
        samples = []
        for iteration in range(config["warmup"] + config["steps"]):
            tensor.fill_(1)
            dist.barrier()
            _sync(device)
            start = time.perf_counter()
            dist.all_reduce(tensor)
            _sync(device)
            elapsed = 1000 * (time.perf_counter() - start)
            if iteration >= config["warmup"]:
                samples.append(elapsed)
        assert tensor[0].item() == dist.get_world_size()
        rows.append({"size_MB": megabytes, "mean_ms": statistics.mean(samples),
                     "std_ms": statistics.pstdev(samples), "samples_ms": samples})
    return rows


def _train(device, config):
    from cs336_basics.model import BasicsTransformerLM
    from cs336_basics.optimizer import AdamW
    sizes = {"tiny": (64, 256, 2, 4), "small": (768, 3072, 12, 12), "xl": (2560, 10240, 32, 32)}
    width, hidden, layers, heads = sizes[config["size"]]
    torch.manual_seed(42)
    with torch.device(device):
        raw = BasicsTransformerLM(10000, config["sequence"], width, layers, heads, hidden)
    parameter_count = sum(p.numel() for p in raw.parameters())
    mode = config["mode"]
    model = (FSDP(raw, compute_dtype=torch.bfloat16 if config["bf16"] else None) if mode == "fsdp"
             else NaiveDDP(raw, flatten=mode == "flat") if mode in ("naive", "flat") else DDP(raw))
    del raw
    optimizer = (ShardedOptimizer(model.parameters(), AdamW, lr=1e-4)
                 if mode == "sharded" else AdamW(model.parameters(), lr=1e-4))
    torch.manual_seed(100 + dist.get_rank())
    batch = config["batch"] // dist.get_world_size()
    x = torch.randint(10000, (batch, config["sequence"]), device=device)
    targets = torch.randint(10000, x.shape, device=device)
    _sync(device)
    initial_bytes = torch.cuda.memory_allocated(device)
    initial_peak = torch.cuda.max_memory_allocated(device)
    samples = []
    profiler = None
    if config["trace"]:
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                      torch.profiler.ProfilerActivity.CUDA])
    for iteration in range(config["warmup"] + config["steps"]):
        optimizer.zero_grad(set_to_none=True)
        dist.barrier()
        _sync(device)
        torch.cuda.reset_peak_memory_stats(device)
        if profiler and iteration == config["warmup"]:
            profiler.__enter__()
        start = time.perf_counter()
        with torch.profiler.record_function("forward"), torch.cuda.nvtx.range("forward"):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config["bf16"]):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
        with torch.profiler.record_function("backward"), torch.cuda.nvtx.range("backward"):
            loss.backward()
        _sync(device)
        sync_start = time.perf_counter()
        with torch.profiler.record_function("gradient_synchronization"), torch.cuda.nvtx.range("gradient_synchronization"):
            model.finish_gradient_synchronization()
        _sync(device)
        sync_ms = 1000 * (time.perf_counter() - sync_start)
        before_bytes = torch.cuda.memory_allocated(device)
        before_peak = torch.cuda.max_memory_allocated(device)
        with torch.profiler.record_function("optimizer_step"), torch.cuda.nvtx.range("optimizer_step"):
            optimizer.step()
        _sync(device)
        elapsed = 1000 * (time.perf_counter() - start)
        after_bytes = torch.cuda.memory_allocated(device)
        after_peak = torch.cuda.max_memory_allocated(device)
        if iteration == 0:
            first_step = {"before_optimizer_bytes": before_bytes, "before_optimizer_peak_bytes": before_peak,
                          "after_optimizer_bytes": after_bytes, "after_optimizer_peak_bytes": after_peak}
        if iteration >= config["warmup"]:
            samples.append({"step_ms": elapsed, "sync_ms": sync_ms, "before_optimizer_bytes": before_bytes,
                            "before_optimizer_peak_bytes": before_peak, "after_optimizer_bytes": after_bytes,
                            "after_optimizer_peak_bytes": after_peak, "loss": loss.item()})
        if profiler and iteration == config["warmup"]:
            profiler.__exit__(None, None, None)
            profiler.export_chrome_trace(f'{config["trace"]}.rank{dist.get_rank()}.json')
            profiler = None
        del logits, loss
    state_bytes = sum(v.numel() * v.element_size() for state in optimizer.state.values()
                      for v in state.values() if isinstance(v, torch.Tensor))
    return {"parameter_count": parameter_count, "initial_bytes": initial_bytes, "initial_peak_bytes": initial_peak,
            "local_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "optimizer_state_bytes": state_bytes, "first_step": first_step,
            "mean_step_ms": statistics.mean(row["step_ms"] for row in samples),
            "std_step_ms": statistics.pstdev(row["step_ms"] for row in samples),
            "mean_sync_ms": statistics.mean(row["sync_ms"] for row in samples), "samples": samples}


def run_benchmark(config):
    """Launch fresh ranks, writing JSON measurements after successful completion."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(config["world_size"], port, config), nprocs=config["world_size"], join=True)
    return json.loads(Path(config["output"]).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["collective", "train"])
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="nccl")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--mode", choices=["naive", "flat", "overlap", "sharded", "fsdp"], default="naive")
    parser.add_argument("--size", choices=["tiny", "small", "xl"], default="xl")
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4, help="Global batch size")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--trace", help="Optional Chrome trace path prefix")
    parser.add_argument("--output", default="results/distributed.json")
    config = vars(parser.parse_args())
    if config["batch"] % config["world_size"] and config["command"] == "train":
        parser.error("Global batch must divide evenly among ranks")
    run_benchmark(config)


if __name__ == "__main__":
    main()


def result_rows(path):
    """Flatten saved measurements into a notebook-friendly table."""
    data = json.loads(Path(path).read_text())
    rows = []
    for rank, values in enumerate(data["ranks"]):
        if isinstance(values, list):
            rows.extend({"world_size": data["world_size"], "rank": rank,
                         **{k: v for k, v in row.items() if k != "samples_ms"}} for row in values)
        else:
            rows.append({"mode": data["config"]["mode"], "rank": rank,
                         "step_ms": values["mean_step_ms"], "step_std_ms": values["std_step_ms"],
                         "exposed_sync_ms": values["mean_sync_ms"],
                         "initial_GiB": values["initial_bytes"] / 2**30,
                         "peak_GiB": max(s["after_optimizer_peak_bytes"] for s in values["samples"]) / 2**30,
                         "optimizer_state_GiB": values["optimizer_state_bytes"] / 2**30})
    return rows


def render_nsys_timeline(sqlite_path, output_path, title, detail_phase="backward"):
    """Render actual Nsight CUDA timestamps, retaining the report as source.

    This is an exported trace visualization, not a screenshot of the Nsight GUI.
    CUDA launch correlations assign asynchronous kernels to their CPU NVTX phase.
    """
    import sqlite3
    import matplotlib.pyplot as plt
    connection = sqlite3.connect(sqlite_path)
    pid = connection.execute("SELECT min(globalPid) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId=0").fetchone()[0]
    phases = connection.execute(
        "SELECT text,start,end FROM NVTX_EVENTS WHERE globalTid / 16777216 = ? "
        "AND text IN ('forward','backward','gradient_synchronization','optimizer_step') ORDER BY start",
        (pid >> 24,)).fetchall()
    phases = phases[-4:]
    launch = dict(connection.execute(
        "SELECT correlationId,start FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE globalTid / 16777216 = ?", (pid >> 24,)))
    kernels = connection.execute(
        "SELECT k.start,k.end,s.value,k.correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id=k.shortName WHERE globalPid=? ORDER BY k.start", (pid,)).fetchall()
    selected = []
    for start, end, name, correlation in kernels:
        cpu_start = launch.get(correlation, -1)
        phase = next((text for text, a, b in phases if a <= cpu_start <= b), None)
        if phase is not None:
            selected.append((start, end, name, phase))
    origin = min(row[0] for row in selected)
    finish = max(row[1] for row in selected)
    compute = [(a, b) for a, b, name, _ in selected if "nccl" not in name.lower()]
    communication = [(a, b) for a, b, name, _ in selected if "nccl" in name.lower()]
    overlap_ns = sum(max(0, min(b, d) - max(a, c)) for a, b in communication for c, d in compute)
    communication_ns = sum(b - a for a, b in communication)
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), constrained_layout=True)
    for axis in axes:
        axis.broken_barh([((a-origin)/1e6,(b-a)/1e6) for a,b in compute], (1.1, .7), facecolors="#3178a8")
        axis.broken_barh([((a-origin)/1e6,(b-a)/1e6) for a,b in communication], (.1, .7), facecolors="#e28224")
        axis.set_yticks([.45, 1.45], ["NCCL kernels", "Compute kernels"])
        axis.set_xlabel("Milliseconds from first GPU kernel of measured step")
        axis.grid(axis="x", alpha=.25)
    axes[0].set_xlim(0, (finish-origin)/1e6)
    # Zoom into a backward portion containing reductions, or the backward tail.
    backward = [(a,b) for a,b,_,phase in selected if phase == detail_phase]
    pivot = (backward[len(backward)//2][0] if any(phase == detail_phase and "nccl" in name.lower()
             for _,_,name,phase in selected) else max(b for a,b in backward))
    axes[1].set_xlim(max(0,(pivot-origin)/1e6-50), (pivot-origin)/1e6+100)
    axes[0].set_title(title + " — Nsight Systems CUDA trace export (rank 0)")
    axes[1].set_title(detail_phase.capitalize() + " detail; bars use recorded kernel start/end times")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    stats = {"communication_kernel_ms": communication_ns/1e6,
             "communication_overlapped_compute_ms": overlap_ns/1e6,
             "communication_overlap_fraction": overlap_ns / communication_ns if communication_ns else 0,
             "phase_gpu_span_ms": {phase: (max(b for a,b,n,p in selected if p==phase)-min(a for a,b,n,p in selected if p==phase))/1e6
                                    for phase in {p for a,b,n,p in selected}}}
    stats["phase_communication"] = {}
    for phase in {p for a,b,n,p in selected}:
        phase_comm = [(a,b) for a,b,n,p in selected if p == phase and "nccl" in n.lower()]
        comm_ns = sum(b-a for a,b in phase_comm)
        hidden_ns = sum(max(0, min(b,d)-max(a,c)) for a,b in phase_comm for c,d in compute)
        stats["phase_communication"][phase] = {"kernel_ms": comm_ns / 1e6,
                                             "overlapped_ms": hidden_ns / 1e6}
    Path(output_path).with_suffix(".json").write_text(json.dumps(stats, indent=2))
    connection.close()
    return stats
