import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import json
    from pathlib import Path
    import pandas as pd

    return Path, json, mo, pd


@app.cell
def _(mo):
    mo.md("""
    # Memory profiling

    真实测量 xl（32 layers，d_model=2560，32 heads，d_ff=10240），batch=4，context=128/2048，FP32 与 BF16 autocast；记录 `max_memory_allocated`，不是显存 reserved 或 nvidia-smi。snapshot 由 PyTorch `_record_memory_history/_dump_snapshot` 生成，存于 `models/profiles/`，可在 memory_viz 打开。

    推理中每层临时激活随计算释放，呈重复尖峰；训练前向保留 residuals，形成逐层上升，反向释放激活并产生参数梯度，AdamW 首次调用再分配两份状态。OOM 不代表零显存：该组合在 80 GiB A100 上无法完成完整训练，保留错误原文。

    单个 FP32 residual-stream tensor 大小为 4 × S × 2560 × 4 / 1024² MiB，即 S=128 时 5 MiB，S=2048 时 80 MiB。长上下文的 attention scores/exp/probabilities 是 [4,32,S,S]，S=2048 时每份 FP32 为 2048 MiB；参数矩阵和 FFN 中间激活也是较大的分配。BF16 减少部分激活/GEMM 输出，但 FP32 参数、梯度、Adam states 与显式 FP32 loss 仍保留。

    下面列出实际测量、保存张量 inventory 和PyTorch memory_viz 的真实浏览器截图（forward: S=2048，train: S=128）；同时附 Nsight Systems 2025.6.3 的真实 GUI 截图（xl S=128），原始二进制报告另存本地。
    """)
    return


@app.cell
def _(Path, json, pd):
    result_files = sorted(Path("results").glob("memory_profiling*.json"))
    records = [json.loads(path.read_text()) for path in result_files]
    rows = []
    for record, path in zip(records, result_files):
        row = {"file": path.name, **record.get("config", {}), "precision": record.get("precision"),
               "mode": record.get("mode"), "warmup": record.get("warmup"),
               "checkpoint_size": record.get("checkpoint_size"), "status": record.get("status", "ok"),
               **record.get("mean", {})}
        row.update({key+"_std": value for key, value in record.get("std", {}).items()})
        row["peak_GiB"] = (record.get("peak_allocated_bytes") or 0)/1024**3
        rows.append(row)
    pd.DataFrame(rows)
    return


@app.cell
def _(Path, mo):
    _images = sorted(Path("results").glob("memory_profiling_*.png"))
    mo.vstack([mo.image(str(_path)) for _path in _images])
    return


@app.cell
def _(mo):
    from cs336_systems.profiling import SavedTensorInventory
    from cs336_systems.benchmarking import benchmark
    mo.md(f"核心实现：`{benchmark.__module__}.benchmark(snapshot=...)`；去重保存张量计数：`{SavedTensorInventory.__name__}`。")
    return


@app.cell
def _(Path, json, mo, pd):
    _inventory = json.loads(Path("results/memory_profiling_block_inventory.json").read_text())
    mo.vstack([
        mo.md(f"单个 xl block（S=2048）保存的去重非参数 storage：{_inventory['saved_storage_bytes']/1024**2:.2f} MiB。参数梯度 {_inventory['parameter_gradient_bytes']/1024**2:.3f} MiB，加输入梯度 {_inventory['input_gradient_bytes']/1024**2:.1f} MiB。参数量公式 4D²+3DDff+2D，与直接累加 p.grad.numel() 一致。前向分配增量与反向净变化不同，是因为残差释放和梯度生成同时发生；保留的输出、输入和 view 共享 storage 也必须区分。下表按 producer 分组：view 是共享存储的来源标记，不应误称独立算子分配。"),
        pd.DataFrame(_inventory['saved_by_producer']),
        pd.DataFrame(_inventory['storages'][:10]),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Nsight 显存口径与复现

    Nsight 截图的 `Memory usage` 行记录 CUDA allocation 层的变化，PyTorch caching allocator 释放张量时未必 cudaFree，故它不能独自证明某个 residual 已释放；capture 从 measurement 开始，模型初始化已经发生。上方实际 PyTorch active-memory 时间线和按 storage 去重的 saved_tensors_hooks 才用于 active tensor 的量化。以 S=2048 的单层计，保存张量约 6503.7 MiB，参数梯度 400.0195 MiB，输入梯度 80 MiB；前五类来源与比例在 inventory 表中逐项列出，最大的 view storage 包含 attention probabilities 与 FFN/投影输入。

    ```bash
    NSYS_NVTX_PROFILER_REGISTER_ONLY=0 CUDA_VISIBLE_DEVICES=1 nsys profile --force-overwrite=true --sample=none --trace=cuda,nvtx --cuda-memory-usage=true --capture-range=nvtx --nvtx-capture=measurement --capture-range-end=stop --output=models/profiles/memory_profiling_xl_128_nsys uv run python -m cs336_systems.benchmarking --size xl --context-length 128 --mode train --warmup 0 --steps 1 --annotate-attention --output results/memory_profiling_xl_128_nsys.json
    CUDA_VISIBLE_DEVICES=1 uv run python -m cs336_systems.benchmarking --size xl --context-length 2048 --mode forward --warmup 0 --steps 1 --snapshot models/profiles/memory_profiling_2048_fp32_forward.pickle
    uv run python -m torch.cuda._memory_viz trace_plot models/profiles/memory_profiling_2048_fp32_forward.pickle -o models/profiles/memory_profiling_forward.html
    ```

    内存实验刻意保留第一次 AdamW 的状态创建（warmup=0），与前面吞吐 benchmark 的稳态 warmup=5 不混淆。使用 BF16 autocast 的结果仍包含 FP32 参数和优化器状态，长上下文完整训练在本机80 GiB上OOM的组合不会给出虚构峰值。
    """)
    return


if __name__ == "__main__":
    app.run()
