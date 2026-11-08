import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import json

    return Path, mo


@app.cell
def _(mo):
    mo.md("""
    # Minimal DDP with flat gradients

    复用仓库原有 DDP 两个测试（普通模型和 tied weights），flat 版本均通过。

    同 naive 基准：本地 2×A100，XL、FP32、global batch=4、context=512、AdamW、5 warmup+10测量。将所有梯度合并为一个 tensor 后，平均迭代 1758.15/1758.78 ms，包含 flatten/copy-back 的同步阶段 120.67/118.81 ms，比逐参数版本（1742ms、103ms）略慢：此大模型单个矩阵本就较大，额外约12.69GiB的扁平缓冲和拷贝成本超过减少调用的收益。首次默认缓存分配发生碎片 OOM；成功运行使用 `PYTORCH_ALLOC_CONF=expandable_segments:True`，这不改变数学行为。复现：`CUDA_VISIBLE_DEVICES=6,7 PYTORCH_ALLOC_CONF=expandable_segments:True uv run python -m cs336_systems.distributed_benchmark train --mode flat --size xl --output results/minimal_ddp_flat_benchmarking.json`。
    """)
    return


@app.cell
def _(Path, mo):
    _root = Path(__file__).resolve().parents[1]
    from cs336_systems.distributed_benchmark import result_rows
    _data = result_rows(_root / 'results/minimal_ddp_flat_benchmarking.json')
    mo.ui.table(_data)
    return


@app.cell
def _():
    from cs336_systems.distributed_benchmark import run_benchmark
    run_benchmark
    return


if __name__ == "__main__":
    app.run()
