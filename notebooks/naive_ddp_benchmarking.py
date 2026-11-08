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
    # Naïve DDP benchmarking

    本地 2×A100-SXM4-80GB，XL（d_model=2560，d_ff=10240，32层，32 heads），vocab=10000，context=512，global batch=4，每 rank batch=2，FP32，assignment 1 AdamW；5 次预热，10 次测量，固定随机种子。两个 rank 的平均迭代分别为 1741.65/1742.36 ms，梯度同步为 103.30/101.60 ms，约占 5.9%。每阶段用 CUDA synchronize 测量完成时间；rank 0/1 的逐步原始值均保存在下表。复现：`CUDA_VISIBLE_DEVICES=6,7 uv run python -m cs336_systems.distributed_benchmark train --mode naive --size xl --output results/naive_ddp_benchmarking.json`。
    """)
    return


@app.cell
def _(Path, mo):
    _root = Path(__file__).resolve().parents[1]
    from cs336_systems.distributed_benchmark import result_rows
    _data = result_rows(_root / 'results/naive_ddp_benchmarking.json')
    mo.ui.table(_data)
    return


@app.cell
def _():
    from cs336_systems.distributed_benchmark import run_benchmark
    run_benchmark
    return


if __name__ == "__main__":
    app.run()
