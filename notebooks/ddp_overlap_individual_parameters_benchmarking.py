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
    # Overlapping DDP benchmarking

    同前两题 XL / FP32 / context512 / global batch4 / 2×A100 / AdamW，5 warmup+10测量。overlap 平均1691.23/1690.56ms/step，比naive1742ms快约2.9%，比flat1759ms快约3.8%；backward结束后的额外wait仅2.78/2.35ms，但这不是通信总耗时，因为大部分通信已在backward内发生。Nsight单独采样（1次预热+1次测量）中，naive NCCL kernel总时长85.91ms、与计算重叠0ms；overlap为128.74ms，其中124.60ms与计算重叠（96.8%），说明隐藏通信虽有效，资源竞争仍增加kernel时间。下面两张图由真实Nsight CUDA kernel开始/结束时间导出，不是Nsight桌面GUI截图；原始 `.nsys-rep` 可在Nsight打开核验。复现profile：`CUDA_VISIBLE_DEVICES=6,7 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -o results/ddp_overlap_trace uv run python -m cs336_systems.distributed_benchmark train --mode overlap --size xl --warmup 1 --steps 1`（naive相应修改）。
    """)
    return


@app.cell
def _(Path, mo):
    _root = Path(__file__).resolve().parents[1]
    from cs336_systems.distributed_benchmark import result_rows
    _data = result_rows(_root / 'results/ddp_overlap_individual_parameters_benchmarking.json')
    mo.ui.table(_data)
    return


@app.cell
def _():
    from cs336_systems.distributed_benchmark import run_benchmark
    run_benchmark
    return


@app.cell
def _(Path, mo):
    _root = Path(__file__).resolve().parents[1]
    mo.vstack([mo.image(_root / "results/ddp_naive_timeline.png"), mo.image(_root / "results/ddp_overlap_timeline.png")])
    return


if __name__ == "__main__":
    app.run()
