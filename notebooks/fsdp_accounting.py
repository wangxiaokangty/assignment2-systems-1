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
    # FSDP accounting

    对于P个FP32参数、N个rank，忽略norm、激活和gather buffer，optimizer-state sharding是8P+8P/N bytes，FSDP是16P/N，因此再省8P(1−1/N)。XL P=3,406,809,600、N=2时理论静态状态从38.07GiB降到25.38GiB，再省12.69GiB；实测训练峰值从39.40GiB降到32.96GiB（仅省6.44GiB），因为FSDP的新峰值发生在activation仍存活的backward，峰值是不同阶段的最大值，不能把静态state节省直接从旧峰值相减。FSDP构造先建立全量模型再切片，初始化瞬时峰值13.06GiB，但切片后实际占用6.41GiB；稳态权重6.35GiB、Adam状态12.69GiB。

    同标准XL FP32配置5次warmup+10次测量，FSDP平均1623ms/step。独立Nsight记录的forward GPU跨度721.58ms，其中all-gather kernels合计136.80ms，94.65ms与计算重叠（约69.2%）；仍有42.15ms通信没有计算覆盖，因此不能说所有gather都及时完成。下一层预取有收益，但首层和部分较短层仍暴露通信；backward当前按需gather而非预取，留有进一步优化空间。下图来自真实Nsight导出的CUDA时间线（非桌面GUI截图），原始 `results/fsdp_trace.nsys-rep` 可在Nsight打开；profile引入开销，721.58ms不等同于未插桩forward基准。复现：`CUDA_VISIBLE_DEVICES=4,5 uv run python -m cs336_systems.distributed_benchmark train --mode fsdp --size xl --output results/fsdp_accounting.json`。
    """)
    return


@app.cell
def _(Path, mo):
    _root = Path(__file__).resolve().parents[1]
    from cs336_systems.distributed_benchmark import result_rows
    _data = result_rows(_root / 'results/fsdp_accounting.json')
    mo.ui.table(_data)
    return


@app.cell
def _():
    from cs336_systems.distributed_benchmark import run_benchmark
    run_benchmark
    return


@app.cell
def _(Path, mo):
    mo.image(Path(__file__).resolve().parents[1] / "results/fsdp_timeline.png")
    return


if __name__ == "__main__":
    app.run()
