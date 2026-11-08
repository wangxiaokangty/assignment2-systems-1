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
    # Benchmarking mixed precision

    在 FP16 autocast 中，FP32 参数保持 FP32；fc1 输出 FP16、LayerNorm 输出 FP32、logits FP16，显式采用交叉熵时 loss FP32，FP32 参数梯度也是 FP32。下面在本地 GPU 上通过 hooks 实测，而非只依赖规则推断。

    LayerNorm 的均值、方差归约以及减均值、平方容易积累舍入误差，FP16 还可能溢出。BF16 的动态范围更宽，但有效精度更低，仍然值得以 FP32 累加统计量；动态范围改善不等于归约误差消失。

    表中每种模型均为 batch=4/context=512，5 warmup + 10 measurements。BF16 autocast 使用 Tensor Cores，矩阵乘法加速，但参数/梯度仍 FP32、归约和内存流量仍存在，不能期待整个模型按峰值 FLOPS 比例加速。OOM 表示该完整配置确实在本地 GPU 上尝试失败。
    """)
    return


@app.cell
def _(Path, json, pd):
    result_files = sorted(Path("results").glob("benchmarking_mixed_precision*.json"))
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
def _():
    from cs336_systems.profiling import mixed_precision_dtypes
    dtype_observations = mixed_precision_dtypes()
    dtype_observations
    return


if __name__ == "__main__":
    app.run()
