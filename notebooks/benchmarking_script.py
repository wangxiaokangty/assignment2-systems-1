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
    # Benchmarking script

    `cs336_systems.benchmarking` 初始化真实 reference Transformer、随机 token/target，并在每个阶段后 synchronize；FP32 参数，batch=4，vocab=10000，context=512，reference AdamW。表中均值/标准差来自 10 次测量，warmup 分别 0、1、2、5；forward 包含模型前向，loss 单列，backward 与 optimizer 分开计时，total 不含 zero_grad。

    warmup=0 的首次迭代包含 CUDA 库和算子初始化，因此均值和方差往往更大；1–2 步能消除部分冷启动，但缓存、时钟与 Adam 状态/内存池尚未完全稳定。10B 若超过本地 80 GiB 会明确列为 OOM，不把小模型结果冒充完整配置。

    复现：`CUDA_VISIBLE_DEVICES=0 uv run python -m cs336_systems.benchmarking --size small --warmup 5 --steps 10`。随机初始化不需要下载 ./models 下的预训练权重。
    """)
    return


@app.cell
def _(Path, json, pd):
    result_files = sorted(Path("results").glob("benchmarking_script*.json"))
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
def _(mo):
    from cs336_systems.benchmarking import ModelConfig, MODEL_SIZES, benchmark
    MODEL_SIZES
    return

if __name__ == "__main__":
    app.run()
