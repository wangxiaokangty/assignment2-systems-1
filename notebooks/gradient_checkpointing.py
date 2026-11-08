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
    # Memory-optimal gradient checkpointing

    (a) 二分递归 checkpoint 每层递归保留 O(1) 个边界激活，得到 O(log N) activation memory 和 O(N log N) compute；对应模块中的 `recursive_checkpoint`。若“忽略 compute”允许任意重算，这并非全局内存下界：每次反传一个 block 前，从最初输入以 no_grad 重算到该 block 的输入，单独重建该 block 的图并反传，然后释放，可只保留 O(1) 个激活和一个 block 的 residuals，代价 O(N²) 次 block 计算；参数及梯度存储不计入 activation memory。

    (b) 无嵌套时将连续 k 层分组，峰值近似 ceil(N/k) A + k R，其中 A 是边界激活，R 是单层 residuals，因此 k≈sqrt(NA/R)，取合法整数并实测相邻值。这里 A=80 MiB，参考资料中单层 R 已约 3.6 GiB，故最佳整数通常 k=1；k=0 表示不 checkpoint，作为“更小”的边界基线，另测 k=2/4。此实验真实使用本地 A100、完整 xl，未将论文中的 3651.31 MiB 冒充本地测量。

    checkpoint 用 `use_reentrant=False`，闭包绑定当前分组以避免反向时错误地重算最后一组；下表展示峰值显存与时间的实测折中。
    """)
    return


@app.cell
def _(Path, json, pd):
    result_files = sorted(Path("results").glob("gradient_checkpointing*.json"))
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
    import inspect
    from cs336_systems.checkpointing import recursive_checkpoint, checkpoint_blocks
    mo.md("```python\n" + inspect.getsource(recursive_checkpoint) + "\n" + inspect.getsource(checkpoint_blocks) + "\n```")
    return


@app.cell
def _():
    import torch
    from cs336_systems.benchmarking import ModelConfig, build_model
    from cs336_systems.checkpointing import checkpoint_model
    torch.manual_seed(7)
    _cfg = ModelConfig(d_model=32, d_ff=64, num_layers=4, num_heads=4, vocab_size=64, context_length=16, batch_size=2)
    _eager = build_model(_cfg, 'cpu')
    _checked = build_model(_cfg, 'cpu')
    _checked.load_state_dict(_eager.state_dict())
    checkpoint_model(_checked, 2)
    _tokens = torch.randint(64, (2,16))
    _a, _b = _eager(_tokens), _checked(_tokens)
    torch.testing.assert_close(_a, _b)
    _a.square().mean().backward()
    _b.square().mean().backward()
    for _p, _q in zip(_eager.parameters(), _checked.parameters()):
        torch.testing.assert_close(_p.grad, _q.grad)
    "Reference vs checkpoint: logits and every parameter gradient match."
    return


if __name__ == "__main__":
    app.run()
