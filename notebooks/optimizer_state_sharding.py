import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import json

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Optimizer state sharding

    核心实现：`cs336_systems.sharded_optimizer.ShardedOptimizer(params, optimizer_cls, **kwargs)`。按 tensor 元素数贪心分配 owner，各 rank 仅为自己负责的完整参数创建优化器状态，更新后广播给其他 rank。支持参数组与训练期间 `add_param_group`，保持组超参数与 scheduler 修改同步；checkpoint 采用每个 rank 独立保存的局部 optimizer state。运行：`CUDA_VISIBLE_DEVICES=4,5 uv run pytest tests/test_sharded_optimizer.py`。已有 tests/test_sharded_optimizer.py 的普通与 tied-weight 两项已通过；内存与速度实测见 accounting notebook。
    """)
    return


@app.cell
def _():
    from cs336_systems.sharded_optimizer import ShardedOptimizer
    ShardedOptimizer
    return


if __name__ == "__main__":
    app.run()
