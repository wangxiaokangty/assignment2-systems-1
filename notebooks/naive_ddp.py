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
    # Naïve DDP

    核心实现：`cs336_systems.distributed.NaiveDDP`。构造时广播 rank 0 的参数及 buffer；每个 rank 使用不相交的等大 batch，backward 后逐参数 all-reduce 并除以 world size，再做优化器更新。共享参数由 `module.parameters()` 去重，冻结参数不会生成梯度。可复现实验命令：`CUDA_VISIBLE_DEVICES=4,5 uv run python -m cs336_systems.distributed_benchmark train --mode naive --size xl`。已通过临时入口把仓库原有 `tests/test_ddp.py` 的 get_ddp 绑定到 NaiveDDP，原有 ToyModel 和 ToyModelWithTiedWeights 两个测试均通过（没有新增测试）；完整结果见相邻 benchmarking notebook。
    """)
    return


@app.cell
def _():
    from cs336_systems.distributed import NaiveDDP
    NaiveDDP
    return


if __name__ == "__main__":
    app.run()
