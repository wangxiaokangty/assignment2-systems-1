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
    # DDP with overlapping individual parameters

    核心实现：`cs336_systems.distributed.DDP`。使用 `register_post_accumulate_grad_hook` 在梯度累积完成时启动异步 all-reduce，因此 tied weights 只同步一次；`finish_gradient_synchronization()` 等待所有 work 后清空句柄，随后可以安全调用 optimizer.step。初始参数和 buffers 来自 rank 0。现有 tests/test_ddp.py 的普通与 tied-weight 两项已通过，最终完整仓库14项测试也全部通过。运行：`CUDA_VISIBLE_DEVICES=4,5 uv run pytest tests/test_ddp.py`。
    """)
    return


@app.cell
def _():
    from cs336_systems.distributed import DDP
    DDP
    return


if __name__ == "__main__":
    app.run()
