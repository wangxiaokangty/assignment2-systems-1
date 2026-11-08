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
    # Fully-sharded data parallel

    核心实现：`cs336_systems.fsdp.FSDP`，未调用 PyTorch 内置 FSDP。Linear/Embedding 权重按展平后的连续区间切片，尾部 padding 支持不能整除的维度；norm/bias 保持复制。自定义 autograd 在 forward 临时 all-gather 并释放完整权重，backward 重建权重和 reduce-scatter 梯度；FP32 master 和 optimizer states，compute_dtype 可用 FP16/BF16。学习真实执行顺序后最多预取下一层，当前层 i 的计算与 i+1 的 gather 重叠，符合 i+1 的 gather 不早于 i-1 完成的限制。已有 tests/test_fsdp.py 的 FP32/FP16 correctness、gradient_sync 四项已在本地两卡通过（20.30秒）；Gloo 使用等价 all-reduce+slice 回退，GPU benchmark 用 NCCL 真正 reduce-scatter。当前限制：不支持不同层绑定同一个权重，也不支持稀疏 embedding；FSDP 用于本实验未绑权重的 BasicsTransformerLM。
    """)
    return


@app.cell
def _():
    from cs336_systems.fsdp import FSDP
    FSDP
    return


if __name__ == "__main__":
    app.run()
