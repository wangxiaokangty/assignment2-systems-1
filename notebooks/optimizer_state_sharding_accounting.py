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
    # Optimizer state sharding accounting

    配置为2×A100、XL 3,406,809,600参数、FP32、global batch4、context512、AdamW，使用overlap DDP以控制变量。初始化实际占用两种方式均12.82GiB；第一次optimizer前峰值均26.71GiB、之后峰值未分片51.43GiB与分片38.74GiB；预热后的稳态训练峰值为52.09GiB与39.40GiB。参数和梯度各约12.69GiB；Adam一阶二阶状态从25.38GiB降到每卡12.69GiB，余量来自buffer、activation及临时张量，内存实测与预期减半optimizer状态吻合。

    稳态迭代从1691ms降到1618ms（约4.3%）；本地NVLink下新增参数广播成本小于减少一半Adam更新的节省，不能据此推断慢互联上也会提速。

    与[ZeRO论文](https://arxiv.org/pdf/1910.02054)的stage 1相比，本实现只对完整参数tensor分配owner，并保留全量FP32参数和梯度；完整DDP all-reduce之后又广播更新的参数，等长dtype时逻辑传输量约为3(N−1)P/N个元素，而ZeRO可采用reduce-scatter梯度加all-gather更新参数，约2(N−1)P/N，与普通DDP相同。论文混合精度方案还把FP32 master视作可分片optimizer state；本实验FP32全量参数仍复制，且按整tensor分配可能不均衡，因此内存节省不应直接套用论文的4倍数字。复现：`CUDA_VISIBLE_DEVICES=6,7 uv run python -m cs336_systems.distributed_benchmark train --mode sharded --size xl --output results/optimizer_state_sharding_accounting.json`。
    """)
    return


@app.cell
def _(Path, mo):
    _root = Path(__file__).resolve().parents[1]
    from cs336_systems.distributed_benchmark import result_rows
    _data = result_rows(_root / 'results/optimizer_state_sharding_accounting.json')
    mo.ui.table(_data)
    return


@app.cell
def _():
    from cs336_systems.distributed_benchmark import run_benchmark
    run_benchmark
    return


if __name__ == "__main__":
    app.run()
