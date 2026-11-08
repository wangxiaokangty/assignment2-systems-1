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
    # Distributed communication on one node

    本地 A100-SXM4-80GB，NCCL，FP32，2/4/6 个 GPU，各自独立进程。使用十进制 MB（1MB=1,000,000 bytes，1GB=1,000MB），每个大小5次预热与10次测量；每步 barrier 后同步设备，计时 all_reduce 直至 CUDA synchronize 完成，保存各 rank 全部样本。1MB 时固定启动开销较明显，大消息增长趋于带宽受限；更多 rank 需要传输更大的 ring 比例和更多通信阶段，因此相同 tensor 通常更慢。下表展示每个rank均值和标准差，可比较慢 rank；复现：`CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 uv run python -m cs336_systems.distributed_benchmark collective --world-size 6 --output results/distributed_communication_single_node_6.json`（2/4卡相应修改）。
    """)
    return


@app.cell
def _(Path, mo):
    from cs336_systems.distributed_benchmark import result_rows, run_benchmark
    _root = Path(__file__).resolve().parents[1]
    mo.ui.table([_row for _world in (2, 4, 6)
                 for _row in result_rows(_root / f"results/distributed_communication_single_node_{_world}.json")])
    return


if __name__ == "__main__":
    app.run()
