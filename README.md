# CS336 Spring 2026 Assignment 2: Systems

For a full description of the assignment, see the assignment handout at
[cs336_assignment2_systems.pdf](./cs336_assignment2_systems.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

This directory is organized as follows:

- [`./cs336-basics`](./cs336-basics): directory containing a module
  `cs336_basics` and its associated `pyproject.toml`. This module contains the staff 
  implementation of the language model from assignment 1. If you want to use your own 
  implementation, you can replace this directory with your own implementation.
- [`./cs336_systems`](./cs336_systems): This module contains reusable
  implementations for profiling, attention kernels, distributed training,
  sharded optimization, parallelism accounting, and the complete training benchmark.

Visually, it should look something like:

``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   ├── __init__.py
│   └── ... other files in the cs336_basics module, taken from assignment 1 ...
├── cs336_systems  # TODO(you): code that you'll write for assignment 2 
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 2 ...
├── README.md
├── pyproject.toml
├── notebooks/  # One marimo notebook per PDF problem
└── results/    # Recorded local measurements and figures
```

If you would like to use your own implementation of assignment 1, replace the `cs336-basics`
directory with your own implementation, or edit the outer `pyproject.toml` file to point to your
own implementation.

0. We use `uv` to manage dependencies. You can verify that the code from the `cs336-basics`
package is accessible by running:

```sh
$ uv run python
Using CPython 3.13.13
Creating virtual environment at: /path/to/uv/env/dir
      Built cs336-systems @ file:///path/to/systems/dir
      Built cs336-basics @ file:///path/to/basics/dir
Installed 78 packages in 168ms
Python 3.13.13 (main, Apr  7 2026, 20:49:46) [Clang 22.1.1 ] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import cs336_basics
...
```

`uv run` installs dependencies automatically as dictated in the `pyproject.toml` file.

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a gzipped tarball with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.

## 本地完成版

PDF 的 27 个 `Problem (name)` 分别对应 `notebooks/name.py`。核心实现位于
`cs336_systems/`，`tests/adapters.py` 仅调用这些实现；`results/` 保存真实实验数据，
`models/` 保存本地模型配置、显存快照和 profiler 原始文件。

```bash
uv sync
uv run marimo edit notebooks/flash_forward.py
CUDA_VISIBLE_DEVICES=0,1 uv run pytest tests
uv run marimo check notebooks
```

实验使用本机 A100-SXM4-80GB，PDF 中的 B200 性能要求不能直接与这些计时比较。
所有实验的尺寸、精度、预热和采样次数见结果 JSON 和对应 notebook；OOM 记录为
实验结果，不填造计时。通信推导 notebook 的数值表是公式计算，不是硬件测量。
运行 notebook 默认展示保存的结果；昂贵实验由显式按钮或所列命令启动。
CUDA 的环境变量拼写是 `CUDA_VISIBLE_DEVICES`。

模块按用途组织：`benchmarking.py` / `profiling.py` / `checkpointing.py` 处理模型
基准与显存分析；`attention.py` 处理自写 PyTorch/Triton attention；
`distributed.py` / `fsdp.py` / `sharded_optimizer.py` 处理分布式训练；
`parallelism.py` 计算通信成本；`leaderboard.py` 提供双卡完整训练步。
每道题有独立 git commit，便于逐题审阅。

完整 leaderboard 本地复现（需要两张空闲 80GB GPU）：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python -m cs336_systems.leaderboard --steps 3
```

先用 `--smoke --output results/leaderboard_smoke.json` 可快速检查运行环境。
该命令不会向外部排行榜提交结果。
