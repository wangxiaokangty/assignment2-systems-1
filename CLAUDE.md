# 核心原则

0. 我不是 stanford 的学生，所以希望用本地的 GPU
1. 核心实现按真实项目的方式分模块组织，可被 import、可测试、可复用。
2. **PDF 中的每一个 `Problem (xxx)` 对应一个 marimo notebook**（`notebooks/` 下），notebook 只负责
   *引用* 1 中的代码来解决该问题，并展示实际效果。
3. **notebook 里不写核心逻辑**。如果某段代码在 notebook 中超过“胶水代码”的程度（例如一个可复用的函数），
   就把它移到 1 中再 import。
4. `tests/adapters.py` 只做薄封装：每个 `run_*` 函数直接调用 1 中模块的实现，不在其中写逻辑。
5. 模型在 ./models 目录下
6. 不要新加测试，一次性脚本可以直接写在 marimo notebook 中
7. 代码尽量清晰，简洁，不要过度设计，修改/新增文件尽量少
8. 必要时可以用多 GPU 来并行处理任务，通过设置 CUDA_VISIABLE 环境变量
9. 每一个 problem 对应一个 commit
10. 一次性的 python 命令尽量使用 uvx