import marimo
app = marimo.App(width="full")

@app.cell
def _():
    import json
    from pathlib import Path
    import pandas as pd
    import marimo as mo
    from cs336_systems.flash_benchmark import benchmark_flash
    result = json.loads((Path(__file__).resolve().parents[1]/'results/flash_benchmarking.json').read_text())
    mo.md(f"# FlashAttention benchmarking\nLocal hardware: **{result['gpu']}**, rather than the assignment's B200. Results therefore characterize this local GPU only. Batch 1; causal attention; all powers of two from N=128 through 65536, d=16 through 128, BF16 and FP32. `triton.testing.do_bench` uses 10 ms warmup and 30 ms repetition budgets; inputs and upstream gradients are allocated outside timing.")
    return pd, mo, result

@app.cell
def _(pd, mo, result):
    mo.ui.table(pd.DataFrame(result['rows']))
    return

@app.cell
def _(mo):
    mo.md("""
    Rerun the complete sweep: `CUDA_VISIBLE_DEVICES=2 uv run python -m cs336_systems.flash_benchmark`.
    Forward runs with autograd enabled; backward reuses its saved graph, while end-to-end rebuilds
    the graph each iteration. The flash rows use the required Triton forward plus compiled PyTorch
    backward. Dense backward recomputation still needs quadratic temporary memory; an OOM in backward
    is reported separately so a successful long-sequence forward measurement remains visible.

    At N=16384,d=128,BF16, forward improves from 5.20 ms (PyTorch) to 0.984 ms (flash),
    approximately 5.3× faster. However the full step slows from 10.65 to 25.00 ms: the required
    backward casts intermediate computations to FP32 and costs 24.00 ms versus 5.57 ms for
    the BF16 dense baseline. At N=65536,d=16, FP32 dense forward succeeds (146.08 ms) but
    backward runs out of memory; flash completes forward/backward in 160.79 ms combined.

    The tiled forward avoids global N×N scores/probabilities and benefits from BF16 tensor cores.
    FP32 uses IEEE dot products for numerical accuracy on the A100, sacrificing tensor-core throughput.
    A faster forward does not imply an equally fast training step: compiled backward reconstructs dense
    probabilities. `FlashAttentionTritonFull` supplies the optional Algorithm 2 tiled backward for the
    long-context training task, but is intentionally separate from this required partially-Triton comparison.
    """)
    return

if __name__ == '__main__':
    app.run()
