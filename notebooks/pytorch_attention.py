import marimo

app = marimo.App(width="full")

@app.cell
def _():
    import json
    from pathlib import Path
    import marimo as mo
    import pandas as pd
    from cs336_systems.attention_benchmark import benchmark_attention
    root = Path(__file__).resolve().parents[1]
    result = json.loads((root / 'results/pytorch_attention.json').read_text())
    mo.md(f"# PyTorch attention\nMeasured on {result['gpu']}; float32, batch 8, 5 warmups and 100 synchronized forward/backward passes. Rerun with `uv run python -m cs336_systems.attention_benchmark --output results/pytorch_attention.json`.")
    return mo, pd, result

@app.cell
def _(mo, pd, result):
    mo.ui.table(pd.DataFrame(result['rows']))
    return

@app.cell
def _(mo):
    mo.md(r"""
    Scores/probabilities require $4BN^2$ bytes each, while each Q/K/V/output requires $4BNd$ bytes.
    Autograd retains probabilities plus Q/K/V; the saved activations therefore grow quadratically
    with sequence length. The memory counter also includes allocator-visible library workspaces.
    At B=8, N=16384, d=16, one attention matrix is 8 GiB and Q/K/V together are 24 MiB.
    Backward may simultaneously hold probabilities, dP and dS (about 24 GiB), plus linear-size inputs
    and gradients. None of the required configurations exhausts this 80 GiB GPU; thus there is no
    observed smallest OOM within the requested sweep. N=32768 would already require 32 GiB per matrix
    and approximately 96 GiB for those three matrices, exceeding capacity (an estimate, not a measurement).

    Online softmax tiles remove the need to store the complete probability matrix. Save a per-row
    log-sum-exp and recompute probabilities during backward; a tiled backward is needed to keep its
    transient memory linear too. Checkpointing alone reduces retained tensors but does not eliminate
    the peak quadratic temporary memory of a dense attention implementation.
    """)
    return

if __name__ == "__main__":
    app.run()
