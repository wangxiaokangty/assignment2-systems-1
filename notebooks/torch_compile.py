import marimo
app = marimo.App(width="full")

@app.cell
def _():
    import json
    from pathlib import Path
    import pandas as pd
    import marimo as mo
    from cs336_systems.attention_benchmark import benchmark_attention
    from cs336_systems.benchmarking import ModelConfig, benchmark
    root = Path(__file__).resolve().parents[1]
    eager = json.loads((root/'results/pytorch_attention.json').read_text())
    compiled = json.loads((root/'results/torch_compile_attention.json').read_text())
    mo.md(f"# torch.compile\nMeasurements on {eager['gpu']}. Compilation and 5 warmup iterations are excluded. Attention uses float32 and 100 iterations; backward reuses a graph with donated buffers disabled.")
    return json, pd, mo, root, eager, compiled

@app.cell
def _(pd, mo, eager, compiled):
    columns = ['d','n','forward_ms','backward_ms']
    comparison = pd.DataFrame(eager['rows'])[columns].merge(pd.DataFrame(compiled['rows'])[columns],on=['d','n'],suffixes=('_eager','_compiled'))
    comparison['forward_speedup'] = comparison.forward_ms_eager/comparison.forward_ms_compiled
    comparison['backward_speedup'] = comparison.backward_ms_eager/comparison.backward_ms_compiled
    mo.ui.table(comparison)
    return

@app.cell
def _(json, pd, mo, root):
    model_results = [json.loads((root/f'results/torch_compile_model_{kind}.json').read_text()) for kind in ['eager','compiled']]
    mo.ui.table(pd.DataFrame([dict(compiled=r['compile_model'], **r['mean']) for r in model_results]))
    return

@app.cell
def _(mo):
    mo.md("""
    For attention at N=16384,d=128, compilation reduces forward latency from 81.3 to 70.6 ms
    and backward from 164.2 to 137.9 ms. Gains are smaller at short sequences where Python,
    launch and synchronization overhead dominate. Compilation fuses pointwise operations but
    retains the dense attention matrix, so saved memory still scales quadratically.

    The Transformer comparison uses the complete small model (12 layers, width 768, FFN 3072,
    12 heads, vocabulary 10000), batch 4, context 512, BF16 autocast and FP32 optimizer states.
    Both runs use 5 warmups and 10 measured steps. `forward_ms` measures the forward phase with
    autograd enabled; `total_ms` includes loss, backward and AdamW. The model is compiled as a whole;
    the loss and optimizer remain outside that compiled module. Rerun through
    `uv run python -m cs336_systems.benchmarking --size small --mode train --precision bf16`,
    adding `--compile-model` for the compiled variant and `--output` to save its results.
    """)
    return

if __name__ == '__main__':
    app.run()
