import marimo
app = marimo.App(width="full")

@app.cell
def _():
    import marimo as mo
    import torch
    from cs336_systems.attention import attention, FlashAttentionPyTorch, FlashAttentionTriton
    mo.md("# FlashAttention-2 forward\n64×64 PyTorch online softmax reference and a fused Triton kernel (32 query rows × 64 keys), with per-row float32 maxima, denominators and output accumulators. Triton launches `(ceil(Nq/32), batch)`; causal blocks beyond the diagonal are skipped. Inputs may be noncontiguous; the wrapper normalizes their layout.")
    return torch, attention, FlashAttentionPyTorch, FlashAttentionTriton

@app.cell
def _(torch, attention, FlashAttentionPyTorch, FlashAttentionTriton):
    torch.manual_seed(0)
    q, k, v = [torch.randn(4,128,64,device='cuda',requires_grad=True) for _ in range(3)]
    measurements = []
    for impl in [FlashAttentionPyTorch, FlashAttentionTriton]:
        for causal in [False, True]:
            out = impl.apply(q,k,v,causal)
            ref = attention(q,k,v,causal)
            measurements.append(dict(implementation=impl.__name__, causal=causal, max_absolute_error=(out-ref).abs().max().item(), saved_bytes=sum(t.numel()*t.element_size() for t in out.grad_fn.saved_tensors)))
    measurements
    return

if __name__ == '__main__':
    app.run()
