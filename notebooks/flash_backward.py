import marimo
app = marimo.App(width="full")

@app.cell
def _():
    import torch
    import marimo as mo
    from cs336_systems.attention import attention, FlashAttentionPyTorch, FlashAttentionTriton
    mo.md(r"""# FlashAttention-2 backward
    The reusable `flash_backward` function is compiled by `torch.compile`. It reconstructs
    $P=\exp(QK^T/\sqrt d-L)$ and uses $D=\mathrm{rowsum}(O\odot dO)$, then
    $dS=P\odot(dOV^T-D)$ to produce dQ, dK and dV. Accumulation is float32.
    This required implementation stores only Q/K/V/O/L across forward and backward, but recomputes
    a full quadratic matrix during backward. Consequently it does not yet have a linear peak-memory bound.
    The optional `FlashAttentionTritonFull` used for long-context training tiles backward too.
    """)
    return torch, attention, FlashAttentionPyTorch, FlashAttentionTriton

@app.cell
def _(torch, attention, FlashAttentionPyTorch, FlashAttentionTriton):
    torch.manual_seed(0)
    q,k,v = [torch.randn(4,128,64,device='cuda',requires_grad=True) for _ in range(3)]
    do = torch.randn_like(q)
    errors = []
    for impl in [FlashAttentionPyTorch,FlashAttentionTriton]:
        for causal in [False,True]:
            expected = torch.autograd.grad(attention(q,k,v,causal),(q,k,v),do)
            actual = torch.autograd.grad(impl.apply(q,k,v,causal),(q,k,v),do)
            errors.append(dict(implementation=impl.__name__,causal=causal, **{name:(a-b).abs().max().item() for name,a,b in zip(['dQ_error','dK_error','dV_error'],actual,expected)}))
    errors
    return

if __name__ == '__main__':
    app.run()
