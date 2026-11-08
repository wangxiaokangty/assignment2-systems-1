"""Explicit scaled dot-product attention and memory-efficient tiled attention."""

import torch
import triton
import triton.language as tl


def attention(q, k, v, is_causal=False):
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    if is_causal:
        mask = torch.arange(q.shape[-2], device=q.device)[:, None] < torch.arange(k.shape[-2], device=q.device)[None, :]
        scores = scores.masked_fill(mask, float("-inf"))
    return scores.softmax(-1) @ v


class FlashAttentionPyTorch(torch.autograd.Function):
    """Online softmax reference, with FP32 accumulators and 64 by 64 tiles."""

    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        output = torch.empty_like(q)
        lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
        for start in range(0, q.shape[-2], 64):
            qi = q[..., start : start + 64, :].float()
            m = torch.full(qi.shape[:-1], -float("inf"), device=q.device)
            ell = torch.zeros_like(m)
            acc = torch.zeros_like(qi)
            for j in range(0, k.shape[-2], 64):
                scores = qi @ k[..., j : j + 64, :].float().transpose(-1, -2) / q.shape[-1] ** 0.5
                if is_causal:
                    mask = torch.arange(start, start + qi.shape[-2], device=q.device)[:, None] < torch.arange(j, min(j + 64, k.shape[-2]), device=q.device)[None, :]
                    scores = scores.masked_fill(mask, -1e6)
                new_m = torch.maximum(m, scores.amax(-1))
                alpha = torch.exp(m - new_m)
                p = torch.exp(scores - new_m[..., None])
                ell = alpha * ell + p.sum(-1)
                acc = alpha[..., None] * acc + p @ v[..., j : j + 64, :].float()
                m = new_m
            output[..., start : start + 64, :] = (acc / ell[..., None]).to(q.dtype)
            lse[..., start : start + 64] = m + ell.log()
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, do):
        grads = flash_backward(*ctx.saved_tensors[:3], ctx.saved_tensors[3], do, ctx.saved_tensors[4], ctx.is_causal)
        return (*grads, None) if len(ctx.needs_input_grad) == 4 else grads


@triton.jit
def flash_fwd_kernel(Q, K, V, Out, L, NQ: tl.constexpr, NK: tl.constexpr, D: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr, is_causal: tl.constexpr):
    tile, batch = tl.program_id(0), tl.program_id(1)
    qp = tl.make_block_ptr(Q + batch * NQ * D, (NQ, D), (D, 1), (tile * BQ, 0), (BQ, D), (1, 0))
    kp = tl.make_block_ptr(K + batch * NK * D, (NK, D), (D, 1), (0, 0), (BK, D), (1, 0))
    vp = tl.make_block_ptr(V + batch * NK * D, (NK, D), (D, 1), (0, 0), (BK, D), (1, 0))
    op = tl.make_block_ptr(Out + batch * NQ * D, (NQ, D), (D, 1), (tile * BQ, 0), (BQ, D), (1, 0))
    q = tl.load(qp, boundary_check=(0, 1), padding_option="zero")
    m = tl.full((BQ,), -float("inf"), tl.float32)
    ell = tl.zeros((BQ,), tl.float32)
    acc = tl.zeros((BQ, D), tl.float32)
    rows = tile * BQ + tl.arange(0, BQ)
    end = NK
    if is_causal:
        end = tl.minimum(NK, (tile + 1) * BQ)
    for j in range(tl.cdiv(end, BK)):
        k = tl.load(kp, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(vp, boundary_check=(0, 1), padding_option="zero")
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * (D**-0.5)
        cols = j * BK + tl.arange(0, BK)
        s = tl.where(cols[None, :] < NK, s, -float("inf"))
        if is_causal:
            s = tl.where(rows[:, None] >= cols[None, :], s, -1e6)
        new_m = tl.maximum(m, tl.max(s, 1))
        alpha = tl.exp(m - new_m)
        p = tl.exp(s - new_m[:, None])
        ell = ell * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc=acc, input_precision="ieee")
        m = new_m
        kp = tl.advance(kp, (BK, 0))
        vp = tl.advance(vp, (BK, 0))
    tl.store(op, (acc / ell[:, None]).to(Out.dtype.element_ty), boundary_check=(0, 1))
    tl.store(L + batch * NQ + rows, m + tl.log(ell), rows < NQ)


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        output = torch.empty_like(q)
        lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
        n, d = q.shape[-2:]
        if d < 16 or d & (d - 1):
            raise ValueError("Triton attention requires a power-of-two head dimension >=16")
        with torch.cuda.device(q.device):
            flash_fwd_kernel[(triton.cdiv(n, 32), q.numel() // (n * d))](q, k, v, output, lse, n, k.shape[-2], d, 32, 64, is_causal)
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, do):
        grads = flash_backward(*ctx.saved_tensors[:3], ctx.saved_tensors[3], do, ctx.saved_tensors[4], ctx.is_causal)
        return (*grads, None) if len(ctx.needs_input_grad) == 4 else grads


@torch.compile(dynamic=True)
def flash_backward(q, k, v, output, do, lse, is_causal):
    """Recompute probabilities from L; D is the row-wise dot of dO and Out.

    This exercise's compiled dense backward has quadratic transient memory.
    Only forward and saved activations have FlashAttention's linear memory bound.
    """
    qf, kf, vf, dof = q.float(), k.float(), v.float(), do.float()
    scores = qf @ kf.transpose(-1, -2) / q.shape[-1] ** 0.5
    if is_causal:
        mask = torch.arange(q.shape[-2], device=q.device)[:, None] < torch.arange(k.shape[-2], device=q.device)[None, :]
        scores = scores.masked_fill(mask, -1e6)
    p = torch.exp(scores - lse[..., None])
    delta = (output.float() * dof).sum(-1, keepdim=True)
    dp = dof @ vf.transpose(-1, -2)
    ds = p * (dp - delta)
    dq = ds @ kf / q.shape[-1] ** 0.5
    dk = ds.transpose(-1, -2) @ qf / q.shape[-1] ** 0.5
    dv = p.transpose(-1, -2) @ dof
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


@triton.jit
def _flash_bwd(Q, K, V, Out, DO, L, DQ, DK, DV, N: tl.constexpr, D: tl.constexpr, B: tl.constexpr, QUERY: tl.constexpr, CAUSAL: tl.constexpr):
    tile, batch = tl.program_id(0), tl.program_id(1)
    rows = tile * B + tl.arange(0, B)
    dims = tl.arange(0, D)
    base = batch * N * D
    offsets = base + rows[:, None] * D + dims[None, :]
    if QUERY:
        q = tl.load(Q + offsets, rows[:, None] < N, 0)
        do = tl.load(DO + offsets, rows[:, None] < N, 0)
        o = tl.load(Out + offsets, rows[:, None] < N, 0)
        ell = tl.load(L + batch * N + rows, rows < N, 0)
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), 1)
        dq = tl.zeros((B, D), tl.float32)
        end = N
        if CAUSAL:
            end = tl.minimum(N, (tile + 1) * B)
        for j in range(tl.cdiv(end, B)):
            cols = j * B + tl.arange(0, B)
            ptr = base + cols[:, None] * D + dims[None, :]
            k = tl.load(K + ptr, cols[:, None] < N, 0)
            v = tl.load(V + ptr, cols[:, None] < N, 0)
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * (D**-0.5)
            valid = cols[None, :] < N
            if CAUSAL:
                valid = valid & (rows[:, None] >= cols[None, :])
            p = tl.exp(tl.where(valid, s - ell[:, None], -float("inf")))
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
            ds = p * (dp - delta[:, None]) * (D**-0.5)
            dq = tl.dot(ds.to(k.dtype), k, acc=dq, input_precision="ieee")
        tl.store(DQ + offsets, dq, rows[:, None] < N)
    else:
        k = tl.load(K + offsets, rows[:, None] < N, 0)
        v = tl.load(V + offsets, rows[:, None] < N, 0)
        dk = tl.zeros((B, D), tl.float32)
        dv = tl.zeros((B, D), tl.float32)
        start = 0
        if CAUSAL:
            start = tile
        for i in range(start, tl.cdiv(N, B)):
            qs = i * B + tl.arange(0, B)
            ptr = base + qs[:, None] * D + dims[None, :]
            q = tl.load(Q + ptr, qs[:, None] < N, 0)
            do = tl.load(DO + ptr, qs[:, None] < N, 0)
            o = tl.load(Out + ptr, qs[:, None] < N, 0)
            ell = tl.load(L + batch * N + qs, qs < N, 0)
            delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), 1)
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * (D**-0.5)
            valid = (qs[:, None] < N) & (rows[None, :] < N)
            if CAUSAL:
                valid = valid & (qs[:, None] >= rows[None, :])
            p = tl.exp(tl.where(valid, s - ell[:, None], -float("inf")))
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
            ds = p * (dp - delta[:, None]) * (D**-0.5)
            dk = tl.dot(tl.trans(ds.to(q.dtype)), q, acc=dk, input_precision="ieee")
            dv = tl.dot(tl.trans(p.to(do.dtype)), do, acc=dv, input_precision="ieee")
        tl.store(DK + offsets, dk, rows[:, None] < N)
        tl.store(DV + offsets, dv, rows[:, None] < N)


class FlashAttentionTritonFull(FlashAttentionTriton):
    """Optional fully tiled GPU backward for long-context training."""

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, ell = ctx.saved_tensors
        if q.shape[-2] != k.shape[-2]:
            grads = flash_backward(q, k, v, o, do, ell, ctx.is_causal)
        else:
            dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
            n, d = q.shape[-2:]
            do = do.contiguous()
            grid = (triton.cdiv(n, 32), q.numel() // (n * d))
            with torch.cuda.device(q.device):
                for query in (True, False):
                    _flash_bwd[grid](q, k, v, o, do, ell, dq, dk, dv, n, d, 32, query, ctx.is_causal)
            grads = dq, dk, dv
        return (*grads, None) if len(ctx.needs_input_grad) == 4 else grads
