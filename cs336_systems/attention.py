"""Explicit scaled dot-product attention and memory-efficient tiled attention."""
import torch


def attention(q, k, v, is_causal=False):
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    if is_causal:
        mask = torch.arange(q.shape[-2], device=q.device)[:, None] < torch.arange(k.shape[-2], device=q.device)[None, :]
        scores = scores.masked_fill(mask, float('-inf'))
    return scores.softmax(-1) @ v


class FlashAttentionPyTorch(torch.autograd.Function):
    """Online softmax reference, with FP32 accumulators and 64 by 64 tiles."""
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        output = torch.empty_like(q)
        lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
        for start in range(0, q.shape[-2], 64):
            qi = q[..., start:start+64, :].float()
            m = torch.full(qi.shape[:-1], -float('inf'), device=q.device)
            l = torch.zeros_like(m)
            acc = torch.zeros_like(qi)
            for j in range(0, k.shape[-2], 64):
                scores = qi @ k[..., j:j+64, :].float().transpose(-1, -2) / q.shape[-1]**0.5
                if is_causal:
                    mask = torch.arange(start, start+qi.shape[-2], device=q.device)[:, None] < torch.arange(j, min(j+64, k.shape[-2]), device=q.device)[None, :]
                    scores = scores.masked_fill(mask, -1e6)
                new_m = torch.maximum(m, scores.amax(-1))
                alpha = torch.exp(m-new_m)
                p = torch.exp(scores-new_m[..., None])
                l = alpha*l+p.sum(-1)
                acc = alpha[..., None]*acc+p @ v[..., j:j+64, :].float()
                m = new_m
            output[..., start:start+64, :] = (acc/l[..., None]).to(q.dtype)
            lse[..., start:start+64] = m+l.log()
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, do):
        grads = flash_backward(*ctx.saved_tensors[:3], ctx.saved_tensors[3], do, ctx.saved_tensors[4], ctx.is_causal)
        return (*grads, None) if len(ctx.needs_input_grad)==4 else grads


import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(Q, K, V, O, L, NQ: tl.constexpr, NK: tl.constexpr, D: tl.constexpr,
                     BQ: tl.constexpr, BK: tl.constexpr, is_causal: tl.constexpr):
    tile, batch = tl.program_id(0), tl.program_id(1)
    qp = tl.make_block_ptr(Q+batch*NQ*D, (NQ,D), (D,1), (tile*BQ,0), (BQ,D), (1,0))
    kp = tl.make_block_ptr(K+batch*NK*D, (NK,D), (D,1), (0,0), (BK,D), (1,0))
    vp = tl.make_block_ptr(V+batch*NK*D, (NK,D), (D,1), (0,0), (BK,D), (1,0))
    op = tl.make_block_ptr(O+batch*NQ*D, (NQ,D), (D,1), (tile*BQ,0), (BQ,D), (1,0))
    q = tl.load(qp, boundary_check=(0,1), padding_option='zero')
    m = tl.full((BQ,), -float('inf'), tl.float32)
    l = tl.zeros((BQ,), tl.float32)
    acc = tl.zeros((BQ,D), tl.float32)
    rows = tile*BQ+tl.arange(0,BQ)
    end = NK
    if is_causal:
        end = tl.minimum(NK, (tile+1)*BQ)
    for j in range(tl.cdiv(end, BK)):
        k = tl.load(kp, boundary_check=(0,1), padding_option='zero')
        v = tl.load(vp, boundary_check=(0,1), padding_option='zero')
        s = tl.dot(q, tl.trans(k), input_precision='ieee') * (D**-0.5)
        cols = j*BK+tl.arange(0,BK)
        s = tl.where(cols[None,:] < NK, s, -float('inf'))
        if is_causal:
            s = tl.where(rows[:,None] >= cols[None,:], s, -1e6)
        new_m = tl.maximum(m, tl.max(s,1))
        alpha = tl.exp(m-new_m)
        p = tl.exp(s-new_m[:,None])
        l = l*alpha+tl.sum(p,1)
        acc = acc*alpha[:,None]
        acc = tl.dot(p.to(v.dtype), v, acc=acc, input_precision='ieee')
        m = new_m
        kp = tl.advance(kp,(BK,0))
        vp = tl.advance(vp,(BK,0))
    tl.store(op, (acc/l[:,None]).to(O.dtype.element_ty), boundary_check=(0,1))
    tl.store(L+batch*NQ+rows, m+tl.log(l), rows<NQ)


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        q,k,v = q.contiguous(),k.contiguous(),v.contiguous()
        output = torch.empty_like(q)
        lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
        n, d = q.shape[-2:]
        if d < 16 or d & (d-1):
            raise ValueError('Triton attention requires a power-of-two head dimension >=16')
        flash_fwd_kernel[(triton.cdiv(n,32), q.numel()//(n*d))](q,k,v,output,lse,n,k.shape[-2],d,32,64,is_causal)
        ctx.save_for_backward(q,k,v,output,lse)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, do):
        grads = flash_backward(*ctx.saved_tensors[:3], ctx.saved_tensors[3], do, ctx.saved_tensors[4], ctx.is_causal)
        return (*grads, None) if len(ctx.needs_input_grad)==4 else grads


@torch.compile(dynamic=True)
def flash_backward(q, k, v, output, do, lse, is_causal):
    """Recompute probabilities from L; D is the row-wise dot of dO and O.

    This exercise's compiled dense backward has quadratic transient memory.
    Only forward and saved activations have FlashAttention's linear memory bound.
    """
    qf, kf, vf, dof = q.float(), k.float(), v.float(), do.float()
    scores = qf @ kf.transpose(-1,-2) / q.shape[-1]**0.5
    if is_causal:
        mask = torch.arange(q.shape[-2], device=q.device)[:,None] < torch.arange(k.shape[-2], device=q.device)[None,:]
        scores = scores.masked_fill(mask, -1e6)
    p = torch.exp(scores-lse[...,None])
    delta = (output.float()*dof).sum(-1,keepdim=True)
    dp = dof @ vf.transpose(-1,-2)
    ds = p*(dp-delta)
    dq = ds @ kf / q.shape[-1]**0.5
    dk = ds.transpose(-1,-2) @ qf / q.shape[-1]**0.5
    dv = p.transpose(-1,-2) @ dof
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)
