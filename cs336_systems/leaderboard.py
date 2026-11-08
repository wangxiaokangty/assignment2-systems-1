"""Local full-step benchmark: layer placement, recomputation, tiled attention and loss.

The benchmark is intentionally explicit about hardware and configuration. It does
not submit to the external leaderboard. Checkpoints/configurations live in models/.
"""
import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import triton
import triton.language as tl
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import BasicsTransformerLM, CausalMultiHeadSelfAttention


@dataclass
class Config:
    vocab_size: int = 151936
    context_length: int = 32768
    d_model: int = 4096
    d_ff: int = 11008
    num_layers: int = 34
    num_heads: int = 32


class FlashSelfAttention(CausalMultiHeadSelfAttention):
    def forward(self, x, token_positions=None):
        from cs336_systems.attention import FlashAttentionTritonFull

        shape = (*x.shape[:-1], self.num_heads, self.d_k)
        q, k, v = (proj(x).view(shape).transpose(-3, -2) for proj in (self.q_proj, self.k_proj, self.v_proj))
        if self.positional_encoder is not None:
            positions = None if token_positions is None else token_positions.unsqueeze(-2)
            q = self.positional_encoder(q, positions)
            k = self.positional_encoder(k, positions)
        out = FlashAttentionTritonFull.apply(q, k, v, True)
        return self.output_proj(out.transpose(-3, -2).reshape_as(x))


class ChunkedCrossEntropy(torch.autograd.Function):
    """Mean linear cross entropy; recompute one token chunk at a time in backward."""
    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk=512):
        x, y = hidden.reshape(-1, hidden.shape[-1]), targets.reshape(-1)
        loss = torch.zeros((), device=x.device, dtype=torch.float32)
        for i in range(0, len(x), chunk):
            scores = (x[i:i+chunk] @ weight.T).float()
            loss += (scores.logsumexp(-1) - scores.gather(1, y[i:i+chunk, None]).squeeze(1)).sum()
        ctx.save_for_backward(hidden, weight, targets)
        ctx.chunk = chunk
        return loss / len(x)

    @staticmethod
    def backward(ctx, grad):
        hidden, weight, targets = ctx.saved_tensors
        x, y = hidden.reshape(-1, hidden.shape[-1]), targets.reshape(-1)
        dx = torch.empty_like(x)
        dw = torch.zeros_like(weight, dtype=torch.float32)
        for i in range(0, len(x), ctx.chunk):
            xi = x[i:i+ctx.chunk]
            p = (xi @ weight.T).float().softmax(-1)
            p[torch.arange(len(xi), device=x.device), y[i:i+ctx.chunk]] -= 1
            p = (p * (grad / len(x))).to(x.dtype)
            dx[i:i+ctx.chunk] = p @ weight
            dw.add_((p.T @ xi).float())
        return (dx.view_as(hidden), dw.to(weight.dtype), None, None)[:len(ctx.needs_input_grad)]


@triton.jit
def _adam(P, G, M, V, N: tl.constexpr, LR: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr,
          EPS: tl.constexpr, WD: tl.constexpr, ALPHA, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    p = tl.load(P+i, i<N, 0).to(tl.float32)
    g = tl.load(G+i, i<N, 0).to(tl.float32)
    m = B1 * tl.load(M+i, i<N, 0) + (1-B1)*g
    v = B2 * tl.load(V+i, i<N, 0) + (1-B2)*g*g
    # Same epsilon placement and decoupled weight decay as cs336_basics.AdamW.
    p = p*(1-LR*WD)-ALPHA*m/(tl.sqrt(v)+EPS)
    tl.store(P+i, p, i<N)
    tl.store(M+i, m, i<N)
    tl.store(V+i, v, i<N)


class FusedAdamW(torch.optim.Optimizer):
    """One original Triton update per tensor, FP32 moments, BF16 model weights."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        with torch.enable_grad():
            loss = closure() if closure else None
        for group in self.param_groups:
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state.update(m=torch.zeros_like(p, dtype=torch.float32), v=torch.zeros_like(p, dtype=torch.float32), step=0)
                state['step'] += 1
                t = state['step']
                alpha = group['lr'] * (1-b2**t)**0.5 / (1-b1**t)
                with torch.cuda.device(p.device):
                    _adam[(triton.cdiv(p.numel(), 1024),)](p, p.grad, state['m'], state['v'], p.numel(),
                        group['lr'], b1, b2, group['eps'], group['weight_decay'], alpha, 1024)
        return loss


class TrainingModel(BasicsTransformerLM):
    """Original parameter names and math, split layers across local CUDA devices."""
    def __init__(self, config, devices=(0, 1), checkpoint_blocks=True):
        with torch.device('meta'):
            super().__init__(**asdict(config))
        self.checkpoint_blocks = checkpoint_blocks
        self.devices = tuple(torch.device(f'cuda:{i}') for i in devices)
        # Rebuild each part on its final GPU to avoid a full CPU/GPU staging copy.
        from cs336_basics.model import Embedding, Linear, RMSNorm, RotaryEmbedding, TransformerBlock
        with torch.device(self.devices[0]):
            self.token_embeddings = Embedding(config.vocab_size, config.d_model).to(torch.bfloat16)
        blocks = []
        for i in range(config.num_layers):
            device = self.devices[min(i * len(devices) // config.num_layers, len(devices)-1)]
            with torch.device(device):
                rope = RotaryEmbedding(config.context_length, config.d_model // config.num_heads, 10000.0)
                block = TransformerBlock(config.d_model, config.num_heads, config.d_ff, rope).to(torch.bfloat16)
                block.attn.__class__ = FlashSelfAttention
                blocks.append(block)
        self.layers = torch.nn.ModuleList(blocks)
        self.positional_encoder = None
        with torch.device(self.devices[-1]):
            self.ln_final = RMSNorm(config.d_model).to(torch.bfloat16)
            self.lm_head = Linear(config.d_model, config.vocab_size).to(torch.bfloat16)

    def hidden(self, tokens):
        x = self.token_embeddings(tokens.to(self.devices[0]))
        for layer in self.layers:
            x = x.to(next(layer.parameters()).device)
            x = checkpoint(layer, x, use_reentrant=False) if self.checkpoint_blocks else layer(x)
        return self.ln_final(x.to(self.devices[-1]))

    def forward(self, tokens):
        return self.lm_head(self.hidden(tokens))

    def loss(self, tokens, targets, chunk=512):
        return ChunkedCrossEntropy.apply(self.hidden(tokens), self.lm_head.weight, targets.to(self.devices[-1]), chunk)


def benchmark(config, devices=(0, 1), batch=2, warmup=1, steps=3):
    torch.manual_seed(42)
    start = time.perf_counter()
    model = TrainingModel(config, devices)
    optimizer = FusedAdamW(model.parameters())
    tokens = torch.randint(config.vocab_size, (batch, config.context_length), device=model.devices[0])
    targets = torch.randint(config.vocab_size, tokens.shape, device=model.devices[-1])
    times, losses = [], []
    for i in range(warmup + steps):
        for device in model.devices:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(tokens, targets)
        loss.backward()
        optimizer.step()
        for device in model.devices:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - tick
        print(f'step={i} seconds={elapsed:.6f} loss={loss.item():.6f}', flush=True)
        if i >= warmup:
            times.append(elapsed)
            losses.append(loss.item())
    return dict(config=asdict(config), batch=batch, devices=[torch.cuda.get_device_name(d) for d in model.devices],
                seconds=times, best_seconds=min(times), median_seconds=statistics.median(times), losses=losses,
                peak_allocated_gb=[torch.cuda.max_memory_allocated(d)/1e9 for d in model.devices],
                total_seconds=time.perf_counter()-start, warmup=warmup, torch_version=torch.__version__,
                official_B200_score=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--output', default='results/leaderboard.json')
    args = parser.parse_args()
    config = Config(vocab_size=1024, context_length=128, d_model=256, d_ff=704, num_layers=2, num_heads=4) if args.smoke else Config()
    Path('models').mkdir(exist_ok=True)
    Path('models/leaderboard_config.json').write_text(json.dumps(asdict(config), indent=2))
    result = benchmark(config, steps=args.steps, warmup=args.warmup)
    Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
