"""Reproducible synchronized benchmarks of the assignment's reference Transformer."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
from timeit import default_timer

import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

MODEL_SIZES = {
    'small': (768, 3072, 12, 12), 'medium': (1024, 4096, 24, 16),
    'large': (1280, 5120, 36, 20), 'xl': (2560, 10240, 32, 32),
    '10B': (4608, 12288, 50, 36),
}

@dataclass
class ModelConfig:
    size: str = 'small'
    context_length: int = 512
    batch_size: int = 4
    vocab_size: int = 10000
    d_model: int | None = None
    d_ff: int | None = None
    num_layers: int | None = None
    num_heads: int | None = None

    def model_kwargs(self):
        values = dict(zip(('d_model', 'd_ff', 'num_layers', 'num_heads'), MODEL_SIZES[self.size]))
        values.update({key: getattr(self, key) for key in values if getattr(self, key) is not None})
        return dict(vocab_size=self.vocab_size, context_length=self.context_length, **values)


def build_model(config: ModelConfig, device='cuda'):
    # Initialize directly on the GPU, avoiding a second CPU copy of large models.
    with torch.device(device):
        return BasicsTransformerLM(**config.model_kwargs())


def benchmark(config: ModelConfig, mode='train', precision='fp32', warmup=5, steps=10,
              device='cuda', snapshot=None, checkpoint_size=0, compile_model=False):
    if mode not in ('forward', 'backward', 'train') or warmup < 0 or steps < 1:
        raise ValueError('invalid benchmark mode or iteration count')
    torch.manual_seed(42)
    model = build_model(config, device)
    if checkpoint_size:
        from cs336_systems.checkpointing import checkpoint_model
        checkpoint_model(model, checkpoint_size)
    if compile_model:
        model = torch.compile(model)
    optimizer = AdamW(model.parameters()) if mode == 'train' else None
    tokens = torch.randint(config.vocab_size, (config.batch_size, config.context_length), device=device)
    targets = torch.randint(config.vocab_size, tokens.shape, device=device)
    cuda = torch.device(device).type == 'cuda'
    def sync():
        if cuda:
            torch.cuda.synchronize(device)
    def region(name):
        return torch.cuda.nvtx.range(name) if cuda else nullcontext()
    def step():
        model.zero_grad(set_to_none=True)
        sync()
        start = default_timer()
        autocast = torch.autocast(torch.device(device).type, dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()
        with region('forward'), torch.set_grad_enabled(mode != 'forward'), autocast:
            logits = model(tokens)
        sync()
        forward_end = default_timer()
        if mode != 'forward':
            with region('loss'):
                loss = cross_entropy(logits.float(), targets)
            sync()
            loss_end = default_timer()
            with region('backward'):
                loss.backward()
            sync()
            backward_end = default_timer()
        else:
            loss_end = backward_end = forward_end
        if optimizer:
            with region('optimizer'):
                optimizer.step()
            sync()
        end = default_timer()
        return {'forward_ms': (forward_end-start)*1000, 'loss_ms': (loss_end-forward_end)*1000,
                'backward_ms': (backward_end-loss_end)*1000, 'optimizer_ms': (end-backward_end)*1000,
                'total_ms': (end-start)*1000}
    for _ in range(warmup):
        step()
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    if snapshot:
        torch.cuda.memory._record_memory_history(max_entries=200000)
    try:
        with region('measurement'):
            samples = [step() for _ in range(steps)]
    finally:
        if snapshot:
            Path(snapshot).parent.mkdir(parents=True, exist_ok=True)
            torch.cuda.memory._dump_snapshot(str(snapshot))
            torch.cuda.memory._record_memory_history(enabled=None)
    return {'config': asdict(config), 'mode': mode, 'precision': precision, 'warmup': warmup, 'steps': steps,
            'checkpoint_size': checkpoint_size, 'compile_model': compile_model,
            'device': torch.cuda.get_device_name(device) if cuda else str(device), 'torch': torch.__version__,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if cuda else None,
            'samples': samples,
            'mean': {k: statistics.mean(s[k] for s in samples) for k in samples[0]},
            'std': {k: statistics.pstdev(s[k] for s in samples) for k in samples[0]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', choices=MODEL_SIZES, default='small')
    for key, default in [('context-length',512), ('batch-size',4), ('vocab-size',10000),
                         ('d-model',None), ('d-ff',None), ('num-layers',None), ('num-heads',None)]:
        parser.add_argument('--'+key, type=int, default=default)
    parser.add_argument('--mode', choices=['forward','backward','train'], default='train')
    parser.add_argument('--precision', choices=['fp32','bf16'], default='fp32')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--snapshot')
    parser.add_argument('--checkpoint-size', type=int, default=0)
    parser.add_argument('--compile-model', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--annotate-attention', action='store_true')
    args = vars(parser.parse_args())
    if args.pop('annotate_attention'):
        from cs336_systems.profiling import annotate_attention
        annotate_attention()
    output = args.pop('output')
    config = ModelConfig(**{k: args.pop(k) for k in ModelConfig.__dataclass_fields__})
    try:
        result = benchmark(config, **args)
    except torch.OutOfMemoryError as error:
        result = {'config': asdict(config), **args, 'status': 'OOM', 'error': str(error)}
    text = json.dumps(result, indent=2)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text+'\n')
    print(text)

if __name__ == '__main__':
    main()
