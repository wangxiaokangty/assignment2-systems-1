import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path
    import marimo as mo
    import torch
    from cs336_systems.leaderboard import ChunkedCrossEntropy, Config, TrainingModel, FusedAdamW
    return ChunkedCrossEntropy, Config, FusedAdamW, Path, TrainingModel, json, mo, torch


@app.cell
def _(mo):
    mo.md(r"""
    # Leaderboard — local A100 full training step

    The handout specifies **two B200 GPUs**. This machine has A100 80GB GPUs, so
    these results are local measurements and are **not an official B200 score**.
    The full configuration is batch 2, sequence 32768, vocabulary 151936,
    width 4096, FFN 11008, 34 blocks, 32 heads, causal BF16.

    `cs336_systems.leaderboard` reuses the reference model's projections, RoPE,
    normalization and SwiGLU. It adds our tiled Triton attention forward/backward,
    layer placement across two GPUs, activation checkpointing, chunked linear
    cross entropy (mean reduction, as in `cs336_basics.nn_utils.cross_entropy`),
    and an original fused Triton AdamW update with FP32 moments. Unlike the reference
    optimizer applied directly to BF16 parameters, optimizer moments stay FP32 and
    the parameter update rounds once to BF16; this preserves the AdamW formula but
    is not bitwise identical to BF16-state AdamW. `forward` still
    returns logits; `loss` avoids materializing the entire logits tensor during
    training. The layer split is sequential model parallelism and does not
    overlap the two devices' computation.

    Run from the repository root (choose two idle physical GPUs):
    ```bash
    CUDA_VISIBLE_DEVICES=0,1 uv run python -m cs336_systems.leaderboard --steps 3
    ```
    To also reproduce the cold-cache condition, use fresh cache directories:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1 TRITON_CACHE_DIR=$(mktemp -d) TORCHINDUCTOR_CACHE_DIR=$(mktemp -d) /usr/bin/time -p uv run python -m cs336_systems.leaderboard --steps 3
    ```
    A smaller sanity run uses `--smoke --output results/leaderboard_smoke.json`.
    Configuration files live in `models/`. Timing includes zeroing gradients,
    forward, loss, backward and optimizer update, with both devices synchronized.
    The first full warmup includes compilation and optimizer-state allocation;
    `total_seconds` includes model initialization and all warmup/measured steps.
    The final run uses fresh isolated Triton and TorchInductor cache directories;
    `process_wall_seconds` includes Python startup, compilation, warmup, and all three
    measured steps. The original tile baseline is retained for comparison.
    No results are submitted to any external service.
    """)
    return


@app.cell
def _(Path, json, mo):
    _path = Path("results/leaderboard.json")
    _data = [{"variant": _name, **json.loads(_file.read_text())} for _name, _file in [("original tiles", Path("results/leaderboard_baseline.json")), ("tuned cold-cache run", _path)] if _file.exists()]
    mo.ui.table(_data or [{"status": "Run the full local benchmark to produce results."}])
    return


@app.cell
def _(mo):
    mo.md("""
    **Full-size local result:** best **73.145 s/step**, median **73.177 s/step**,
    versus **78.006 s/step** with the original tiles (about 6.2% faster).
    Both A100s fit the complete configuration: peaks are **68.70 GB / 69.95 GB**.
    The cold-cache process, including initialization, compilation, one warmup and
    three measured steps, took **302.08 s**, below the ten-minute startup budget.
    This is an A100 result; the B200 leaderboard's ten-second performance target
    has not been demonstrated on its required hardware.
    """)
    return


@app.cell
def _(ChunkedCrossEntropy, torch, mo):
    # Small one-off numerical verification; the reusable loss stays in the module.
    torch.manual_seed(123)
    _hidden = torch.randn(2, 7, 16, requires_grad=True)
    _weight = torch.randn(37, 16, requires_grad=True)
    _labels = torch.randint(37, (2, 7))
    _loss = ChunkedCrossEntropy.apply(_hidden, _weight, _labels, 5)
    _reference = torch.nn.functional.cross_entropy((_hidden @ _weight.T).flatten(0, 1), _labels.flatten())
    _got = torch.autograd.grad(_loss, (_hidden, _weight))
    _expected = torch.autograd.grad(_reference, (_hidden, _weight))
    torch.testing.assert_close(_loss, _reference)
    for _actual, _wanted in zip(_got, _expected):
        torch.testing.assert_close(_actual, _wanted)
    mo.md(f"Chunked loss and both gradients agree with dense cross entropy (FP32 CPU, loss **{_loss.item():.6f}**).")
    return


@app.cell
def _(mo):
    verify_gpu = mo.ui.run_button(label="Verify small BF16 model and fused AdamW on two GPUs")
    verify_gpu
    return (verify_gpu,)


@app.cell
def _(Config, TrainingModel, FusedAdamW, torch, mo, verify_gpu):
    mo.stop(not verify_gpu.value, mo.md("GPU correctness verification is opt-in to avoid interfering with other benchmarks."))
    from cs336_basics.model import BasicsTransformerLM
    from cs336_basics.optimizer import AdamW
    from dataclasses import asdict
    _config = Config(vocab_size=256, context_length=1024, d_model=256, d_ff=704, num_layers=2, num_heads=2)
    _optimized = TrainingModel(_config)
    _base = BasicsTransformerLM(**asdict(_config)).cuda().bfloat16()
    _base.load_state_dict(_optimized.state_dict())
    _tokens = torch.randint(256, (2, _config.context_length), device='cuda:0')
    _targets = torch.randint(256, (2, _config.context_length), device='cuda:0')
    _out = _optimized(_tokens)
    _ref_out = _base(_tokens)
    torch.testing.assert_close(_out.cpu(), _ref_out.cpu(), atol=0.08, rtol=0.08)
    _loss = _optimized.loss(_tokens, _targets)
    _ref_loss = torch.nn.functional.cross_entropy(_ref_out.float().flatten(0, 1), _targets.flatten())
    _loss.backward()
    _ref_loss.backward()
    torch.testing.assert_close(_loss.cpu(), _ref_loss.cpu(), atol=0.03, rtol=0.02)
    _errors = []
    for _name, _param in _optimized.named_parameters():
        _ref_grad = dict(_base.named_parameters())[_name].grad.cpu().float()
        _grad = _param.grad.cpu().float()
        _relative = (_grad - _ref_grad).norm() / _ref_grad.norm().clamp_min(1e-8)
        assert _relative < 0.15, (_name, _relative)
        _errors.append(float(_relative))
    _p = torch.nn.Parameter(torch.randn(1024, device='cuda'))
    _q = torch.nn.Parameter(_p.detach().clone())
    _ours, _reference = FusedAdamW([_p]), AdamW([_q])
    for _ in range(3):
        _p.grad = torch.randn_like(_p)
        _q.grad = _p.grad.clone()
        _ours.step()
        _reference.step()
    torch.testing.assert_close(_p, _q, atol=1e-6, rtol=1e-5)
    mo.md(f"BF16 logits/loss and all parameter gradients verified; max relative gradient error {_errors and max(_errors):.4f}. Fused AdamW agrees with reference over three FP32 updates.")
    return


if __name__ == "__main__":
    app.run()
