"""Idealized ring communication costs (FP16, ignoring latency and elementwise FLOPs)."""


def ring_seconds(size_bytes: float, devices: int, bandwidth: float, *, alternate=False):
    if devices < 1 or bandwidth <= 0 or size_bytes < 0:
        raise ValueError("Require devices >= 1, bandwidth > 0 and size >= 0")
    return (devices - 1) * size_bytes / bandwidth * (1 if alternate else 2 / devices)


def costs(batch=4096, width=4096, hidden=11008, fsdp=1, tp=1, compute=312e12, bandwidth=300e9, *, strategy="fsdp", overlap=True):
    """Per-device forward/backward seconds, with axis overlap for 2D FSDP/TP."""
    if min(batch, width, hidden, fsdp, tp, compute, bandwidth) <= 0:
        raise ValueError("All dimensions, device counts and rates must be positive")
    if strategy not in {"dp", "fsdp"}:
        raise ValueError("strategy must be dp or fsdp")
    forward = 6 * batch * width * hidden / (fsdp * tp * compute)
    weight_gather = (fsdp - 1) / fsdp * 6 * width * hidden / (tp * bandwidth)
    activation_reduce = (tp - 1) / tp * 4 * batch * width / (fsdp * bandwidth)
    combine = max if overlap else lambda a, b: a + b
    return {
        "forward_compute_s": forward,
        "backward_compute_s": 2 * forward,
        "forward_communication_s": combine(0 if strategy == "dp" else weight_gather, activation_reduce),
        "backward_communication_s": combine(2 * weight_gather, activation_reduce),
    }


def optimal_grid(batch=4096, hidden=11008, compute=312e12, bandwidth=300e9, *, overlap=True):
    """Continuous upper bound; physical grids must also satisfy dimension divisibility."""
    a = compute / (batch * bandwidth)
    b = 2 * compute / (3 * hidden * bandwidth)
    if overlap:
        p, q = 1 + 1 / a, 1 + 1 / b
    else:
        k = 1 + a + b
        p = min(max(k / (2 * a), 1), (k - b) / a)
        q = (k - a * p) / b
    return {"fsdp": p, "tp": q, "total": p * q}
