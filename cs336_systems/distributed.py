"""Small, explicit data-parallel wrappers (one process per device)."""
import torch
from torch import distributed as dist, nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class NaiveDDP(nn.Module):
    """Replicate rank zero's model and average gradients after backward."""
    def __init__(self, module: nn.Module, *, flatten: bool = False):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.flatten = flatten
        for tensor in list(module.parameters()) + list(module.buffers()):
            dist.broadcast(tensor.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @torch.no_grad()
    def finish_gradient_synchronization(self):
        gradients = [p.grad for p in self.module.parameters() if p.grad is not None]
        if self.flatten:
            # Separate dtypes/devices so flattening never silently changes precision.
            groups = {}
            for grad in gradients:
                groups.setdefault((grad.device, grad.dtype), []).append(grad)
            for group in groups.values():
                flat = _flatten_dense_tensors(group)
                dist.all_reduce(flat)
                flat.div_(self.world_size)
                for grad, reduced in zip(group, _unflatten_dense_tensors(flat, group)):
                    grad.copy_(reduced)
        else:
            for grad in gradients:
                dist.all_reduce(grad)
                grad.div_(self.world_size)


class DDP(NaiveDDP):
    """Overlap per-parameter reductions with autograd's remaining work.

    Call finish_gradient_synchronization after each backward and before step.
    As with the exercise, every rank must use the same parameters each step.
    """
    def __init__(self, module: nn.Module):
        super().__init__(module)
        self.pending = []
        self.hooks = [p.register_post_accumulate_grad_hook(self._reduce)
                      for p in module.parameters() if p.requires_grad]

    def _reduce(self, parameter):
        parameter.grad.div_(self.world_size)
        self.pending.append(dist.all_reduce(parameter.grad, async_op=True))

    def finish_gradient_synchronization(self):
        for handle in self.pending:
            handle.wait()
        self.pending.clear()
