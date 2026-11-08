"""Layer-wise FSDP with FP32 master shards and optional low precision gathers.

Linear/Embedding weights are sharded; small parameters (norms and biases) are
replicated. Full weights are recomputed for backward rather than saved by autograd.
"""
import math
import torch
from torch import distributed as dist, nn
from torch.nn import functional as F
from cs336_basics.model import Linear, Embedding


@torch.no_grad()
def _gather(shard, shape, dtype, async_op=False):
    local = shard.to(dtype).contiguous()
    full = torch.empty(local.numel() * dist.get_world_size(), device=local.device, dtype=local.dtype)
    work = dist.all_gather_into_tensor(full, local, async_op=async_op)
    return full, work


class _WeightOperation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shard, layer):
        full = layer.take_full_weight()
        ctx.layer = layer
        ctx.save_for_backward(x, shard)
        if layer.embedding:
            out = F.embedding(x, full, padding_idx=layer.padding_idx)
        else:
            out = F.linear(x, full)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, shard = ctx.saved_tensors
        layer = ctx.layer
        grad_input = None
        if layer.embedding:
            grad_weight = torch.zeros(layer.shape, device=shard.device, dtype=grad_output.dtype)
            grad_weight.index_add_(0, x.reshape(-1), grad_output.reshape(-1, layer.shape[1]))
            if layer.padding_idx is not None:
                grad_weight[layer.padding_idx].zero_()
        else:
            with torch.profiler.record_function("fsdp_backward_all_gather"):
                full, _ = _gather(shard, layer.shape, layer.compute_dtype)
                full = full[:math.prod(layer.shape)].view(layer.shape)
            grad_input = grad_output @ full
            grad_weight = grad_output.reshape(-1, layer.shape[0]).T @ x.reshape(-1, layer.shape[1])
        # Reduce in FP32: the compute gradient has already been rounded to the
        # compute dtype, matching low-precision computation with FP32 masters.
        flat = grad_weight.float().flatten()
        padding = shard.numel() * dist.get_world_size() - flat.numel()
        if padding:
            flat = F.pad(flat, (0, padding))
        reduced = torch.empty_like(shard)
        with torch.profiler.record_function("fsdp_gradient_reduce_scatter"):
            if dist.get_backend() == "gloo":
                # Gloo lacks reduce_scatter on some builds; equivalent fallback.
                dist.all_reduce(flat)
                reduced.copy_(flat.chunk(dist.get_world_size())[dist.get_rank()])
            else:
                dist.reduce_scatter_tensor(reduced, flat)
            reduced.div_(dist.get_world_size())
        return grad_input, reduced, None


class _ShardedLayer(nn.Module):
    def __init__(self, module, compute_dtype):
        nn.Module.__init__(self)
        weight = module.weight
        self.shape = tuple(weight.shape)
        self.compute_dtype = compute_dtype or weight.dtype
        self.embedding = isinstance(module, (Embedding, nn.Embedding))
        self.padding_idx = getattr(module, "padding_idx", None)
        flat = weight.detach().float().flatten()
        chunk = math.ceil(flat.numel() / dist.get_world_size())
        flat = F.pad(flat, (0, chunk * dist.get_world_size() - flat.numel()))
        self.weight = nn.Parameter(flat.chunk(dist.get_world_size())[dist.get_rank()].clone(), requires_grad=weight.requires_grad)
        self.bias = getattr(module, "bias", None)
        self._prefetched = None

    def prefetch(self):
        if self._prefetched is None:
            with torch.profiler.record_function("fsdp_forward_all_gather"):
                self._prefetched = _gather(self.weight, self.shape, self.compute_dtype, async_op=True)

    def take_full_weight(self):
        self.prefetch()
        full, work = self._prefetched
        work.wait()
        self._prefetched = None
        return full[:math.prod(self.shape)].view(self.shape)

    def forward(self, x):
        out = _WeightOperation.apply(x, self.weight, self)
        if self.bias is not None:
            out = out + self.bias
        return out


class _ShardedLinear(_ShardedLayer, Linear):
    pass


class _ShardedEmbedding(_ShardedLayer, Embedding):
    pass


class FSDP(nn.Module):
    def __init__(self, module, compute_dtype=None):
        super().__init__()
        self.module = module
        for tensor in list(module.parameters()) + list(module.buffers()):
            dist.broadcast(tensor.data, src=0)
        self.layers = []
        self._replace(module, compute_dtype, {})
        self._previous_order = []
        self._order = []
        for layer in self.layers:
            layer.register_forward_pre_hook(self._before_layer)
            layer.register_forward_hook(self._after_layer)
        sharded = {id(layer.weight) for layer in self.layers}
        self.replicated = [p for p in module.parameters() if id(p) not in sharded and p.requires_grad]
        self.pending = []
        for parameter in self.replicated:
            parameter.register_post_accumulate_grad_hook(self._reduce_replicated)

    def _replace(self, parent, dtype, replacements):
        for name, child in list(parent.named_children()):
            if isinstance(child, (Linear, Embedding, nn.Linear, nn.Embedding)):
                # A reused module is replaced once. Distinct modules tying the same
                # weight are not supported because their gradient reductions need
                # a shared accumulation schedule.
                if id(child.weight) in replacements:
                    raise ValueError("FSDP requires untied Linear/Embedding weights")
                layer_cls = _ShardedEmbedding if isinstance(child, (Embedding, nn.Embedding)) else _ShardedLinear
                layer = layer_cls(child, dtype)
                replacements[id(child.weight)] = layer
                setattr(parent, name, layer)
                self.layers.append(layer)
            else:
                self._replace(child, dtype, replacements)

    def _before_layer(self, layer, inputs):
        index = len(self._order)
        layer.prefetch()
        # At most one layer ahead: layer i+1 starts only after i-1 has
        # completed. The first iteration learns actual execution order.
        if (self._previous_order[:index] == self._order
                and index + 1 < len(self._previous_order)
                and self._previous_order[index] is layer):
            self._previous_order[index + 1].prefetch()

    def _after_layer(self, layer, inputs, output):
        self._order.append(layer)

    def _reduce_replicated(self, parameter):
        parameter.grad.div_(dist.get_world_size())
        self.pending.append(dist.all_reduce(parameter.grad, async_op=True))

    def forward(self, *args, **kwargs):
        self._order = []
        result = self.module(*args, **kwargs)
        self._previous_order = self._order.copy()
        # Dynamic branches may leave a speculative gather unused.
        for layer in self.layers:
            if layer._prefetched is not None:
                layer._prefetched[1].wait()
                layer._prefetched = None
        return result

    def finish_gradient_synchronization(self):
        for handle in self.pending:
            handle.wait()
        self.pending.clear()

    @torch.no_grad()
    def gather_full_params(self):
        result = {}
        modules = dict(self.module.named_modules())
        for name, parameter in self.module.named_parameters():
            prefix, _, leaf = name.rpartition(".")
            layer = modules[prefix]
            if isinstance(layer, _ShardedLayer) and leaf == "weight":
                full, _ = _gather(parameter, layer.shape, parameter.dtype)
                result[name] = full[:math.prod(layer.shape)].view(layer.shape)
            else:
                result[name] = parameter.detach().clone()
        return result
