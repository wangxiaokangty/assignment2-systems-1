"""NVTX instrumentation and saved-activation accounting."""
import math
import torch
from cs336_basics.nn_utils import softmax


def annotated_attention(Q, K, V, mask=None):
    with torch.cuda.nvtx.range('attention'):
        with torch.cuda.nvtx.range('attention_scores'):
            scores = Q @ K.transpose(-1, -2) / math.sqrt(K.shape[-1])
            if mask is not None:
                scores = torch.where(mask, scores, float('-inf'))
        with torch.cuda.nvtx.range('attention_softmax'):
            weights = softmax(scores, -1)
        with torch.cuda.nvtx.range('attention_values'):
            return weights @ V


def annotate_attention():
    import cs336_basics.model
    cs336_basics.model.scaled_dot_product_attention = annotated_attention


class SavedTensorInventory:
    """Count distinct saved storages, excluding model parameters (including views)."""
    def __init__(self, module):
        self.parameters = {p.untyped_storage().data_ptr() for p in module.parameters()}
        self.records = {}

    def pack(self, tensor):
        storage = tensor.untyped_storage()
        key = storage.data_ptr()
        if key not in self.parameters and key not in self.records:
            self.records[key] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                                 'bytes': storage.nbytes(), 'producer': str(tensor.grad_fn).split(' object')[0]}
        return tensor

    @staticmethod
    def unpack(tensor):
        return tensor


class ToyModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, 10, bias=False)
        self.ln = torch.nn.LayerNorm(10)
        self.fc2 = torch.nn.Linear(10, out_features, bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.fc2(self.ln(self.relu(self.fc1(x))))


def mixed_precision_dtypes(device='cuda'):
    model = ToyModel(8, 4).to(device)
    observations = {}
    handles = []
    for name in ('fc1', 'ln', 'fc2'):
        def record(module, inputs, output, name=name):
            observations[name] = str(output.dtype)
        handles.append(getattr(model, name).register_forward_hook(record))
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16):
        output = model(torch.randn(16, 8, device=device))
        loss = torch.nn.functional.cross_entropy(output, torch.randint(4, (16,), device=device))
        observations['parameters'] = sorted({str(p.dtype) for p in model.parameters()})
        observations['loss'] = str(loss.dtype)
    loss.backward()
    observations['gradients'] = sorted({str(p.grad.dtype) for p in model.parameters()})
    for handle in handles:
        handle.remove()
    return observations


def block_memory_inventory(context_length=2048, device='cuda'):
    from collections import defaultdict
    from cs336_basics.model import TransformerBlock, RotaryEmbedding
    with torch.device(device):
        block = TransformerBlock(2560, 32, 10240, RotaryEmbedding(context_length, 80))
        x = torch.randn(4, context_length, 2560, requires_grad=True)
    inventory = SavedTensorInventory(block)
    baseline = torch.cuda.memory_allocated(device)
    with torch.autograd.graph.saved_tensors_hooks(inventory.pack, inventory.unpack):
        y = block(x)
    torch.cuda.synchronize(device)
    after_forward = torch.cuda.memory_allocated(device)
    y.sum().backward()
    torch.cuda.synchronize(device)
    after_backward = torch.cuda.memory_allocated(device)
    grouped = defaultdict(int)
    for record in inventory.records.values():
        grouped[record['producer']] += record['bytes']
    total = sum(grouped.values())
    return {'context_length': context_length, 'baseline_bytes': baseline,
            'after_forward_bytes': after_forward, 'after_backward_bytes': after_backward,
            'saved_storage_bytes': total,
            'parameter_gradient_bytes': sum(p.grad.numel()*p.grad.element_size() for p in block.parameters()),
            'input_gradient_bytes': x.grad.numel()*x.grad.element_size(),
            'saved_by_producer': [{'producer': k, 'bytes': v, 'percent': v/total*100}
                                  for k,v in sorted(grouped.items(), key=lambda kv: -kv[1])],
            'storages': sorted(inventory.records.values(), key=lambda row: -row['bytes'])}
