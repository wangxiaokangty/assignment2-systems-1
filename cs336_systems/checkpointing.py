"""Activation checkpointing for sequential reference Transformer blocks."""
from types import MethodType
from torch.utils.checkpoint import checkpoint


def checkpoint_blocks(blocks, x, group_size=1):
    if group_size < 1:
        raise ValueError('group_size must be positive')
    for start in range(0, len(blocks), group_size):
        group = blocks[start:start + group_size]
        def run_group(value, group=group):
            for block in group:
                value = block(value)
            return value
        x = checkpoint(run_group, x, use_reentrant=False)
    return x


def checkpoint_model(model, group_size=1):
    """Change execution only; parameter names and state_dict remain unchanged."""
    def forward(self, tokens):
        hidden = checkpoint_blocks(self.layers, self.token_embeddings(tokens), group_size)
        return self.lm_head(self.ln_final(hidden))
    model.forward = MethodType(forward, model)
    return model


def recursive_checkpoint(blocks, x):
    """Binary recursive checkpointing: logarithmic checkpoints, N log N work."""
    if len(blocks) == 0:
        return x
    if len(blocks) == 1:
        return blocks[0](x)
    middle = len(blocks) // 2
    x = checkpoint(lambda value: recursive_checkpoint(blocks[:middle], value), x, use_reentrant=False)
    return checkpoint(lambda value: recursive_checkpoint(blocks[middle:], value), x, use_reentrant=False)
