"""Whole-parameter optimizer-state sharding, followed by owner broadcasts."""
import torch
from torch import distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.loads = [0] * self.world_size
        self.owners = {}
        self.local_optimizer = None
        super().__init__(params, kwargs)
        local_groups = [self._local_group(group) for group in self.param_groups]
        self.local_optimizer = optimizer_cls(local_groups, **kwargs)
        self.state = self.local_optimizer.state
        self.defaults = self.local_optimizer.defaults.copy()
        for group, local in zip(self.param_groups, self.local_optimizer.param_groups):
            group.update({key: value for key, value in local.items() if key != "params"})

    def _local_group(self, group):
        return {**group, "params": [p for p in group["params"] if self.owners[p] == self.rank]}

    def add_param_group(self, param_group):
        # Let PyTorch validate duplicates and normalize generators before assignment.
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        for parameter in group["params"]:
            if parameter not in self.owners:
                owner = min(range(self.world_size), key=self.loads.__getitem__)
                self.owners[parameter] = owner
                self.loads[owner] += parameter.numel()
        if self.local_optimizer is not None:
            self.local_optimizer.add_param_group(self._local_group(group))

    def step(self, closure=None, **kwargs):
        # Mirror scheduler/user changes to group hyperparameters.
        for group, local in zip(self.param_groups, self.local_optimizer.param_groups):
            local.update({key: value for key, value in group.items() if key != "params"})
        loss = self.local_optimizer.step(closure=closure, **kwargs)
        handles = [dist.broadcast(p.data, src=owner, async_op=True)
                   for p, owner in self.owners.items()]
        for handle in handles:
            handle.wait()
        return loss

    def state_dict(self):
        """Return this rank's optimizer checkpoint (save one file per rank)."""
        return self.local_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.local_optimizer.load_state_dict(state_dict)
        self.state = self.local_optimizer.state
        for group, local in zip(self.param_groups, self.local_optimizer.param_groups):
            group.update({key: value for key, value in local.items() if key != "params"})
