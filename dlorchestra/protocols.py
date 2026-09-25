"""Type-only contracts for the hooks TorchExperiment calls back into.

These are documentation/type-checking aids, not base classes to subclass —
a plain function or an ``nn.Module`` already satisfies them structurally.
"""

from typing import Any, Protocol

MetricsDict = dict[str, float]


class BaseModel(Protocol):
    """What TorchExperiment requires of a model: forward() + state_dict()/load_state_dict()."""

    def __call__(self, *args, **kwargs) -> Any: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state_dict: dict, strict: bool = True) -> Any: ...
    def parameters(self): ...
    def to(self, device: str) -> "BaseModel": ...


class BaseDataLoader(Protocol):
    """What TorchExperiment requires of a dataloader: iterable of batches, sized."""

    def __iter__(self): ...
    def __len__(self) -> int: ...
