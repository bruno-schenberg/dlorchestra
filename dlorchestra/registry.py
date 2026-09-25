"""Generic name -> builder registry.

Adding an entry = one call to ``register()``. The registry stores the
constructor and an arbitrary metadata dict (e.g. the dotted attribute path
to a model's final classifier layer) — everything else is up to the caller.
"""

from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RegistryEntry(Generic[T]):
    build: Callable[..., T]
    metadata: dict = field(default_factory=dict)


class Registry(Generic[T]):
    """Case-insensitive name -> builder registry.

    Example::

        models = Registry[nn.Module]()
        models.register("MY_MODEL", _build_my_model, classifier_attr="fc")
        model = models.build("my_model", num_classes=2)
    """

    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry[T]] = {}

    def register(self, name: str, build: Callable[..., T], **metadata) -> None:
        key = name.upper()
        if key in self._entries:
            raise ValueError(f"'{name}' is already registered")
        self._entries[key] = RegistryEntry(build=build, metadata=metadata)

    def get(self, name: str) -> RegistryEntry[T]:
        key = name.upper()
        entry = self._entries.get(key)
        if entry is None:
            raise ValueError(f"Unknown '{name}'. Registered: {self.names}")
        return entry

    def build(self, name: str, **kwargs) -> T:
        return self.get(name).build(**kwargs)

    @property
    def names(self) -> list[str]:
        return list(self._entries)

    def __contains__(self, name: str) -> bool:
        return name.upper() in self._entries
