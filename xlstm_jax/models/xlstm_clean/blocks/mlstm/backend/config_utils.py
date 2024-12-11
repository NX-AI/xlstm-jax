from dataclasses import dataclass, field
from typing import Any


@dataclass
class NameAndKwargs:
    name: str = ""
    kwargs: dict[str, Any] | None = field(default_factory=dict)
    _registry: dict[str, type] | None = field(default_factory=dict)

    def get_config_class_for_kwargs(self) -> type:
        if self.name in self._registry:
            return self._registry[self.name].config_class
        raise ValueError(
            f"Unknown {self.__class__} name: {self.name}. Available {self.__class__} are: {list(self._registry.keys())}"
        )

    def get_class_for_name(self) -> type:
        if self.name in self._registry:
            return self._registry[self.name]
        raise ValueError(
            f"Unknown {self.__class__} name: {self.name}. Available {self.__class__} are: {list(self._registry.keys())}"
        )

    def create_name_instance(self) -> Any:
        return self.get_class_for_name()(self.kwargs)