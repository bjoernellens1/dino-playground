from typing import Any, Callable, Dict, Optional

class Registry:
    """
    A simple registry for mapping strings to classes/functions.
    """

    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={list(self._module_dict.keys())})"

    def register(self, name: Optional[str] = None) -> Callable:
        """
        Decorator to register a class or function.
        """
        def _register(obj: Any) -> Any:
            key = name if name is not None else obj.__name__
            if key in self._module_dict:
                raise KeyError(f"{key} is already registered in {self._name}")
            self._module_dict[key] = obj
            return obj
        return _register

    def get(self, name: str) -> Any:
        """
        Get a registered object by name.
        """
        if name not in self._module_dict:
            raise KeyError(f"{name} is not registered in {self._name}")
        return self._module_dict[name]

# Global registries
BACKBONES = Registry("backbones")
DATASETS = Registry("datasets")
HEADS = Registry("heads")
LOSSES = Registry("losses")
