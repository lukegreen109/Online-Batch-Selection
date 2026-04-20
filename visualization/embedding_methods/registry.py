"""
visualization/embedding_methods/registry.py

Central registry for dimensionality reduction methods.  Kept in its own
file so method modules can import `register` without circular imports.
"""

from .base import EmbeddingMethod

_REGISTRY: dict[str, type[EmbeddingMethod]] = {}


def register(name: str):
    """
    Class decorator — self-registers an EmbeddingMethod under *name*.

    Add this to any new method file and it will be picked up automatically:

        from visualization.embedding_methods.registry import register
        from visualization.embedding_methods.base import EmbeddingMethod

        @register("mymethod")
        class MyMethod(EmbeddingMethod):
            ...

    Then reference it by name in your visualization config:
        embedding_methods: ["mymethod"]
    """
    def decorator(cls: type) -> type:
        if not (isinstance(cls, type) and issubclass(cls, EmbeddingMethod)):
            raise TypeError(
                f"@register target must be a subclass of EmbeddingMethod, got {cls!r}"
            )
        _REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_method(name: str, **kwargs) -> EmbeddingMethod:
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown embedding method '{name}'. "
            f"Available: {list(_REGISTRY)}. "
            "Register custom methods with @register('name') from visualization.embedding_methods.registry."
        )
    return _REGISTRY[key](**kwargs)


def list_methods() -> list[str]:
    return list(_REGISTRY.keys())
