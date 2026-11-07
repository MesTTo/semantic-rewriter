"""Purpose: the engine registry: make(name) builds a logic engine implementing
semrewrite.protocols.Engine, importing its backend only when asked, so the
pure-Python engine works without PyMeTTa installed.

    from semrewrite.engines import make
    engine = make("python")          # always available
    engine = make("pymetta")         # needs the pymetta package
"""

from __future__ import annotations

import importlib

from semrewrite.protocols import Engine

#: Engine name -> (module, class); a new backend adds one line.
REGISTRY: dict[str, tuple[str, str]] = {
    "python": ("semrewrite.engines.python", "PythonEngine"),
    "pymetta": ("semrewrite.engines.pymetta", "PyMeTTaEngine"),
}


def make(name: str, **options: object) -> Engine:
    """The engine registered under `name`, built with `options`; an unknown name raises listing the known ones."""
    if name not in REGISTRY:
        msg = f"unknown engine {name!r}; known engines: {', '.join(sorted(REGISTRY))}"
        raise KeyError(msg)
    module, cls = REGISTRY[name]
    return getattr(importlib.import_module(module), cls)(**options)


def available() -> list[str]:
    """The registered engines whose backend imports in this environment."""
    out = []
    for name, (module, _) in REGISTRY.items():
        try:
            importlib.import_module(module)
        except ImportError:
            continue
        out.append(name)
    return out
