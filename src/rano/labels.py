"""
The one bridge between the package and the locked ``label_schema.py`` at the project root.

``label_schema.py`` is the declared single source of truth for integer -> compartment, and it
lives at the repo root by decision, not inside ``src/rano/``. With a src-layout install only
``src/`` lands on ``sys.path``, so a bare ``import label_schema`` from a package module works
only when the process happens to be started from the project root — a silent, cwd-dependent
trap of exactly the kind this project exists to avoid.

So: every package module imports label semantics from HERE, and this is the only file that
knows where the schema physically lives. If the schema is ever moved into the package, this
file collapses to a plain re-export and nothing else changes.

Nothing is redefined below — the maps are re-exported by reference, never copied, so there is
still exactly one place where an integer is written down.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

#: src/rano/labels.py -> src/rano -> src -> <project root>
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = _PROJECT_ROOT / "label_schema.py"


def _load() -> ModuleType:
    """Import the root ``label_schema`` module, preferring an already-importable one."""
    if "label_schema" in sys.modules:
        return sys.modules["label_schema"]
    try:  # cwd is the project root, or it's on sys.path some other way
        import label_schema as mod  # type: ignore[import-not-found]

        return mod
    except ModuleNotFoundError:
        pass

    if not _SCHEMA_PATH.is_file():
        raise ModuleNotFoundError(
            f"label_schema.py not found at {_SCHEMA_PATH}. It is the locked single source of "
            "truth for label integers; rano cannot compute volumes without it."
        )
    spec = importlib.util.spec_from_file_location("label_schema", _SCHEMA_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["label_schema"] = mod
    spec.loader.exec_module(mod)
    return mod


label_schema = _load()

LABEL_SCHEMA: dict[str, dict[int, str]] = label_schema.LABEL_SCHEMA
COMPOSITE_REGIONS: dict[str, dict[str, set[int]]] = label_schema.COMPOSITE_REGIONS
CANONICAL_SOURCE: str = label_schema.CANONICAL_SOURCE

__all__ = ["LABEL_SCHEMA", "COMPOSITE_REGIONS", "CANONICAL_SOURCE", "label_schema"]
