"""Pytest bootstrap: put the repo root and ``scripts/`` on ``sys.path``.

Historically each test file carried its own ``sys.path.insert`` line, and the
suite only worked under ``python -m pytest`` run from the repo root — the
``-m`` form puts the CWD on ``sys.path`` as a side effect.  The ``pytest``
console script omitted it, so a fresh checkout failed collection with
``ModuleNotFoundError: No module named 'scripts'``.

Test files use two different import styles, and both are live:

  * ``from scripts.postprocess_textgrids import ...``  -> needs repo root
  * ``from postprocess_textgrids import ...``          -> needs ``scripts/``

Adding only one of the two breaks the other half, so both entries are
inserted.  ``scripts/`` is inserted first so that root lands at index 0 and
``scripts/`` at index 1: a bare import misses on root (which holds no script
modules) and then resolves from ``scripts/``, while ``scripts.X`` resolves
through root.

``scripts/__init__.py`` must NOT be created — ``scripts`` is importable purely
as an implicit namespace package, and adding an ``__init__.py`` would change
that.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

for _entry in (SCRIPTS, ROOT):
    _resolved = str(_entry)
    # Re-insert at the front rather than appending, and never duplicate, so the
    # list does not grow once per collected test module.
    while _resolved in sys.path:
        sys.path.remove(_resolved)
    sys.path.insert(0, _resolved)
