"""Loads configs/config.yaml and resolves every path in it relative to the
project root (the directory containing configs/), so scripts can be invoked
from any working directory without hard-coded absolute paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class Config:
    def __init__(self, data: Dict[str, Any], root: Path):
        self._data = data
        self.root = root

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def path(self, *keys: str) -> Path:
        """Resolve a dotted path.path key (e.g. path('paths', 'outputs'))
        to an absolute Path anchored at the project root."""
        node: Any = self._data
        for k in keys:
            node = node[k]
        return (self.root / node).resolve()

    @property
    def raw(self) -> Dict[str, Any]:
        return self._data


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(data, root=path.resolve().parent.parent)
