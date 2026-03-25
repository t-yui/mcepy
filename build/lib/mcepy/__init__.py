"""mcepy: Median Consensus Embedding utilities."""

from ._core import MCE, drmce, normalize_embedding, tsnemce, umapmce
from ._version import __version__

__all__ = [
    "__version__",
    "MCE",
    "drmce",
    "normalize_embedding",
    "tsnemce",
    "umapmce",
]
