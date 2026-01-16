"""QCANT: quantum computing utilities (chemistry/materials science).

This package is currently a lightweight scaffold created from the MolSSI
cookiecutter template. The public API is intentionally small and stable.

Public API
----------
- :func:`QCANT.canvas` – small example function used by the template.
- :data:`QCANT.__version__` – package version string.
"""

from .QCANT import canvas
from .adapt import aps_adapt
from .qsceom import aps_qscEOM
from ._version import __version__

__all__ = [
	"aps_adapt",
	"aps_qscEOM",
	"canvas",
	"__version__",
]
