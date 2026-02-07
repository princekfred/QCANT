"""qsc_errorm method.

This subpackage contains a small variation of qscEOM that includes the identity
(reference) configuration in the effective subspace so the ground state can be
recovered alongside excited states.
"""

from .qsc_errorm import qsc_errorm

__all__ = [
    "qsc_errorm",
]

