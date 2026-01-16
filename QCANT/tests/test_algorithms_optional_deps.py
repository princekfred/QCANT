"""Tests for algorithm entry points without heavy optional dependencies.

These tests are intentionally lightweight:
- validate argument checking that should work without PennyLane/PySCF,
- validate that a clear ImportError is raised when optional deps are missing.
"""

from __future__ import annotations

import pytest

import QCANT


def test_adapt_vqe_geometry_length_mismatch_raises_value_error():
    symbols = ["H", "H"]
    geometry = [[0.0, 0.0, 0.0]]

    with pytest.raises(ValueError, match=r"geometry must have the same length as symbols"):
        QCANT.adapt_vqe(
            symbols=symbols,
            geometry=geometry,
            adapt_it=1,
            active_electrons=2,
            active_orbitals=2,
        )


def test_adapt_vqe_missing_optional_deps_raises_import_error():
    symbols = ["H", "H"]
    geometry = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

    with pytest.raises(ImportError) as excinfo:
        QCANT.adapt_vqe(
            symbols=symbols,
            geometry=geometry,
            adapt_it=1,
            active_electrons=2,
            active_orbitals=2,
        )

    # Should guide users toward installing missing deps.
    msg = str(excinfo.value).lower()
    assert "pennylane" in msg or "pyscf" in msg


def test_qsceom_missing_optional_deps_raises_import_error():
    symbols = ["H", "H"]
    geometry = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

    with pytest.raises(ImportError) as excinfo:
        QCANT.qscEOM(
            symbols=symbols,
            geometry=geometry,
            active_electrons=2,
            active_orbitals=2,
            charge=0,
            params=[0.0],
            ash_excitation=[[0, 1]],
        )

    assert "pennylane" in str(excinfo.value).lower()
