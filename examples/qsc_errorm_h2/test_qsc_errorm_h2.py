from __future__ import annotations


def test_qsc_errorm_adds_ground_state_and_preserves_excited_spectrum():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    pytest.importorskip("pennylane")
    pytest.importorskip("pyscf")

    import QCANT

    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.735]], dtype=float)

    params, ash_excitation, energies = QCANT.adapt_vqe(
        symbols=symbols,
        geometry=geometry,
        adapt_it=1,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=2,
        active_orbitals=2,
        device_name="default.qubit",
        optimizer_maxiter=25,
    )

    vals_eom = QCANT.qscEOM(
        symbols=symbols,
        geometry=geometry,
        active_electrons=2,
        active_orbitals=2,
        charge=0,
        ansatz=(params, ash_excitation, energies),
        basis="sto-3g",
        method="pyscf",
        spin=0,
        shots=0,
    )[0].real

    vals_err = QCANT.qsc_errorm(
        symbols=symbols,
        geometry=geometry,
        active_electrons=2,
        active_orbitals=2,
        charge=0,
        ansatz=(params, ash_excitation, energies),
        basis="sto-3g",
        method="pyscf",
        spin=0,
        shots=0,
    )[0].real

    assert vals_err.size == vals_eom.size + 1
    assert np.isclose(vals_err[0], float(energies[-1]), atol=1e-7)
    assert np.allclose(vals_err[1:], vals_eom, atol=1e-7)
