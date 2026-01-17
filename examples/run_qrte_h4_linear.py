"""Run QRTE for linear H4 and plot energies vs iteration.

This script runs QCANT.qrte on a linear H4 chain for multiple timestep values
(delta_t) and plots the lowest projected-basis energy at each iteration.

Usage
-----
From the repo root:

    python examples/run_qrte_h4_linear.py

Notes
-----
- Requires the full QCANT runtime deps (NumPy, SciPy, PennyLane, PySCF, autoray).
- The QRTE implementation currently constructs a dense Hamiltonian matrix, so this
  will only be practical for small active spaces.
"""

from __future__ import annotations

import numpy as np

import QCANT


def compute_active_space_fci_energy(
    *,
    symbols: list[str],
    geometry: np.ndarray,
    basis: str,
    charge: int,
    spin: int,
    active_electrons: int,
    active_orbitals: int,
) -> float:
    """Compute the exact ground-state energy (FCI in the chosen active space).

    This is done by constructing the same qubit Hamiltonian used by QRTE and
    dense-diagonalizing it. For the default H4 active space (8 qubits), this is
    inexpensive and provides a clean reference for log-scale convergence plots.
    """

    import pennylane as qml

    H, qubits = qml.qchem.molecular_hamiltonian(
        symbols,
        geometry,
        basis=basis,
        charge=charge,
        mult=spin + 1,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        method="pyscf",
    )

    wires = list(range(qubits))
    H_dense = qml.matrix(H, wire_order=wires)
    return float(np.linalg.eigvalsh(H_dense).min().real)


def make_linear_hn_geometry(n: int, bond_length: float = 1.5) -> np.ndarray:
    """Return a linear H_n geometry with uniform spacing (Angstrom)."""

    if n < 2:
        raise ValueError("n must be >= 2")
    z = bond_length * np.arange(n, dtype=float)
    return np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)


def make_linear_h4_geometry(bond_length: float = 1.5) -> np.ndarray:
    """Return a linear H4 geometry with uniform spacing (Angstrom)."""

    return make_linear_hn_geometry(4, bond_length=bond_length)


def make_linear_h6_geometry(bond_length: float = 1.5) -> np.ndarray:
    """Return a linear H6 geometry with uniform spacing (Angstrom)."""

    return make_linear_hn_geometry(6, bond_length=bond_length)


def run_for_delta_t(delta_t: float, n_steps: int) -> np.ndarray:
    """Run QRTE and collect the lowest energy after each iteration (1..n_steps)."""
    symbols = ["H", "H", "H", "H"]
    geometry = make_linear_h4_geometry(bond_length=1.5)

    # Choose a small active space for practicality.
    # For H4 sto-3g, 4 electrons / 4 orbitals is a common minimal choice.
    active_electrons = 4
    active_orbitals = 4

    energies_over_time = []
    for k in range(1, n_steps + 1):
        energies, _basis_states, _times = QCANT.qrte(
            symbols=symbols,
            geometry=geometry,
            delta_t=delta_t,
            n_steps=k,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            basis="sto-3g",
            charge=0,
            spin=0,
            device_name="default.qubit",
            trotter_steps=1,
        )
        energies_over_time.append(float(np.min(energies)))

    return np.array(energies_over_time, dtype=float)


def main() -> None:
    import matplotlib.pyplot as plt

    delta_ts = [10]
    n_steps = 50

    plt.figure(figsize=(9, 5))

    iterations = np.arange(1, n_steps + 1)

    symbols = ["H"] * 6
    geometry = make_linear_h6_geometry(bond_length=5.0)
    basis = "sto-3g"
    charge = 0
    spin = 0
    active_electrons = 6
    active_orbitals = 6

    e_fci = compute_active_space_fci_energy(
        symbols=symbols,
        geometry=geometry,
        basis=basis,
        charge=charge,
        spin=spin,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )
    print(f"Active-space FCI reference energy (Hartree): {e_fci:.12f}")

    for dt in delta_ts:
        energies, _, _, min_energy_history = QCANT.qrte(
            symbols=symbols,
            geometry=geometry,
            delta_t=dt,
            n_steps=n_steps,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            basis=basis,
            charge=charge,
            spin=spin,
            device_name="default.qubit",
            trotter_steps=1,
            return_min_energy_history=True,
        )

        e_min = np.asarray(min_energy_history, dtype=float)

        # Log scale requires positive values; use the variational gap to the
        # (active-space) FCI energy.
        e_shift = e_min - e_fci
        if np.any(e_shift < -1e-8):
            print(
                f"Warning: E_min - E_FCI has negative values (min={e_shift.min():.3e}); "
                "check that FCI/QRTE use identical system and active space."
            )
        e_shift = np.maximum(e_shift, 0.0)

        print(f"\nΔt = {dt}")
        print("iter\tE_min (Ha)\t\tE_min - E_FCI (Ha)")
        for k, (e_k, gap_k) in enumerate(zip(e_min, e_min - e_fci), start=1):
            print(f"{k:>4d}\t{e_k: .12f}\t{gap_k: .12e}")

        plt.plot(iterations, e_shift, marker="o", label=f"Δt={dt}")

    plt.xlabel("Iteration")
    plt.ylabel("E_min(k) − E_FCI (Hartree)")
    plt.title("QRTE on linear H6 (sto-3g)")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
