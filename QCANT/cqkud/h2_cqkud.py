"""Input file for H2 Chebyshev-QKUD at bond distance 0.735 Angstrom.

Run from the QCANT repo root with:

    python3.9 QCANT/cqkud/h2_cqkud.py

or from this folder with:

    python3.9 h2_cqkud.py
"""

import numpy as np

from cqkud import chebqkud


def main():
    symbols = ["H", "H"]
    geometry = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.735],
        ],
        dtype=float,
    )

    energies, basis_states, s_matrix, h_k, energy_history = chebqkud(
        symbols=symbols,
        geometry=geometry,
        n_steps=4,
        active_electrons=2,
        active_orbitals=2,
        basis="sto-3g",
        charge=0,
        spin=0,
        method="pyscf",
        use_sparse=False,
        overlap_tol=1e-10,
        return_matrices=True,
        return_min_energy_history=True,
    )

    np.set_printoptions(precision=12, suppress=True)

    print("\n=== H2 Chebyshev-QKUD ===")
    print("Bond distance: 0.735 Angstrom")
    print("\nRitz energies:")
    print(energies)
    print("\nGround-state estimate:")
    print(energies[0])
    print("\nBasis shape:")
    print(basis_states.shape)
    print("\nOverlap matrix S:")
    print(s_matrix)
    print("\nProjected Hamiltonian H_K:")
    print(h_k)
    print("\nGround-state energy history:")
    for step, value in enumerate(energy_history):
        print(f"basis size {step + 1}: {value:.12f}")


if __name__ == "__main__":
    main()
