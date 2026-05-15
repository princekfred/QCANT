"""Input file for running Chebyshev-QKUD.

Run from the same folder as chebyshev_qkud.py:

    python input.py

Required packages:

    pip install numpy scipy pennylane pyscf
"""

import numpy as np

from cqkud import chebyshev_qkud


def main():
    # Example: H2 molecule in STO-3G.
    # Coordinates are in Angstrom.
    symbols = ["H", "H"]

    geometry = np.array(
        [
            [0.0, 0.0, 0],
            [0.0, 0.0, 0.735],
        ],
        dtype=float,
    )

    energies, basis_states, S, H_K, energy_history = chebyshev_qkud(
        symbols=symbols,
        geometry=geometry,
        n_steps=4,
        epsilon=0.10,
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

    print("\n=== Chebyshev-QKUD results ===")
    print("All Ritz energies:")
    print(energies)

    print("\nGround-state estimate:")
    print(energies[0])

    print("\nBasis shape:")
    print(basis_states.shape)

    print("\nOverlap matrix S:")
    print(S)

    print("\nProjected Hamiltonian H_K:")
    print(H_K)

    print("\nGround-state energy history:")
    for step, value in enumerate(energy_history):
        print(f"basis size {step + 1}: {value:.12f}")


if __name__ == "__main__":
    main()
