"""Input file for H4 Chebyshev-QKUD.

Run from the QCANT repo root with:

    python3.9 QCANT/cqkud/h4_cqkud.py

or from this folder with:

    python3.9 h4_cqkud.py
"""

from pathlib import Path

import numpy as np

from cqkud import chebqkud


OUTPUT_FILE = Path(__file__).with_name("h4_cqkud_output.txt")


def _format_results(energies, basis_states, s_matrix, h_k, energy_history):
    lines = [
        "=== H4 Chebyshev-QKUD ===",
        "",
        "Ritz energies:",
        np.array2string(energies),
        "",
        "Ground-state estimate:",
        f"{float(energies[0]):.12f}",
        "",
       # "Basis shape:",
       # str(basis_states.shape),
        "",
       # "Overlap matrix S:",
       # np.array2string(s_matrix),
        "",
       # "Projected Hamiltonian H_K:",
       # np.array2string(h_k),
        "",
        "Ground-state energy convergence:",
    ]
    for step, value in enumerate(energy_history):
        lines.append(f"basis size {step + 1}: {value:.12f}")
    return "\n".join(lines) + "\n"


def main():
    symbols = ["H", "H", "H", "H"]
    geometry = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3],
            [0.0, 0.0, 6],
            [0.0, 0.0, 9]
        ],
        dtype=float,
    )

    energies, basis_states, s_matrix, h_k, energy_history = chebqkud(
        symbols=symbols,
        geometry=geometry,
        n_steps=27,
        active_electrons=4,
        active_orbitals=4,
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
    OUTPUT_FILE.write_text(
        _format_results(energies, basis_states, s_matrix, h_k, energy_history),
        encoding="utf-8",
    )
    print(f"Wrote H4 CQKUD output to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
