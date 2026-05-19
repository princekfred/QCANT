"""Input file for H4 Chebyshev-QKUD.

Run from the QCANT repo root with:

    python3.9 QCANT/cqkud/h4_cqkud.py

or from this folder with:

    python3.9 h4_cqkud.py
"""

from pathlib import Path

import numpy as np

from orth_cqkud import chebqkud



OUTPUT_FILE = Path(__file__).with_name("h4_cqkud_output.txt")
PLOT_FILE = Path(__file__).with_name("h4_cqkud_condition_numbers.png")


def _condition_number_history(basis_states, tol=1e-12):
    history = []
    for n_step in range(basis_states.shape[0]):
        current_basis = basis_states[: n_step + 1]
        s_prefix = current_basis.conj() @ current_basis.T
        singular_values = np.linalg.svd(s_prefix, compute_uv=False)
        max_sv = float(singular_values[0])
        min_sv = float(singular_values[-1])
        rank = int(np.count_nonzero(singular_values > float(tol)))
        cond = float("inf") if min_sv <= 0.0 else max_sv / min_sv
        history.append(
            {
                "n_step": n_step,
                "basis_size": n_step + 1,
                "condition_number": cond,
                "min_singular_value": min_sv,
                "max_singular_value": max_sv,
                "rank": rank,
            }
        )
    return history


def _format_float(value):
    if np.isinf(value):
        return "inf"
    return f"{float(value):.12e}"


def _format_results(
    energies,
    basis_states,
    s_matrix,
    h_k,
    energy_history,
    condition_history,
):
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
    lines.extend(
        [
            "",
            "Overlap matrix condition number by n_step:",
            "n_step\tbasis_size\tcondition_number\tmin_singular_value\tmax_singular_value\trank",
        ]
    )
    for entry in condition_history:
        lines.append(
            f"{entry['n_step']}\t"
            f"{entry['basis_size']}\t"
            f"{_format_float(entry['condition_number'])}\t"
            f"{_format_float(entry['min_singular_value'])}\t"
            f"{_format_float(entry['max_singular_value'])}\t"
            f"{entry['rank']}"
        )
    return "\n".join(lines) + "\n"


def _write_condition_number_plot(condition_history, plot_file):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    basis_sizes = [entry["basis_size"] for entry in condition_history]
    condition_numbers = [entry["condition_number"] for entry in condition_history]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(basis_sizes, condition_numbers, marker="o", linewidth=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("Krylov basis size")
    ax.set_ylabel("Condition number of S")
    ax.set_title("H4 CQKUD Overlap Matrix Conditioning")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    fig.savefig(plot_file, dpi=300)
    plt.close(fig)


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
    condition_history = _condition_number_history(basis_states)
    OUTPUT_FILE.write_text(
        _format_results(
            energies,
            basis_states,
            s_matrix,
            h_k,
            energy_history,
            condition_history,
        ),
        encoding="utf-8",
    )
    _write_condition_number_plot(condition_history, PLOT_FILE)
    print(f"Wrote H4 CQKUD output to: {OUTPUT_FILE}")
    print(f"Wrote H4 CQKUD condition-number plot to: {PLOT_FILE}")


if __name__ == "__main__":
    main()
