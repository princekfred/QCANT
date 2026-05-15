"""Chebyshev Quantum Krylov using Unitary Decomposition (CQKUD).

This module follows the Chebyshev-QKUD construction:

    B = (eta * I - H) / alpha
    B_epsilon = sin(epsilon * B) / epsilon

and builds the Chebyshev Krylov basis

    |chi_0> = |0>
    |chi_1> = B_epsilon |0>
    |chi_{n+1}> = 2 B_epsilon |chi_n> - |chi_{n-1}>

The projected Hamiltonian problem is then

    H_K c = E S c,

where

    S_ij  = <chi_i | chi_j>
    H_Kij = <chi_i | H | chi_j>.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple


def chebyshev_qkud(
    symbols: Sequence[str],
    geometry,
    *,
    n_steps: int,
    epsilon: float,
    active_electrons: int,
    active_orbitals: int,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    method: str = "pyscf",
    initial_state: Optional["object"] = None,
    energy_bounds: Optional[Tuple[float, float]] = None,
    eta: Optional[float] = None,
    alpha: Optional[float] = None,
    overlap_tol: float = 1e-10,
    use_sparse: bool = False,
    return_matrices: bool = False,
    return_min_energy_history: bool = False,
):
    """Run Chebyshev-QKUD for a molecular Hamiltonian.

    Parameters
    ----------
    symbols
        Atomic symbols, e.g. ["H", "H"].
    geometry
        Nuclear coordinates with shape (n_atoms, 3), in Angstrom.
    n_steps
        Highest Chebyshev degree. The returned basis has n_steps + 1 vectors:
        chi_0, chi_1, ..., chi_n_steps.
    epsilon
        Dimensionless deformation parameter in B_epsilon = sin(epsilon B)/epsilon.
        A typical first test value is 0.05 to 0.2.
    active_electrons
        Number of active electrons.
    active_orbitals
        Number of active orbitals.
    basis
        Atomic basis set, e.g. "sto-3g".
    charge
        Molecular charge.
    spin
        PySCF spin parameter, 2S. For singlet, use 0.
    method
        PennyLane quantum chemistry backend.
    initial_state
        Optional initial statevector. If None, Hartree-Fock state is used.
    energy_bounds
        Optional tuple (E_min, E_max). If provided, uses

            eta = (E_max + E_min) / 2
            alpha = (E_max - E_min) / 2

        unless eta and alpha are explicitly supplied.
    eta
        Optional energy shift. Used with alpha if both are supplied.
    alpha
        Optional positive energy scale. Used with eta if both are supplied.
    overlap_tol
        Small-eigenvalue cutoff for solving the generalized eigenvalue problem.
    use_sparse
        If True, use sparse Hamiltonian operations where possible.
    return_matrices
        If True, also return S and H_K.
    return_min_energy_history
        If True, also return the minimum Ritz energy after each basis expansion.

    Returns
    -------
    tuple
        Always starts with:

            energies, basis_states

        Optional returned values are appended in this order:

            S, H_K, min_energy_history
    """

    if len(symbols) == 0:
        raise ValueError("symbols must be non-empty")
    if n_steps < 0:
        raise ValueError("n_steps must be >= 0")
    if epsilon == 0:
        raise ValueError("epsilon must be nonzero")
    if overlap_tol <= 0:
        raise ValueError("overlap_tol must be > 0")

    try:
        n_atoms = len(symbols)
        if len(geometry) != n_atoms:
            raise ValueError
    except Exception as exc:
        raise ValueError("geometry must have the same length as symbols") from exc

    try:
        import numpy as np
        import pennylane as qml
        from scipy.sparse import issparse
        from scipy.sparse.linalg import expm_multiply, eigsh
    except ImportError as exc:
        raise ImportError(
            "chebyshev_qkud requires dependencies. Install them with:\n"
            "    pip install numpy scipy pennylane pyscf"
        ) from exc

    def _basis_state_vector(bits: Sequence[int]):
        idx = 0
        for bit in bits:
            idx = (idx << 1) | int(bit)

        state = np.zeros(2 ** len(bits), dtype=complex)
        state[idx] = 1.0
        return state

    def _get_hamiltonian_matrix(operator, n_qubits):
        wires = range(n_qubits)

        if use_sparse and hasattr(operator, "sparse_matrix"):
            try:
                return operator.sparse_matrix(wire_order=wires, format="csr")
            except TypeError:
                mat = operator.sparse_matrix(wire_order=wires)
                return mat.tocsr() if hasattr(mat, "tocsr") else mat

        return qml.matrix(operator, wire_order=wires)

    def _estimate_energy_bounds(H_mat):
        """Estimate E_min and E_max of H."""
        dim = H_mat.shape[0]

        if issparse(H_mat):
            if dim <= 2:
                dense = H_mat.toarray()
                vals = np.linalg.eigvalsh(dense)
                return float(vals[0]), float(vals[-1])

            e_min = eigsh(H_mat, k=1, which="SA", return_eigenvectors=False)[0]
            e_max = eigsh(H_mat, k=1, which="LA", return_eigenvectors=False)[0]
            return float(e_min), float(e_max)

        vals = np.linalg.eigvalsh(H_mat)
        return float(vals[0]), float(vals[-1])

    def _projected_problem(current_basis_states):
        """Build S, H_K and solve H_K c = E S c."""
        S = current_basis_states.conj() @ current_basis_states.T
        H_K = current_basis_states.conj() @ (H_mat @ current_basis_states.T)

        # Clean small numerical non-Hermitian noise.
        S = 0.5 * (S + S.conj().T)
        H_K = 0.5 * (H_K + H_K.conj().T)

        s_vals, s_vecs = np.linalg.eigh(S)
        keep = s_vals > float(overlap_tol)

        if not np.any(keep):
            raise ValueError(
                "Overlap matrix is numerically singular. "
                "Try fewer Krylov vectors, a different epsilon, or a smaller overlap_tol."
            )

        # Transform generalized problem H_K c = E S c into an ordinary
        # Hermitian eigenproblem in the support of S.
        X = s_vecs[:, keep] / np.sqrt(s_vals[keep])[None, :]
        H_ortho = X.conj().T @ H_K @ X
        H_ortho = 0.5 * (H_ortho + H_ortho.conj().T)

        energies = np.linalg.eigvalsh(H_ortho).real
        return energies, S, H_K

    # ------------------------------------------------------------------
    # Build molecular Hamiltonian H.
    # ------------------------------------------------------------------
    import numpy as np
    import pennylane as qml
    from scipy.sparse.linalg import expm_multiply

    mult = int(spin) + 1

    mol = qml.qchem.Molecule(
        symbols,
        geometry,
        charge=int(charge),
        mult=mult,
        basis_name=basis,
        unit="angstrom",
    )

    H, n_qubits = qml.qchem.molecular_hamiltonian(
        mol,
        method=method,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

    H_mat = _get_hamiltonian_matrix(H, n_qubits)

    # ------------------------------------------------------------------
    # Initial reference state |0>.
    # ------------------------------------------------------------------
    if initial_state is None:
        hf_occ = qml.qchem.hf_state(active_electrons, n_qubits)
        psi0 = _basis_state_vector(hf_occ)
    else:
        psi0 = np.asarray(initial_state, dtype=complex)

        if psi0.ndim != 1:
            raise ValueError("initial_state must be a 1D statevector")

        expected_dim = 2**n_qubits
        if psi0.size != expected_dim:
            raise ValueError(
                f"initial_state must have length {expected_dim}, got {psi0.size}"
            )

    norm0 = np.linalg.norm(psi0)
    if norm0 == 0:
        raise ValueError("initial_state has zero norm")

    psi0 = psi0 / norm0

    # ------------------------------------------------------------------
    # Choose eta and alpha for B = (eta I - H) / alpha.
    # ------------------------------------------------------------------
    if eta is not None and alpha is not None:
        eta_value = float(eta)
        alpha_value = float(alpha)

    elif energy_bounds is not None:
        e_min, e_max = energy_bounds
        e_min = float(e_min)
        e_max = float(e_max)

        if e_max <= e_min:
            raise ValueError("energy_bounds must satisfy E_max > E_min")

        eta_value = 0.5 * (e_max + e_min)
        alpha_value = 0.5 * (e_max - e_min)

    else:
        e_min, e_max = _estimate_energy_bounds(H_mat)
        eta_value = 0.5 * (e_max + e_min)
        alpha_value = 0.5 * (e_max - e_min)

    if alpha_value <= 0:
        raise ValueError("alpha must be positive")

    # ------------------------------------------------------------------
    # Apply B_epsilon = sin(epsilon B) / epsilon using unitary evolution.
    #
    # B = (eta I - H) / alpha, so:
    #
    #   exp(+i epsilon B)
    #       = exp(+i epsilon eta / alpha) exp(-i epsilon H / alpha)
    #
    #   exp(-i epsilon B)
    #       = exp(-i epsilon eta / alpha) exp(+i epsilon H / alpha)
    #
    # Then:
    #
    #   B_epsilon |v>
    #       = [exp(+i epsilon B) - exp(-i epsilon B)] |v> / (2i epsilon).
    # ------------------------------------------------------------------
    def _apply_B_epsilon(state):
        phase_plus = np.exp(1j * epsilon * eta_value / alpha_value)
        phase_minus = np.exp(-1j * epsilon * eta_value / alpha_value)

        forward = phase_plus * expm_multiply(
            (-1j * epsilon / alpha_value) * H_mat,
            state,
        )

        backward = phase_minus * expm_multiply(
            (1j * epsilon / alpha_value) * H_mat,
            state,
        )

        return (forward - backward) / (2j * epsilon)

    # ------------------------------------------------------------------
    # Build Chebyshev-QKUD basis.
    #
    # Do not normalize each vector inside the recurrence; normalization would
    # change the exact Chebyshev polynomial basis.
    # ------------------------------------------------------------------
    basis_states = [psi0]

    if n_steps >= 1:
        chi_prev = psi0
        chi_curr = _apply_B_epsilon(psi0)
        basis_states.append(chi_curr)

        for _ in range(1, n_steps):
            chi_next = 2.0 * _apply_B_epsilon(chi_curr) - chi_prev
            basis_states.append(chi_next)
            chi_prev, chi_curr = chi_curr, chi_next

    basis_states = np.stack(basis_states, axis=0)

    energies, S, H_K = _projected_problem(basis_states)

    outputs = [energies, basis_states]

    if return_matrices:
        outputs.extend([S, H_K])

    if return_min_energy_history:
        min_energy_history = []

        for k in range(1, basis_states.shape[0] + 1):
            e_k, _, _ = _projected_problem(basis_states[:k])
            min_energy_history.append(float(e_k[0]))

        outputs.append(np.asarray(min_energy_history, dtype=float))

    return tuple(outputs)
