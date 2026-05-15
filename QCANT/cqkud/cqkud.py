"""Chebyshev-polynomial Quantum Krylov.

This module mirrors :mod:`QCANT.qkud.qkud`, but generates the Krylov basis
directly with Chebyshev polynomials of a scaled Hamiltonian:

    H_tilde = (H - eta * I) / alpha

    |chi_0>     = |psi_0>
    |chi_1>     = H_tilde |psi_0>
    |chi_{n+1}> = 2 H_tilde |chi_n> - |chi_{n-1}>

Thus |chi_n> = T_n(H_tilde) |psi_0>, where T_n is the nth Chebyshev
polynomial of the first kind.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple


def chebqkud(
    symbols: Sequence[str],
    geometry,
    *,
    n_steps: int,
    active_electrons: int,
    active_orbitals: int,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    method: str = "pyscf",
    device_name: Optional[str] = None,
    initial_state: Optional["object"] = None,
    energy_bounds: Optional[Tuple[float, float]] = None,
    eta: Optional[float] = None,
    alpha: Optional[float] = None,
    overlap_tol: float = 1e-10,
    normalize_basis: bool = False,
    basis_threshold: float = 0.0,
    use_sparse: bool = False,
    return_matrices: bool = False,
    return_min_energy_history: bool = False,
):
    """Generate a Chebyshev Krylov basis and diagonalize H in that basis.

    The basis vectors are Chebyshev-polynomial vectors:

        |chi_n> = T_n(H_tilde) |psi_0>

    where ``H_tilde = (H - eta I) / alpha``. If ``eta`` and ``alpha`` are not
    provided, they are inferred from ``energy_bounds``; if no bounds are given,
    the code estimates the lowest and highest eigenvalues of ``H``.

    Parameters match :func:`QCANT.qkud.qkud` where possible. Unlike QKUD, this
    routine does not use an ``epsilon`` unitary-decomposition step to generate
    the basis.
    """

    if len(symbols) == 0:
        raise ValueError("symbols must be non-empty")
    if n_steps < 0:
        raise ValueError("n_steps must be >= 0")
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
        from scipy.sparse import identity, issparse
        from scipy.sparse.linalg import eigsh
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "chebqkud requires dependencies. Install at least: "
            "`pip install numpy pennylane pyscf scipy`."
        ) from exc

    del device_name  # Present for API compatibility with qkud().

    def _basis_state_vector(bits: Sequence[int]) -> "object":
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
            except TypeError as exc:
                if "format" not in str(exc):
                    raise
                mat = operator.sparse_matrix(wire_order=wires)
                return mat.tocsr() if hasattr(mat, "tocsr") else mat
        return qml.matrix(operator, wire_order=wires)

    def _estimate_energy_bounds(matrix):
        dim = matrix.shape[0]
        if issparse(matrix):
            if dim <= 2:
                vals = np.linalg.eigvalsh(matrix.toarray())
                return float(vals[0]), float(vals[-1])
            e_min = eigsh(matrix, k=1, which="SA", return_eigenvectors=False)[0]
            e_max = eigsh(matrix, k=1, which="LA", return_eigenvectors=False)[0]
            return float(e_min), float(e_max)
        vals = np.linalg.eigvalsh(matrix)
        return float(vals[0]), float(vals[-1])

    def _apply_basis_threshold(state):
        if basis_threshold <= 0:
            return state
        state = np.asarray(state, dtype=complex)
        mask = np.abs(state) >= basis_threshold
        if not np.any(mask):
            idx = int(np.argmax(np.abs(state)))
            mask[idx] = True
        state = np.where(mask, state, 0.0)
        norm = np.linalg.norm(state)
        if norm == 0:
            raise ValueError("thresholded basis vector has zero norm")
        return state / norm

    def _scale_hamiltonian(matrix, eta_value, alpha_value):
        if issparse(matrix):
            ident = identity(matrix.shape[0], dtype=complex, format="csr")
            return (matrix - eta_value * ident) / alpha_value
        return (matrix - eta_value * np.eye(matrix.shape[0], dtype=complex)) / alpha_value

    def _projected_problem(current_basis_states):
        S = current_basis_states.conj() @ current_basis_states.T
        H_k = current_basis_states.conj() @ (H_mat @ current_basis_states.T)

        S = 0.5 * (S + S.conj().T)
        H_k = 0.5 * (H_k + H_k.conj().T)

        s_vals, s_vecs = np.linalg.eigh(S)
        keep = s_vals > float(overlap_tol)
        if not keep.any():
            raise ValueError("overlap matrix is numerically singular; basis collapsed")

        X = s_vecs[:, keep] / np.sqrt(s_vals[keep])[None, :]
        H_ortho = X.conj().T @ H_k @ X
        H_ortho = 0.5 * (H_ortho + H_ortho.conj().T)
        energies = np.linalg.eigvalsh(H_ortho).real
        return energies, S, H_k

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
    psi0 = _apply_basis_threshold(psi0 / norm0)

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
    elif eta is None and alpha is None:
        e_min, e_max = _estimate_energy_bounds(H_mat)
        eta_value = 0.5 * (e_max + e_min)
        alpha_value = 0.5 * (e_max - e_min)
    else:
        raise ValueError("eta and alpha must be supplied together")

    if alpha_value <= 0:
        raise ValueError("alpha must be positive")

    H_tilde = _scale_hamiltonian(H_mat, eta_value, alpha_value)

    def _prepare_vector(state):
        if normalize_basis:
            norm = np.linalg.norm(state)
            if norm == 0:
                raise ValueError("Chebyshev Krylov vector has zero norm")
            state = state / norm
        return _apply_basis_threshold(state)

    basis_states = [psi0]

    if n_steps >= 1:
        chi_prev = psi0
        chi_curr = _prepare_vector(H_tilde @ psi0)
        basis_states.append(chi_curr)

        for _ in range(1, n_steps):
            chi_next = _prepare_vector(2.0 * (H_tilde @ chi_curr) - chi_prev)
            basis_states.append(chi_next)
            chi_prev, chi_curr = chi_curr, chi_next

    basis_states = np.stack(basis_states, axis=0)
    energies, S, H_k = _projected_problem(basis_states)

    outputs = [energies, basis_states]
    if return_matrices:
        outputs.extend([S, H_k])
    if return_min_energy_history:
        min_energy_history = []
        for k in range(1, basis_states.shape[0] + 1):
            e_k, _, _ = _projected_problem(basis_states[:k])
            min_energy_history.append(float(e_k[0]))
        outputs.append(np.asarray(min_energy_history, dtype=float))

    return tuple(outputs)


chebyshev_qkud = chebqkud


__all__ = ["chebqkud", "chebyshev_qkud"]
