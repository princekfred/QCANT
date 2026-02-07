"""qsc_errorm: qscEOM with the reference configuration included.

This routine mirrors :func:`QCANT.qscEOM` but expands the configuration pool by
one extra entry corresponding to the identity (no-excitation) configuration.
In practice, this adds the Hartree–Fock reference determinant to the subspace,
making the diagonalization "qscEOM + 1" dimensional and providing access to the
ground state.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from ..qsceom.excitations import inite


def qsc_errorm(
    symbols: Sequence[str],
    geometry,
    active_electrons: int,
    active_orbitals: int,
    charge: int,
    params=None,
    ash_excitation=None,
    *,
    ansatz: Optional[Tuple[Any, Any, Any]] = None,
    basis: str = "sto-3g",
    method: str = "pyscf",
    spin: int = 0,
    shots: int = 0,
    device_name: Optional[str] = None,
    bitflip_probs=None,
    max_states: Optional[int] = None,
    state_seed: Optional[int] = None,
    symmetric: bool = True,
    print_matrix: bool = False,
):
    """Compute qscEOM-like eigenvalues including the ground-state configuration.

    Parameters
    ----------
    symbols
        Atomic symbols.
    geometry
        Nuclear coordinates (as an array-like object).
    active_electrons
        Number of active electrons.
    active_orbitals
        Number of active orbitals.
    charge
        Total molecular charge.
    params
        Ansatz parameters.
    ash_excitation
        Excitation list describing the ansatz.
    ansatz
        Optional 3-tuple ``(params, ash_excitation, energies)`` as returned by
        :func:`QCANT.adapt_vqe`. If provided, ``params`` and ``ash_excitation``
        are taken from this tuple.
    basis
        Basis set name understood by PennyLane/PySCF (e.g. ``"sto-3g"``).
    method
        Backend used by PennyLane quantum chemistry tooling (default: ``"pyscf"``).
    spin
        Spin parameter used by PySCF/PennyLane as ``2S`` (e.g. 0 for singlet).
    shots
        If 0, run in analytic mode; otherwise use shot-based estimation.
    device_name
        Optional PennyLane device name (e.g. ``"lightning.qubit"``).
    bitflip_probs
        Optional bit-flip error probabilities to apply **at the end of the circuit**
        (right before measurement) on *all* wires. Provide either:

        - a single float in ``[0, 1]`` (applied to every wire), or
        - a sequence of floats with length ``n_qubits`` (per-wire probabilities).

        When enabled, the routine will use a mixed-state simulator (``default.mixed``),
        regardless of ``device_name``.
    max_states
        If provided, limit the number of excited occupation configurations used
        to build the effective matrix (the reference configuration is always
        included, so the final dimension is ``max_states + 1``).
    state_seed
        Seed for selecting a random subset when ``max_states`` is used.
    symmetric
        If True, compute only the upper-triangular off-diagonal elements and
        mirror them to reduce circuit evaluations.
    print_matrix
        If True, print the effective matrix ``M`` (and configuration ordering)
        before diagonalization.

    Returns
    -------
    list
        Sorted eigenvalues for the constructed effective matrix.
    """

    if ansatz is not None:
        try:
            params_from_adapt, ash_excitation_from_adapt, _energies = ansatz
        except Exception as exc:
            raise ValueError(
                "ansatz must be a 3-tuple like (params, ash_excitation, energies) "
                "as returned by QCANT.adapt_vqe"
            ) from exc

        params = params_from_adapt
        ash_excitation = ash_excitation_from_adapt

    if params is None or ash_excitation is None:
        raise TypeError(
            "qsc_errorm requires either (params, ash_excitation) or ansatz=(params, ash_excitation, energies)."
        )
    if max_states is not None and max_states <= 0:
        raise ValueError("max_states must be > 0")

    try:
        if len(params) != len(ash_excitation):
            raise ValueError
    except Exception as exc:
        raise ValueError("params and ash_excitation must have the same length") from exc

    try:
        import numpy as np
        import pennylane as qml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "qsc_errorm requires dependencies. Install at least: "
            "`pip install numpy pennylane`."
        ) from exc

    mult = int(spin) + 1
    mol = qml.qchem.Molecule(
        symbols,
        geometry,
        charge=int(charge),
        mult=mult,
        basis_name=basis,
        unit="angstrom",
    )
    H, qubits = qml.qchem.molecular_hamiltonian(
        mol,
        method=method,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

    singles, doubles = qml.qchem.excitations(active_electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    wires = range(qubits)

    def _normalize_bitflip_probs(raw, n_qubits: int):
        if raw is None:
            return None
        if isinstance(raw, (int, float, np.floating)):
            probs = [float(raw)] * int(n_qubits)
        else:
            try:
                probs = [float(p) for p in raw]
            except TypeError as exc:
                raise TypeError("bitflip_probs must be a float or a sequence of floats") from exc
            if len(probs) != int(n_qubits):
                raise ValueError(f"bitflip_probs must have length {n_qubits}, got {len(probs)}")

        for p in probs:
            if p < 0.0 or p > 1.0:
                raise ValueError("bitflip_probs entries must be between 0 and 1 (inclusive)")
        if all(p == 0.0 for p in probs):
            return None
        return tuple(probs)

    bitflip_probs_norm = _normalize_bitflip_probs(bitflip_probs, qubits)

    null_state = np.zeros(qubits, int)
    excited_configs = inite(active_electrons, qubits)
    if max_states is not None and max_states < len(excited_configs):
        rng = np.random.default_rng(state_seed)
        indices = rng.choice(len(excited_configs), size=max_states, replace=False)
        excited_configs = [excited_configs[idx] for idx in sorted(indices)]

    # Include the reference (identity) configuration as the first entry.
    # This is the Hartree–Fock determinant occupation pattern.
    hf_occ = qml.qchem.hf_state(active_electrons, qubits)
    reference_config = [int(i) for i, occ in enumerate(hf_occ) if int(occ) == 1]
    configs = [reference_config] + excited_configs

    values = []

    # Preserve original behavior (single iteration) from the prior script.
    for _ in range(1):
        def _make_device(name: Optional[str], wires: int):
            kwargs = {}
            if shots > 0:
                kwargs["shots"] = shots
            if bitflip_probs_norm is not None:
                # Noise channels (e.g. BitFlip) require a mixed-state simulator.
                # Use default.mixed regardless of the requested device.
                return qml.device("default.mixed", wires=wires, **kwargs)
            if name is not None:
                return qml.device(name, wires=wires, **kwargs)
            try:
                return qml.device("lightning.qubit", wires=wires, **kwargs)
            except Exception:
                return qml.device("default.qubit", wires=wires, **kwargs)

        dev = _make_device(device_name, qubits)

        @qml.qnode(dev)
        def circuit_d(params, occ, wires, s_wires, d_wires, hf_state, ash_excitation):
            qml.BasisState(hf_state, wires=range(qubits))
            for w in occ:
                qml.X(wires=w)
            for i, excitations in enumerate(ash_excitation):
                if len(excitations) == 4:
                    qml.FermionicDoubleExcitation(
                        weight=params[i],
                        wires1=list(range(excitations[0], excitations[1] + 1)),
                        wires2=list(range(excitations[2], excitations[3] + 1)),
                    )
                elif len(excitations) == 2:
                    qml.FermionicSingleExcitation(
                        weight=params[i],
                        wires=list(range(excitations[0], excitations[1] + 1)),
                    )
            if bitflip_probs_norm is not None:
                for w, p in zip(wires, bitflip_probs_norm):
                    if p != 0.0:
                        qml.BitFlip(p, wires=w)
            return qml.expval(H)

        @qml.qnode(dev)
        def circuit_od(params, occ1, occ2, wires, s_wires, d_wires, hf_state, ash_excitation):
            qml.BasisState(hf_state, wires=range(qubits))
            for w in occ1:
                qml.X(wires=w)
            first = -1
            for v in occ2:
                if v not in occ1:
                    if first == -1:
                        first = v
                        qml.Hadamard(wires=v)
                    else:
                        qml.CNOT(wires=[first, v])
            for v in occ1:
                if v not in occ2:
                    if first == -1:
                        first = v
                        qml.Hadamard(wires=v)
                    else:
                        qml.CNOT(wires=[first, v])
            for i, excitations in enumerate(ash_excitation):
                if len(excitations) == 4:
                    qml.FermionicDoubleExcitation(
                        weight=params[i],
                        wires1=list(range(excitations[0], excitations[1] + 1)),
                        wires2=list(range(excitations[2], excitations[3] + 1)),
                    )
                elif len(excitations) == 2:
                    qml.FermionicSingleExcitation(
                        weight=params[i],
                        wires=list(range(excitations[0], excitations[1] + 1)),
                    )
            if bitflip_probs_norm is not None:
                for w, p in zip(wires, bitflip_probs_norm):
                    if p != 0.0:
                        qml.BitFlip(p, wires=w)
            return qml.expval(H)

        M = np.zeros((len(configs), len(configs)))
        for i in range(len(configs)):
            for j in range(len(configs)):
                if i == j:
                    M[i, i] = circuit_d(
                        params, configs[i], wires, s_wires, d_wires, null_state, ash_excitation
                    )

        if symmetric:
            for i in range(len(configs)):
                for j in range(i + 1, len(configs)):
                    Mtmp = circuit_od(
                        params,
                        configs[i],
                        configs[j],
                        wires,
                        s_wires,
                        d_wires,
                        null_state,
                        ash_excitation,
                    )
                    value = Mtmp - M[i, i] / 2.0 - M[j, j] / 2.0
                    M[i, j] = value
                    M[j, i] = value
        else:
            for i in range(len(configs)):
                for j in range(len(configs)):
                    if i != j:
                        Mtmp = circuit_od(
                            params,
                            configs[i],
                            configs[j],
                            wires,
                            s_wires,
                            d_wires,
                            null_state,
                            ash_excitation,
                        )
                        M[i, j] = Mtmp - M[i, i] / 2.0 - M[j, j] / 2.0

        if print_matrix:
            print("qsc_errorm configurations (occupied indices):", flush=True)
            for idx, occ in enumerate(configs):
                print(f"  {idx}: {occ}", flush=True)
            print("qsc_errorm effective matrix M:", flush=True)
            print(np.array2string(M, precision=10, suppress_small=True), flush=True)

        eig, _ = np.linalg.eig(M)
        values.append(np.sort(eig))

    return values
