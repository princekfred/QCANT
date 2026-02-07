from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _setup_local_cache() -> Path:
    here = Path(__file__).resolve().parent
    cache_dir = here / ".cache"
    mpl_dir = cache_dir / "matplotlib"
    xdg_dir = cache_dir / "xdg"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    xdg_dir.mkdir(parents=True, exist_ok=True)

    # Keep caches inside this folder (useful on systems with restricted $HOME).
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_dir))
    return cache_dir


def _jsonable_eigenvalues(evals) -> list[float] | list[dict[str, float]]:
    import numpy as np

    arr = np.asarray(evals)
    if np.iscomplexobj(arr) and not np.allclose(arr.imag, 0.0):
        return [{"re": float(r), "im": float(i)} for r, i in zip(arr.real, arr.imag)]
    return [float(x) for x in arr.real]


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an H2 qscEOM demo using QCANT.")
    parser.add_argument("--bond-length", type=float, default=0.735, help="H–H distance (Angstrom).")
    parser.add_argument("--basis", type=str, default="sto-3g", help="Basis set (PennyLane/PySCF name).")
    parser.add_argument("--active-electrons", type=int, default=2)
    parser.add_argument("--active-orbitals", type=int, default=2)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=0, help="Spin as 2S (PySCF convention).")
    parser.add_argument("--device-name", type=str, default="default.qubit")

    parser.add_argument("--adapt-it", type=int, default=1, help="Number of ADAPT-VQE iterations.")
    parser.add_argument("--optimizer-maxiter", type=int, default=25, help="Max SciPy optimizer iterations.")

    parser.add_argument("--shots", type=int, default=0, help="qscEOM shots (0 = analytic).")
    parser.add_argument("--max-states", type=int, default=None, help="Limit qscEOM configurations (optional).")
    parser.add_argument("--state-seed", type=int, default=None, help="Seed for selecting configurations.")
    parser.add_argument(
        "--print-matrix",
        action="store_true",
        help="Print the qscEOM effective matrix M (and configuration ordering).",
    )
    parser.add_argument(
        "--no-symmetric",
        action="store_true",
        help="Disable symmetric fill (computes full off-diagonal matrix).",
    )

    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    _setup_local_cache()

    import numpy as np
    import QCANT

    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, float(args.bond_length)]], dtype=float)

    print("Running ADAPT-VQE...", flush=True)
    params, ash_excitation, energies = QCANT.adapt_vqe(
        symbols=symbols,
        geometry=geometry,
        adapt_it=int(args.adapt_it),
        basis=args.basis,
        charge=int(args.charge),
        spin=int(args.spin),
        active_electrons=int(args.active_electrons),
        active_orbitals=int(args.active_orbitals),
        device_name=args.device_name,
        optimizer_maxiter=int(args.optimizer_maxiter),
    )

    print(f"ADAPT energies: {energies}", flush=True)
    print("Running qscEOM...", flush=True)
    values = QCANT.qscEOM(
        symbols=symbols,
        geometry=geometry,
        active_electrons=int(args.active_electrons),
        active_orbitals=int(args.active_orbitals),
        charge=int(args.charge),
        ansatz=(params, ash_excitation, energies),
        basis=args.basis,
        method="pyscf",
        shots=int(args.shots),
        device_name=args.device_name,
        max_states=args.max_states,
        state_seed=args.state_seed,
        symmetric=not bool(args.no_symmetric),
        print_matrix=bool(args.print_matrix),
    )

    evals = values[0]
    print("qscEOM eigenvalues:")
   
    print(evals, flush=True)

    if args.out is not None:
        payload: dict[str, Any] = {
            "symbols": symbols,
            "geometry_angstrom": geometry.tolist(),
            "basis": args.basis,
            "active_electrons": int(args.active_electrons),
            "active_orbitals": int(args.active_orbitals),
            "charge": int(args.charge),
            "spin_2s": int(args.spin),
            "device_name": args.device_name,
            "adapt": {
                "adapt_it": int(args.adapt_it),
                "optimizer_maxiter": int(args.optimizer_maxiter),
                "params": [float(x) for x in np.asarray(params).ravel()],
                "ash_excitation": [[int(i) for i in ex] for ex in ash_excitation],
                "energies": [float(x) for x in energies],
            },
            "qsceom": {
                "shots": int(args.shots),
                "max_states": args.max_states,
                "state_seed": args.state_seed,
                "symmetric": not bool(args.no_symmetric),
                "eigenvalues": _jsonable_eigenvalues(evals),
            },
        }
        _dump_json(args.out, payload)
        print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
