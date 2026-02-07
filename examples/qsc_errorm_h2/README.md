# qsc_errorm: H2 test (example)

This folder contains a small `pytest`-style check that:

- runs `QCANT.adapt_vqe` for H₂, then
- verifies `QCANT.qsc_errorm` returns the same excited-state eigenvalues as `QCANT.qscEOM`,
  plus one extra lowest eigenvalue corresponding to the ground state.

Run:

```bash
conda activate qcant
pytest -q examples/qsc_errorm_h2/test_qsc_errorm_h2.py
```

