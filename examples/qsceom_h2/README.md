# qscEOM: H2 example

This folder contains a small, runnable example for **H₂** that:

1. runs `QCANT.adapt_vqe` to generate an ansatz, then
2. feeds that ansatz into `QCANT.qscEOM`.

## Run

```bash
conda activate qcant
python examples/qsceom_h2/run_qsceom_h2.py
```

### Common options

```bash
# Use a different bond length (Angstrom)
python examples/qsceom_h2/run_qsceom_h2.py --bond-length 0.74

# Run more ADAPT iterations / optimizer steps
python examples/qsceom_h2/run_qsceom_h2.py --adapt-it 2 --optimizer-maxiter 200

# Reduce the qscEOM configuration space (useful for bigger systems)
python examples/qsceom_h2/run_qsceom_h2.py --max-states 3 --state-seed 0

# Write results to a JSON file
python examples/qsceom_h2/run_qsceom_h2.py --out results.json
```

