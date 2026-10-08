# First Quantum Computing

A minimal project based on two quantum-computing tutorials:

- [Build and run your first quantum program](https://qiskit.github.io/documentation/main/learning/courses/use-a-qc-today/build-and-run-your-first-quantum-program)
  from IBM Qiskit.
- [A brief overview of VQE](https://pennylane.ai/qml/demos/tutorial_vqe)
  from PennyLane.

I have also added a $\text{H}_2$ PES example with VQE. 

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
```

## Run

Run the Bell-state example:

```bash
uv run python src/fqc/_hello_world_circuit.py
```

Run the $\text{H}_2$ VQE example:

```bash
uv run python src/fqc/vqe_demo.py
```

Run one UCCSD calculation for $\text{H}_2$ VQE:

```bash
uv run python -m fqc.run_vqe \
  --bond-length 0.74 \
  --ansatz uccsd \
  --backend lightning.qubit
```

Run the complete ten-point HF, FCI, UCCSD, and HEA comparison for VQE PES on
`lightning.qubit`:

```bash
uv run python -m fqc.run_vqe_pes
```
