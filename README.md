# First Quantum Computing

A minimal project based on two quantum-computing tutorials:

- [Build and run your first quantum program](https://qiskit.github.io/documentation/main/learning/courses/use-a-qc-today/build-and-run-your-first-quantum-program)
  from IBM Qiskit.
- [A brief overview of VQE](https://pennylane.ai/qml/demos/tutorial_vqe)
  from PennyLane.

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

Run the H₂ VQE example:

```bash
uv run python src/fqc/vqe_demo.py
```
