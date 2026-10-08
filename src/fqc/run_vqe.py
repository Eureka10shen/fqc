from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import jax
import numpy as np
import optax
import pennylane as qp

from fqc.ansatz import AnsatzName, apply_ansatz, initial_parameters
from fqc.hamiltonian import H2Problem, build_h2_problem

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

SUPPORTED_BACKENDS = ("default.qubit", "lightning.qubit")


@dataclass(frozen=True)
class VQEResult:
    bond_length: float
    ansatz: AnsatzName
    backend: str
    energy: float
    fci_energy: float
    error_mha: float
    iterations: int
    runtime_seconds: float
    parameter_count: int
    initialization_seed: int
    parameters: np.ndarray
    energy_history: tuple[float, ...]


def make_energy_qnode(
    problem: H2Problem,
    ansatz: AnsatzName,
    backend: str,
    *,
    layers: int = 2,
):
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported backend: {backend}")
    device = qp.device(backend, wires=problem.qubits, shots=None)

    @qp.qnode(device, interface="jax")
    def energy_qnode(parameters):
        apply_ansatz(ansatz, parameters, problem, layers)
        return qp.expval(problem.hamiltonian)

    return energy_qnode


def run_vqe(
    problem: H2Problem,
    ansatz: AnsatzName,
    backend: str,
    *,
    layers: int = 2,
    seed: int = 0,
    initialization_scale: float = 0.05,
    learning_rate: float = 0.1,
    max_iterations: int = 150,
    convergence_tolerance: float = 1e-7,
) -> VQEResult:
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    qnode = make_energy_qnode(problem, ansatz, backend, layers=layers)
    parameters = initial_parameters(
        ansatz,
        problem.electrons,
        problem.qubits,
        layers,
        seed=seed,
        scale=initialization_scale,
    )
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(parameters)
    history = [float(qnode(parameters))]
    started = perf_counter()

    for _ in range(max_iterations):
        energy, gradient = jax.value_and_grad(qnode)(parameters)
        updates, optimizer_state = optimizer.update(
            gradient,
            optimizer_state,
            parameters,
        )
        parameters = optax.apply_updates(parameters, updates)
        next_energy = float(qnode(parameters))
        history.append(next_energy)
        if abs(next_energy - float(energy)) <= convergence_tolerance:
            break

    runtime_seconds = perf_counter() - started
    final_energy = history[-1]
    return VQEResult(
        bond_length=problem.bond_length,
        ansatz=ansatz,
        backend=backend,
        energy=final_energy,
        fci_energy=problem.fci_energy,
        error_mha=abs(final_energy - problem.fci_energy) * 1000,
        iterations=len(history) - 1,
        runtime_seconds=runtime_seconds,
        parameter_count=int(parameters.size),
        initialization_seed=seed,
        parameters=np.asarray(parameters),
        energy_history=tuple(history),
    )


def run_best_hea(
    problem: H2Problem,
    backend: str,
    *,
    seeds: Sequence[int] = range(5),
    layers: int = 2,
    initialization_scale: float = 0.05,
    learning_rate: float = 0.1,
    max_iterations: int = 150,
    convergence_tolerance: float = 1e-7,
) -> VQEResult:
    seed_values = tuple(seeds)
    if not seed_values:
        raise ValueError("seeds must not be empty")
    results = [
        run_vqe(
            problem,
            "hea",
            backend,
            layers=layers,
            seed=seed,
            initialization_scale=initialization_scale,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            convergence_tolerance=convergence_tolerance,
        )
        for seed in seed_values
    ]
    return min(results, key=lambda result: result.energy)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one H2 VQE calculation.")
    parser.add_argument("--bond-length", type=float, default=0.74)
    parser.add_argument(
        "--ansatz",
        choices=("uccsd", "hea"),
        default="uccsd",
    )
    parser.add_argument(
        "--backend",
        choices=SUPPORTED_BACKENDS,
        default="lightning.qubit",
    )
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initialization-scale", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-iterations", type=int, default=150)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    problem = build_h2_problem(args.bond_length)
    result = run_vqe(
        problem,
        args.ansatz,
        args.backend,
        layers=args.layers,
        seed=args.seed,
        initialization_scale=args.initialization_scale,
        learning_rate=args.learning_rate,
        max_iterations=args.max_iterations,
    )

    print(f"Bond length: {result.bond_length:.3f} Angstrom")
    print(f"Ansatz: {result.ansatz}")
    print(f"Backend: {result.backend}")
    print(f"HF energy: {problem.hf_energy:.10f} Ha")
    print(f"FCI energy: {result.fci_energy:.10f} Ha")
    print(f"VQE energy: {result.energy:.10f} Ha")
    print(f"FCI error: {result.error_mha:.6f} mHa")
    print(f"Iterations: {result.iterations}")
    print(f"Runtime: {result.runtime_seconds:.3f} s")
    print(f"Parameters: {result.parameter_count}")
    print(f"Initialization seed: {result.initialization_seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
