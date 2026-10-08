from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from fqc.hamiltonian import build_h2_problem
from fqc.run_vqe import VQEResult, run_best_hea, run_vqe

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PES_BACKEND = "lightning.qubit"
CSV_COLUMNS = (
    "bond_length",
    "method",
    "backend",
    "energy_hartree",
    "fci_energy_hartree",
    "error_mha",
    "iterations",
    "runtime_seconds",
    "parameter_count",
    "initialization_seed",
)


def run_pes(
    bond_lengths: Sequence[float],
    *,
    layers: int,
    seed: int,
    hea_seeds: int,
    initialization_scale: float,
    learning_rate: float,
    max_iterations: int,
) -> tuple[
    list[dict[str, object]],
    dict[tuple[float, str, str], VQEResult],
]:
    rows: list[dict[str, object]] = []
    results: dict[tuple[float, str, str], VQEResult] = {}

    for bond_length in bond_lengths:
        problem = build_h2_problem(float(bond_length))
        rows.extend(
            [
                {
                    "bond_length": problem.bond_length,
                    "method": "hf",
                    "backend": "classical",
                    "energy_hartree": problem.hf_energy,
                    "fci_energy_hartree": problem.fci_energy,
                    "error_mha": abs(
                        problem.hf_energy - problem.fci_energy
                    )
                    * 1000,
                    "iterations": 0,
                    "runtime_seconds": 0.0,
                    "parameter_count": 0,
                    "initialization_seed": -1,
                },
                {
                    "bond_length": problem.bond_length,
                    "method": "fci",
                    "backend": "classical",
                    "energy_hartree": problem.fci_energy,
                    "fci_energy_hartree": problem.fci_energy,
                    "error_mha": 0.0,
                    "iterations": 0,
                    "runtime_seconds": 0.0,
                    "parameter_count": 0,
                    "initialization_seed": -1,
                },
            ]
        )

        for ansatz in ("uccsd", "hea"):
            if ansatz == "hea":
                result = run_best_hea(
                    problem,
                    PES_BACKEND,
                    seeds=range(seed, seed + hea_seeds),
                    layers=layers,
                    initialization_scale=initialization_scale,
                    learning_rate=learning_rate,
                    max_iterations=max_iterations,
                )
            else:
                result = run_vqe(
                    problem,
                    ansatz,
                    PES_BACKEND,
                    layers=layers,
                    seed=seed,
                    initialization_scale=initialization_scale,
                    learning_rate=learning_rate,
                    max_iterations=max_iterations,
                )
            results[(problem.bond_length, ansatz, PES_BACKEND)] = result
            rows.append(
                {
                    "bond_length": result.bond_length,
                    "method": result.ansatz,
                    "backend": result.backend,
                    "energy_hartree": result.energy,
                    "fci_energy_hartree": result.fci_energy,
                    "error_mha": result.error_mha,
                    "iterations": result.iterations,
                    "runtime_seconds": result.runtime_seconds,
                    "parameter_count": result.parameter_count,
                    "initialization_seed": result.initialization_seed,
                }
            )
        print(f"Completed R = {problem.bond_length:.3f} Angstrom")

    return rows, results


def write_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def plot_pes(rows: Sequence[dict[str, object]], path: Path) -> None:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["backend"]))
        grouped.setdefault(key, []).append(row)

    figure, (energy_axis, error_axis) = plt.subplots(
        1,
        2,
        figsize=(8, 3.5),
    )
    for (method, backend), group in grouped.items():
        ordered = sorted(group, key=lambda row: float(row["bond_length"]))
        label = method.upper() if backend == "classical" else f"{method.upper()} ({backend})"
        energy_axis.plot(
            [float(row["bond_length"]) for row in ordered],
            [float(row["energy_hartree"]) for row in ordered],
            marker="o",
            label=label,
        )

    for method in ("uccsd", "hea"):
        ordered = sorted(
            (
                row
                for row in rows
                if row["method"] == method and row["backend"] == PES_BACKEND
            ),
            key=lambda row: float(row["bond_length"]),
        )
        error_axis.plot(
            [float(row["bond_length"]) for row in ordered],
            [float(row["error_mha"]) for row in ordered],
            marker="o",
            label=method.upper(),
        )

    energy_axis.set_xlabel("Bond length (Angstrom)")
    energy_axis.set_ylabel("Energy (Hartree)")
    energy_axis.set_title("Potential Energy Surface")
    energy_axis.legend(fontsize=7)
    energy_axis.grid(alpha=0.25)
    error_axis.set_xlabel("Bond length (Angstrom)")
    error_axis.set_ylabel("Absolute error (mHa)")
    error_axis.set_title("Error Relative to FCI")
    error_axis.legend(fontsize=7)
    error_axis.grid(alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def plot_convergence(
    results: dict[tuple[float, str, str], VQEResult],
    path: Path,
) -> None:
    equilibrium_bond_length = min(
        {key[0] for key in results},
        key=lambda value: abs(value - 0.74),
    )
    large_bond_length = max(key[0] for key in results)

    figure, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for axis, bond_length, title in (
        (axes[0], equilibrium_bond_length, "Near Equilibrium"),
        (axes[1], large_bond_length, "Large Bond Length"),
    ):
        selected = [
            result
            for (result_bond_length, _, _), result in results.items()
            if result_bond_length == bond_length
        ]
        for result in selected:
            axis.plot(
                range(len(result.energy_history)),
                result.energy_history,
                label=result.ansatz.upper(),
            )
        axis.axhline(
            selected[0].fci_energy,
            color="black",
            linestyle="--",
            label="FCI",
        )
        axis.set_xlabel("Optimization step")
        axis.set_ylabel("Energy (Hartree)")
        axis.set_title(
            f"{title}\nR = {bond_length:.3f} Angstrom"
        )
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete H2 VQE potential-energy comparison."
    )
    parser.add_argument("--start", type=float, default=0.5)
    parser.add_argument("--stop", type=float, default=1.5)
    parser.add_argument("--points", type=int, default=10)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hea-seeds", type=int, default=5)
    parser.add_argument("--initialization-scale", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-iterations", type=int, default=150)
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "pes_h2.csv",
    )
    parser.add_argument(
        "--pes-figure",
        type=Path,
        default=PROJECT_ROOT / "figs" / "pes_curve_h2.png",
    )
    parser.add_argument(
        "--convergence-figure",
        type=Path,
        default=PROJECT_ROOT / "figs" / "convergence_h2.png",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.points < 1:
        parser.error("--points must be at least 1")
    if args.start <= 0:
        parser.error("--start must be positive")
    if args.stop < args.start:
        parser.error("--stop must be greater than or equal to --start")
    if args.hea_seeds < 1:
        parser.error("--hea-seeds must be at least 1")

    bond_lengths = np.linspace(args.start, args.stop, args.points)
    rows, results = run_pes(
        bond_lengths,
        layers=args.layers,
        seed=args.seed,
        hea_seeds=args.hea_seeds,
        initialization_scale=args.initialization_scale,
        learning_rate=args.learning_rate,
        max_iterations=args.max_iterations,
    )
    write_csv(rows, args.csv_path)
    plot_pes(rows, args.pes_figure)
    plot_convergence(results, args.convergence_figure)
    print(f"Wrote {len(rows)} rows to {args.csv_path}")
    print(f"Saved PES figure to {args.pes_figure}")
    print(f"Saved convergence figure to {args.convergence_figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
