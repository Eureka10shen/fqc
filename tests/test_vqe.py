import csv

import numpy as np
from PIL import Image

from fqc.ansatz import initial_parameters
from fqc.hamiltonian import build_h2_problem
from fqc.run_vqe import make_energy_qnode, run_best_hea, run_vqe
from fqc.run_vqe_pes import CSV_COLUMNS, main as pes_main


def test_backends_agree_at_fixed_uccsd_parameters():
    problem = build_h2_problem(0.74)
    parameters = initial_parameters("uccsd", 2, 4)
    default_energy = make_energy_qnode(
        problem, "uccsd", "default.qubit"
    )(parameters)
    lightning_energy = make_energy_qnode(
        problem, "uccsd", "lightning.qubit"
    )(parameters)

    assert np.isclose(default_energy, lightning_energy, atol=1e-9)


def test_uccsd_reaches_fci_within_one_millihartree():
    problem = build_h2_problem(0.74)
    result = run_vqe(
        problem,
        "uccsd",
        "lightning.qubit",
        max_iterations=150,
    )

    assert result.error_mha < 1.0
    assert result.parameter_count == 3
    assert len(result.energy_history) >= 2


def test_hea_smoke_run_is_finite():
    problem = build_h2_problem(0.74)
    result = run_vqe(
        problem,
        "hea",
        "default.qubit",
        max_iterations=3,
    )

    assert np.isfinite(result.energy)
    assert result.parameter_count == 16
    assert len(result.energy_history) == 4


def test_best_hea_keeps_the_lowest_energy_seed():
    problem = build_h2_problem(0.5)
    candidates = [
        run_vqe(
            problem,
            "hea",
            "lightning.qubit",
            seed=seed,
            max_iterations=2,
        )
        for seed in (0, 1)
    ]
    best = run_best_hea(
        problem,
        "lightning.qubit",
        seeds=(0, 1),
        max_iterations=2,
    )

    assert np.isclose(best.energy, min(result.energy for result in candidates))
    assert best.initialization_seed in {0, 1}


def test_short_pes_scan_writes_csv_and_figures(tmp_path):
    csv_path = tmp_path / "pes.csv"
    pes_figure = tmp_path / "pes.png"
    convergence_figure = tmp_path / "convergence.png"

    exit_code = pes_main(
        [
            "--start",
            "0.70",
            "--stop",
            "0.74",
            "--points",
            "2",
            "--max-iterations",
            "2",
            "--hea-seeds",
            "2",
            "--csv-path",
            str(csv_path),
            "--pes-figure",
            str(pes_figure),
            "--convergence-figure",
            str(convergence_figure),
        ]
    )

    assert exit_code == 0
    assert csv_path.stat().st_size > 0
    assert pes_figure.stat().st_size > 0
    assert convergence_figure.stat().st_size > 0
    assert Image.open(convergence_figure).size == (2400, 1050)
    for path in (pes_figure, convergence_figure):
        dpi = Image.open(path).info["dpi"]
        assert np.isclose(dpi[0], 300, atol=0.1)
        assert np.isclose(dpi[1], 300, atol=0.1)

    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(rows[0]) == CSV_COLUMNS
    assert len(rows) == 8
    assert {
        row["backend"]
        for row in rows
        if row["method"] in {"uccsd", "hea"}
    } == {"lightning.qubit"}
