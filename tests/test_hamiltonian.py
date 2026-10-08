import numpy as np
import pytest

from fqc.hamiltonian import build_h2_problem


def test_rejects_non_positive_bond_length():
    with pytest.raises(ValueError, match="positive"):
        build_h2_problem(0.0)


def test_builds_physical_h2_problem_at_equilibrium():
    problem = build_h2_problem(0.74)

    assert problem.electrons == 2
    assert problem.qubits == 4
    assert problem.hf_state.tolist() == [1, 1, 0, 0]
    assert np.isfinite(problem.hf_energy)
    assert np.isfinite(problem.fci_energy)
    assert -1.12 < problem.hf_energy < -1.11
    assert -1.15 < problem.fci_energy < -1.13
    assert problem.fci_energy <= problem.hf_energy
    assert set(problem.hamiltonian.wires.tolist()) == {0, 1, 2, 3}
