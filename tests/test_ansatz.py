import jax.numpy as jnp
import numpy as np
import pennylane as qp

from fqc.ansatz import apply_ansatz, initial_parameters, parameter_shape
from fqc.hamiltonian import build_h2_problem


def test_h2_uccsd_has_two_singles_and_one_double():
    assert parameter_shape("uccsd", electrons=2, qubits=4) == (3,)
    assert initial_parameters("uccsd", 2, 4).shape == (3,)


def test_two_layer_hea_has_sixteen_parameters():
    assert parameter_shape("hea", electrons=2, qubits=4, layers=2) == (2, 4, 2)
    assert initial_parameters("hea", 2, 4, layers=2).size == 16


def test_hea_gaussian_initialization_is_seeded_and_centered():
    first = initial_parameters("hea", 2, 4, layers=2, seed=7, scale=0.05)
    second = initial_parameters("hea", 2, 4, layers=2, seed=7, scale=0.05)
    different = initial_parameters("hea", 2, 4, layers=2, seed=8, scale=0.05)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert abs(float(jnp.mean(first))) < 0.05
    assert float(jnp.std(first)) > 0.0


def test_both_ansatzes_produce_normalized_states():
    problem = build_h2_problem(0.74)
    device = qp.device("default.qubit", wires=problem.qubits)

    for name in ("uccsd", "hea"):
        parameters = initial_parameters(name, problem.electrons, problem.qubits)

        @qp.qnode(device, interface="jax")
        def state_circuit(values):
            apply_ansatz(name, values, problem)
            return qp.state()

        state = state_circuit(parameters)
        assert np.isclose(float(jnp.vdot(state, state).real), 1.0)
