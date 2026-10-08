from typing import Literal, cast

import jax
import jax.numpy as jnp
import pennylane as qp

from fqc.hamiltonian import H2Problem

jax.config.update("jax_enable_x64", True)

AnsatzName = Literal["uccsd", "hea"]


def _validate(name: str, layers: int) -> AnsatzName:
    if name not in {"uccsd", "hea"}:
        raise ValueError(f"unsupported ansatz: {name}")
    if layers < 1:
        raise ValueError("layers must be at least 1")
    return cast(AnsatzName, name)


def parameter_shape(
    name: AnsatzName,
    electrons: int,
    qubits: int,
    layers: int = 2,
) -> tuple[int, ...]:
    name = _validate(name, layers)
    if name == "uccsd":
        singles, doubles = qp.qchem.excitations(electrons, qubits)
        return (len(singles) + len(doubles),)
    return (layers, qubits, 2)


def initial_parameters(
    name: AnsatzName,
    electrons: int,
    qubits: int,
    layers: int = 2,
    *,
    seed: int = 0,
    scale: float = 0.05,
):
    shape = parameter_shape(name, electrons, qubits, layers)
    if name == "uccsd":
        return jnp.zeros(shape)
    if scale <= 0:
        raise ValueError("scale must be positive")
    key = jax.random.PRNGKey(seed)
    return scale * jax.random.normal(key, shape)


def apply_ansatz(
    name: AnsatzName,
    parameters,
    problem: H2Problem,
    layers: int = 2,
) -> None:
    name = _validate(name, layers)
    wires = list(range(problem.qubits))
    if name == "uccsd":
        singles, doubles = qp.qchem.excitations(
            problem.electrons,
            problem.qubits,
        )
        single_wires, double_wires = qp.qchem.excitations_to_wires(
            singles,
            doubles,
        )
        qp.UCCSD(
            parameters,
            wires=wires,
            s_wires=single_wires,
            d_wires=double_wires,
            init_state=problem.hf_state,
        )
        return

    qp.BasisState(problem.hf_state, wires=wires)
    for layer in range(layers):
        for wire in wires:
            qp.RY(parameters[layer, wire, 0], wires=wire)
            qp.RZ(parameters[layer, wire, 1], wires=wire)
        for wire in range(problem.qubits - 1):
            qp.CNOT(wires=[wire, wire + 1])
        qp.CNOT(wires=[problem.qubits - 1, 0])
