from dataclasses import dataclass

import numpy as np
import pennylane as qp
from pyscf import fci, gto, scf


@dataclass(frozen=True)
class H2Problem:
    bond_length: float
    hamiltonian: qp.Hamiltonian
    qubits: int
    electrons: int
    hf_state: np.ndarray
    hf_energy: float
    fci_energy: float


def build_h2_problem(bond_length: float) -> H2Problem:
    if bond_length <= 0:
        raise ValueError("bond_length must be positive")

    symbols = ["H", "H"]
    coordinates = np.array(
        [[0.0, 0.0, -bond_length / 2], [0.0, 0.0, bond_length / 2]]
    )
    molecule = qp.qchem.Molecule(
        symbols,
        coordinates,
        basis_name="sto-3g",
        unit="angstrom",
    )
    hamiltonian, qubits = qp.qchem.molecular_hamiltonian(
        molecule,
        method="pyscf",
        mapping="jordan_wigner",
    )

    pyscf_molecule = gto.M(
        atom=f"H 0 0 {-bond_length / 2}; H 0 0 {bond_length / 2}",
        basis="sto-3g",
        unit="Angstrom",
        spin=0,
        charge=0,
        verbose=0,
    )
    mean_field = scf.RHF(pyscf_molecule).run(verbose=0)
    fci_solver = fci.FCI(mean_field)
    fci_energy, _ = fci_solver.kernel()

    electrons = 2
    return H2Problem(
        bond_length=bond_length,
        hamiltonian=hamiltonian,
        qubits=qubits,
        electrons=electrons,
        hf_state=qp.qchem.hf_state(electrons, qubits),
        hf_energy=float(mean_field.e_tot),
        fci_energy=float(fci_energy),
    )
