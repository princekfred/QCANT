import pennylane as qml
import numpy as np
#import excitations
from excitations import inite

def qscEOM(symbols, geometry, active_electrons, active_orbitals, charge,params,ash_excitation, shots=0):
    eig = []
    H, qubits = qml.qchem.molecular_hamiltonian(
        symbols, geometry, basis="sto-3g", method='pyscf',
        active_electrons=active_electrons, active_orbitals=active_orbitals, charge=charge
    )
    hf_state = qml.qchem.hf_state(active_electrons, qubits)
    singles, doubles = qml.qchem.excitations(active_electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    wires = range(qubits)

    null_state = np.zeros(qubits, int)
    list1 = inite(active_electrons, qubits)
    values = []
    for t in range(1):
        if shots == 0:
            dev = qml.device("default.qubit", wires=qubits)
        else:
            dev = qml.device("default.qubit", wires=qubits, shots=shots)

        @qml.qnode(dev)
        def circuit_d(params, occ, wires, s_wires, d_wires, hf_state, ash_excitation):
            qml.BasisState(hf_state, wires=range(qubits))
            for w in occ:
                qml.X(wires=w)
            for i, excitations in enumerate(ash_excitation):
                if len(excitations) == 4:
                    qml.FermionicDoubleExcitation(
                        weight=params[i],
                        wires1=list(range(excitations[0], excitations[1] + 1)),
                        wires2=list(range(excitations[2], excitations[3] + 1))
                    )
                elif len(excitations) == 2:
                    qml.FermionicSingleExcitation(
                        weight=params[i], wires=list(range(excitations[0], excitations[1] + 1))
                    )
            return qml.expval(H)

        @qml.qnode(dev)
        def circuit_od(params, occ1, occ2, wires, s_wires, d_wires, hf_state, ash_excitation):
            qml.BasisState(hf_state, wires=range(qubits))
            for w in occ1:
                qml.X(wires=w)
            first = -1
            for v in occ2:
                if v not in occ1:
                    if first == -1:
                        first = v
                        qml.Hadamard(wires=v)
                    else:
                        qml.CNOT(wires=[first, v])
            for v in occ1:
                if v not in occ2:
                    if first == -1:
                        first = v
                        qml.Hadamard(wires=v)
                    else:
                        qml.CNOT(wires=[first, v])
            for i, excitations in enumerate(ash_excitation):
                if len(excitations) == 4:
                    qml.FermionicDoubleExcitation(
                        weight=params[i],
                        wires1=list(range(excitations[0], excitations[1] + 1)),
                        wires2=list(range(excitations[2], excitations[3] + 1))
                    )
                elif len(excitations) == 2:
                    qml.FermionicSingleExcitation(
                        weight=params[i], wires=list(range(excitations[0], excitations[1] + 1))
                    )
            return qml.expval(H)

        M = np.zeros((len(list1), len(list1)))
        for i in range(len(list1)):
            for j in range(len(list1)):
                if i == j:
                    M[i, i] = circuit_d(params, list1[i], wires, s_wires, d_wires, null_state, ash_excitation)
        for i in range(len(list1)):
            for j in range(len(list1)):
                if i != j:
                    Mtmp = circuit_od(params, list1[i], list1[j], wires, s_wires, d_wires, null_state, ash_excitation)
                    M[i, j] = Mtmp - M[i, i] / 2.0 - M[j, j] / 2.0
        eig, evec = np.linalg.eig(M)
        values.append(np.sort(eig))
    return values



if __name__ == "__main__":
    # Example input data
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    active_electrons = 2
    active_orbitals = 2
    charge = 0
   
