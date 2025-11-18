import numpy as np
from pyscf import gto, scf, mcscf, fci
from pyscf.mcscf.addons import project_init_guess
import basis_set_exchange as bse
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import scipy
from pennylane.pauli import pauli_sentence
import os
from pennylane import qchem 
import re
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")
import pyscf
hf_energies = []
casci_energies = []

def aps_adapt(adapt_it):
    basis = bse.get_basis('sto-6g', elements=['H'], fmt='nwchem')

    # Define bond distance range for potential energy surface


    print("Computing H4 potential energy surface...")
    print("Bond Distance (Å)     HF Energy (Hartree)      CASCI Energy (Hartree)")
    print("-" * 75)

    # ---------- Step 1: Reference CASSCF calculation ----------
    mol_ref = gto.Mole()
    mol_ref.atom = '''
    H 0 0 0
    H 0 0 3.0
    H 0 0 6.0
    H 0 0 9.0
    '''
    # Use a built-in basis (fix for BasisNotFoundError)
    mol_ref.basis = basis        # or try 'def2-TZVP' / 'ano-rcc-mb' if available
    mol_ref.charge = 0
    mol_ref.spin = 0
    mol_ref.symmetry = False
    mol_ref.build()

    # Perform ROHF + X2C (spin-free)
    mf_ref = scf.RHF(mol_ref)
    mf_ref.level_shift = 0.5
    mf_ref.diis_space = 12
    mf_ref.max_cycle = 100
    mf_ref.kernel()
    if not mf_ref.converged:
        mf_ref = scf.newton(mf_ref).run()

    # One-shot CASSCF to get proper active orbitals
    mycas_ref = mcscf.CASCI(mf_ref, 4, 4)
    h1ecas, ecore = mycas_ref.get_h1eff(mf_ref.mo_coeff)
    print('core energy computed out of casci', ecore)
    h2ecas = mycas_ref.get_h2eff(mf_ref.mo_coeff)
    print('Shape of h1ecas', h1ecas.shape)
    print('Shape of h2ecas', h2ecas.shape)
    print('No of orb', mycas_ref.mo_coeff.shape[1])
    print('No of elec', sum(mycas_ref.nelecas))
    # In your PySCF, kernel() returns 2 values; get orbitals from .mo_coeff
    en = mycas_ref.kernel()
    print('Ref.CASCI energy:', en[0])



    print('H1 eff Ham', h1ecas.shape)
    #print('H2 eff Hamiltonian', h2ecas.shape)


    two_mo = pyscf.ao2mo.restore('1', h2ecas, norb=mycas_ref.mo_coeff.shape[1])
    print('--------------------Converting to physcist notation--------------')
    #--------------------------Converting to physcist notation----------------------------
    two_mo = np.swapaxes(two_mo, 1, 3)


    print('two_mo shape', two_mo.shape)

    print('Core constant', ecore)

    one_mo = h1ecas
    #two_mo = h2ecas
    core_constant = np.array([ecore]) 

    H_fermionic = qml.qchem.fermionic_observable(core_constant, one_mo, two_mo, cutoff=1e-20)

    #print(H_fermionic)

    H = qml.jordan_wigner(H_fermionic)

    qubits = 2*(mycas_ref.mo_coeff.shape[1])
    active_electrons = sum(mycas_ref.nelecas)

    print('Before going to code', qubits)
    print('Before going to adpat no of electrons', active_electrons)


    energies = []
    ash_excitation = []
    hf_state = qml.qchem.hf_state(active_electrons, qubits)
    #Calculation of HF state
    dev = qml.device("lightning.qubit", wires=qubits)
    @qml.qnode(dev)
    def circuit(hf_state, active_electrons, qubits, H): 
        qml.BasisState(hf_state, wires=range(qubits))
        return qml.expval(H)   
    
    # Commutator calculation for HF state
    @qml.qnode(dev)
    def commutator_0(H,w, k):  #H is the Hamiltonian, w is the operator, k is the basis state - HF state
        qml.BasisState(k, wires=range(qubits))
        res = qml.commutator(H, w)   #Calculating the commutator
        return qml.expval(res)
    
    # Commutator calculation for other states except HF state
    @qml.qnode(dev)
    def commutator_1(H,w, k): #H is the Hamiltonian, w is the operator, k is the basis state
        qml.StatePrep(k, wires=range(qubits))
        res = qml.commutator(H, w) #Calculating the commutator
        return qml.expval(res)

    #Energy calculation 
    @qml.qnode(dev)
    def ash(params, ash_excitation, hf_state, H):
        [qml.PauliX(i) for i in np.nonzero(hf_state)[0]]  #Appln of HF state
        for i, excitation in enumerate(ash_excitation):
            if len(ash_excitation[i]) == 4:
                qml.FermionicDoubleExcitation(weight=params[i], wires1=list(range(ash_excitation[i][0], ash_excitation[i][1] + 1)), wires2=list(range(ash_excitation[i][2], ash_excitation[i][3] + 1)))
            elif len(ash_excitation[i]) == 2:
                qml.FermionicSingleExcitation(weight=params[i], wires=list(range(ash_excitation[i][0], ash_excitation[i][1] + 1)))
        return qml.expval(H)  
    
    # Calculation of New state, same as the above function but with the state return
    dev1 = qml.device("lightning.qubit", wires=qubits)
    @qml.qnode(dev1)
    def new_state(hf_state, ash_excitation, params):
        [qml.PauliX(i) for i in np.nonzero(hf_state)[0]] #Applying the HF state
        for i, excitations in enumerate(ash_excitation):
            if len(ash_excitation[i]) == 4:
                qml.FermionicDoubleExcitation(weight=params[i], wires1=list(range(ash_excitation[i][0], ash_excitation[i][1] + 1)), wires2=list(range(ash_excitation[i][2], ash_excitation[i][3] + 1)))
            elif len(ash_excitation[i]) == 2:
                qml.FermionicSingleExcitation(weight=params[i], wires=list(range(ash_excitation[i][0], ash_excitation[i][1] + 1)))
        return qml.state()
    
    

    def cost(params):
        energy = ash(params, ash_excitation, hf_state, H)
        return energy

    #print('HF state is', circuit(hf_state, active_electrons, qubits, H))
    singles, doubles = qml.qchem.excitations(active_electrons, qubits)
    op1 =  [qml.fermi.FermiWord({(0, x[0]): "+", (1, x[1]): "-"}) for x in singles]
    op2 =  [qml.fermi.FermiWord({(0, x[0]): "+", (1, x[1]): "+", (2, x[2]): "-", (3, x[3]): "-"})for x in doubles]
    operator_pool = (op1) + (op2)  #Operator pool - Singles and Doubles
    #print('Total excitations are', len(operator_pool))
    states = [hf_state]
    params = np.zeros(len(ash_excitation), requires_grad=True) 
    

    for j in range(adapt_it):
        print('The adapt iteration now is', j, flush=True)  #Adapt iteration
        max_value = float('-inf')
        max_operator = None
        k = states[-1] if states else hf_state  # if states is empty, fall back to hf_state
    
        for i in operator_pool:
            #print('The current excitation operator is', i)   #Current excitation operator - fermionic one
            w = qml.fermi.jordan_wigner(i)  
            if np.array_equal(k, hf_state): # If the current state is the HF state
                current_value = abs(2*(commutator_0(H, w, k)))      #Commutator calculation is activated  
            else:
                current_value = abs(2*(commutator_1(H, w, k)))      #For other states, commutator calculation is activated

            if current_value > max_value:
                max_value = current_value
                max_operator = i

        indices_str = re.findall(r'\d+', str(max_operator))
        excitations = [int(index) for index in indices_str]
        ash_excitation.append(excitations) #Appending the excitations to the ash_excitation

        params = np.append(params, 0.0)  #Parameters initialization
        #Energy calculation
        result = minimize(cost, params, method='BFGS', tol = 1e-12, options = {'disp': False, 'maxiter': 1e8})

        #print("Final updated parameters:", result.x)
        energies.append(result.fun)
        
        params= (result.x)
        print('Energies are', energies, flush=True)
        ostate = new_state(hf_state, ash_excitation, params)
        #print(qml.draw(new_state, max_length=100)(hf_state,ash_excitation,params))
        #gs_state = ostate
        states.append(ostate)
    print("energies:", energies[-1])
    return params, ash_excitation, energies


