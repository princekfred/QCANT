from sripyscf import aps_adapt

print('----We are testing adapt code here')

params, ash_excitation,energies  = aps_adapt( adapt_it=5)
print('params are', params)
print('excitations are', ash_excitation)
print('Energies are', energies)