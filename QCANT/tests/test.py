from sripyscf import adapt_vqe

print('----We are testing adapt code here')

params, ash_excitation,energies  = adapt_vqe( adapt_it=5)
print('params are', params)
print('excitations are', ash_excitation)
print('Energies are', energies)