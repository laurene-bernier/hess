#!/usr/bin/env python
import numpy as np
from qutip_utils import (
    all_occupations, random_st_qubit_state,
    qubits_impulsion, hbar_eVs, qubit_impurities, rabi_freqs_per_segment)
import importlib, qutip_utils as qu
importlib.reload(qu)           # remet à jour le module
from qutip_utils import qubit_impurities   # ou toute autre fonction
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  # ou 'Qt5Agg', selon ce que tu as d’installé
plt.ion()                # mode interactif ON


num_sites     = 3
n_electrons   = 3                    # demi-remplissage


t_base = np.zeros((num_sites, num_sites))
t_base[0, 1] = t_base[1, 0] = 1e-3     # lien intrapair (qubit ST)
t_base[1, 2] = t_base[2, 1] = 1e-4    # lien intrapair (qubit ST)
U_base = np.array([3, 3, 60]) * 1e-3     # eV
seed = 2

basis_occ = all_occupations(num_sites, n_electrons)


psi_L = random_st_qubit_state(basis_occ, pair=(0, 1), seed=seed)   
psi_spin = (1, 0)


times, amps_list, coords_list, spin_amps, all_states = qubit_impurities(
    num_sites, n_electrons,
    t_base, U_base,
    t_final=1e-8,
    psi0=[psi_L, psi_spin],             
    display=False,
    seed = seed,
    freq_GHz=20                      
)


f_Rabi_Hz  = rabi_freqs_per_segment(times, amps_list, spin_amps, all_states)
f_Rabi_GHz = [[None if f is None else f/1e9 for f in row] for row in f_Rabi_Hz]

spin_state = (np.asarray(spin_amps) > 0).astype(int)
idx_flip   = np.r_[0, np.where(np.diff(spin_state)!=0)[0]+1, len(times)-1]
t_mid_ns   = 1e9 * 0.5 * (times[idx_flip[:-1]] + times[idx_flip[1:]-1])

"""
fig, ax = plt.subplots(figsize=(6,3))      
ax.plot(t_mid_ns, f_Rabi_GHz[0], 'o-', label='Qubit 0')
ax.set_xlabel('Temps segment (ns)')
ax.set_ylabel('f_Rabi (GHz)')
ax.set_title('Fréquence de Rabi – qubit 0')
ax.legend()
plt.tight_layout()
plt.show()
"""



fig, ax = plt.subplots(figsize=(6,3))     
valid_freqs = [f for f in f_Rabi_GHz[0] if f is not None]

ax.hist(valid_freqs, label='Qubit 0', bins=100)

ax.set_title('Fréquence de Rabi – qubit 0')
ax.legend()
plt.tight_layout()
plt.show()

