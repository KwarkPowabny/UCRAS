# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:16:49 2023

@author: Pawel


using altered Hg_energy_func
"""
import numpy as np
import matplotlib.pyplot as plt
import Hg_energy_func as func
"""
spins = []
E = []
f1 = np.loadtxt("part1_HgRb_30_1.txt")
f2 = np.loadtxt("part2_HgRb_30_1.txt")
f3 = np.loadtxt("part1_spin_HgRb_30_1.txt")
f4 = np.loadtxt("part2_spin_HgRb_30_1.txt")
R = np.transpose(np.array(list(f1[:,0]) + list(f2[:,0])))
for i in range(len(f1[0])-1):
    E.append(np.array(list(f1[:,i+1]) + list(f2[:,i+1])))
    spins.append(np.round(np.array(list(f3[:,i]) + list(f4[:,i])), 3))
"""
spins =  np.load('Hg_files_S_cut.npy')
E =  np.load('Hg_files_E_cut.npy')
R =  np.load('Hg_files_R.npy')
der_E = np.load('Hg_files_der_E_cut.npy')
der2_E = np.load('Hg_files_der2_E_cut.npy')
der_spins = np.load('Hg_files_der_spins_cut.npy')
color = 'rainbow'
E_max = []
"""
NR = len(E[0])
spins_cutoff = []
E_cutoff = []
spins_cut = []
E_cut = []
E_cut_max = []
derivative_cut = []
for i in range(len(E)):
    if np.abs(np.sum(np.array(E[i])) - NR*E[i][len(E[i])-5]) < 1e-4 :
        E_cutoff.append(E[i])
        spins_cutoff.append(spins[i])
    else:
        E_cut.append(E[i])
        E_cut_max.append(np.max(E[i]))
        spins_cut.append(spins[i])
        derivative_cut.append(derivative[i])
E_cut = np.array(E_cut)
E_cut_max = np.array(E_cut_max)
E_cutoff = np.array(E_cutoff)
np.save("Hg_E_cut", E_cut)
np.save("Hg_spins_cut", spins_cut)
np.save("Hg_derivative_cut", derivative_cut)
"""
for i in range(len(E)):
    E_max.append(np.max(E[i]))
E_max = np.array(E_max)
n_biggest = 1 #ile najwyższych stanów plotować
for i in E_max.argsort()[-n_biggest:]:
    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)
    plt.xlabel("R (Bohr)")
    ax1.scatter(R, E[i], c = spins[i], cmap = 'viridis')
    ax1.set_title('State', )
    ax2.scatter(R, der_E[i], c = der_E[i], cmap = 'Reds')
    ax2.set_title('Derivative')
    ax3.scatter(R, der2_E[i], c = der2_E[i], cmap = 'Reds')
    ax3.set_title('Second Derivative')
    ax4.scatter(R, spins[i], c = spins[i], cmap = 'viridis')
    ax4.set_title('Spin')
    ax5.scatter(R, der_spins[i], c = der_spins[i], cmap = 'viridis')
    ax5.set_title('Spin_derivative')
    f.savefig('graphs/Hg_E_cut_biggest_' + str(n_biggest) + '.pdf')
    n_biggest -= 1
plt.show()