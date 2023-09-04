# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:16:49 2023

@author: Pawel


using altered Hg_energy_func
"""
import numpy as np
import matplotlib.pyplot as plt
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
    spins.append(np.round(np.array(list(f3[:,i]) + list(f4[:,i])), 3))       """


spins =  np.load('Hg_spin_tests_filesS.npy')
E =  np.load('Hg_spin_tests_filesE.npy')
R =  np.load('Hg_spin_tests_filesR.npy')
color = 'rainbow'
vmin = np.min(spins) 
vmax = np.max(spins) 

fig1 = plt.figure()
plt.xlabel("R (Bohr)")
plt.ylabel("E (GHz)")
plt.xlim(300,750)
plt.ylim(-000,400)
for i in range(len(E)):
    plt.scatter(R, E[i], s=10, c = spins[i], cmap = color, alpha=0.5, vmin=vmin, vmax=vmax)
#plt.savefig("Hg_E_R.pdf")
plt.show()

