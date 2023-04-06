# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:16:49 2023

@author: lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
#import time
import Hg_energy_params as params
import Hg_energy_func as func

Elim = (-200,20)
Rlim = (1300,1700)

basis = func.prep_basis(params.mk_vals)[0]
spins = []
E = []
f1 = np.loadtxt("part1_HgRb_30_1.txt")
f2 = np.loadtxt("part2_HgRb_30_1.txt")
f3 = np.loadtxt("part1_spin_HgRb_30_1.txt")
f4 = np.loadtxt("part2_spin_HgRb_30_1.txt")
R = np.transpose(np.array(list(f1[:,0]) + list(f2[:,0])))
for i in range(len(basis)):
    #E[i] = np.array(list(f1[:,i+1]) + list(f2[:,i+1]))
    E.append(np.array(list(f1[:,i+1]) + list(f2[:,i+1])))
    spins.append(np.array(list(f3[:,i]) + list(f4[:,i])))
            
color = 'rainbow'         
fig1 = plt.figure()
plt.xlabel("R (Bohr)")
plt.ylabel("E (GHz)")
plt.xlim(Rlim)
#plt.ylim(Elim)
for i in range(len(E)//2):
    plt.scatter(R, E[i], s=2, c = spins[i], cmap = color, alpha=0.5)
plt.savefig("Hg_BD_E_R.pdf")
fig2 = plt.figure()
plt.xlabel("R (Bohr)")
plt.ylabel("Spin (arb. units)")
plt.xlim(Rlim)
#plt.ylim(Elim)
for i in range(len(spins)//2):
    plt.scatter(R, spins[i], s=2, c = np.array(spins[i]), cmap = color, alpha=0.5)
plt.savefig("Hg_BD_spin_R.pdf")