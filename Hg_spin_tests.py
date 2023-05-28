# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:16:49 2023

@author: Pawel

using altered .functions
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
    spins.append(np.round(np.array(list(f3[:,i]) + list(f4[:,i])), 3))       """


spins =  np.load('Hg_spin_tests_filesS.npy')
E =  np.load('Hg_spin_tests_filesE.npy')
R =  np.load('Hg_spin_tests_filesR.npy')
color = 'rainbow'


fig1 = plt.figure()
plt.xlabel("R (Bohr)")
plt.ylabel("E (GHz)")
#plt.xlim(300,750)
#plt.ylim(-000,400)
avg = []
spins_cutoff = []
E_cutoff = []
spins_cut = []
E_cut = []
m = (0,0)
a = 5 #rząd dokładności
for i in range(len(E)):
    avg.append(round(np.sum(E[i])/len(E[i]), a - 1))
    if  (avg[i] < (1+10**(-a))*E[i][798] and avg[i] > (1-10**(-a))*E[i][798]) :
        E_cutoff.append(E[i])
        spins_cutoff.append(spins[i])
    else:
        E_cut.append(E[i])
        spins_cut.append(spins[i])
func.plot_spin(E_cutoff, R, spins_cutoff, color = color)

maxi = round((np.array(E_cut).argmax())/len(E_cut[0]))
plt.scatter(R, E_cut[maxi], s=10, c = spins_cut[maxi], cmap = color, alpha=0.5, vmin=vmin, vmax=vmax)

plt.show()
