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
    spins.append(np.round(np.array(list(f3[:,i]) + list(f4[:,i])), 3))       """


spins =  np.load('Hg_spin_tests_filesS.npy')
E =  np.load('Hg_spin_tests_filesE.npy')
R =  np.load('Hg_spin_tests_filesR.npy')
derivative = np.load('Hg_derE.npy')
color = 'rainbow'
NR = len(E[0])

fig1 = plt.figure()
plt.xlabel("R (Bohr)")
plt.ylabel("E (GHz)")
#plt.xlim(300,750)
#plt.ylim(-000,400)
spins_cutoff = []
E_cutoff = []
spins_cut = []
E_cut = []
E_cut_max = []
derivative_cut = []
m = (0,0)
a = 5 #rząd dokładności
for i in range(len(E)):
    if np.abs(np.sum(np.array(E[i])) - NR*E[i][len(E[i])-5]) < 1e-5 :
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
#func.plot_spin(E_cutoff, R, spins_cutoff, color = color)
#plt.show()
n_biggest = 3 #ile najwyższych stanów plotować
for i in E_cut_max.argsort()[-n_biggest:]:
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.scatter(R, E_cut[i])
    ax1.set_title('State')
    ax2.scatter(R, spins_cut[i])
    ax2.set_title('Spin')
    ax3.scatter(R, derivative_cut[i])
    ax3.set_title('Derivative')
    f.savefig('graphs/Hg_E_cut_biggest_' + str(n_biggest) + '.pdf')
    n_biggest -= 1
print(len(E_cut))
