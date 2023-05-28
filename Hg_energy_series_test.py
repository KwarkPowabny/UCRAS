# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:30:38 2023

@author: Pawel
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
import Hg_energy_params as params
import Hg_energy_func as func



def spin_analysis(basis, vect, R, num):
    spin_matrix = np.zeros((len(basis), len(basis)))
    spin_analysis = np.zeros((len(basis), len(vect)))
    for i in range(len(basis)):
        if num == "J":
            J = basis[i][4]
            if(J==0):
                spin_matrix[i][i] = 0
            if(J==1):
                spin_matrix[i][i] = 0.5
            if(J==2):
                spin_matrix[i][i] = 1
        if num == "L":
            L = basis[i][2]
            for j in range(30):
                if(L==j):
                    spin_matrix[i][i] = j/29
                
    for j in range(len(vect)):
        for i in range(len(basis)):
            spin_analysis[i][j] = np.round(vect[j][:,i].dot(spin_matrix.dot(vect[j][:,i])),5)
    return spin_analysis




"""
#with open('Hg_energy_output.csv', 'r') as data:
    #data = list[data]
    E = np.load('Hg_energy_matrix.npy')
    plt.figure(figsize=(8,6))
    #plt.xlim(200, 1800)
    #plt.ylim(-200, 50)
    plt.xlabel("R (bohr)")
    plt.ylabel("E (GHz)")
    #plt.title("QD hyperfine")
    for i in range(len(E)):
        plt.scatter(params.R, E[i], s=2, color='black')
"""
num = "L"
Rlim = (725, 750)
Elim = (-40, -20)
basis_LS = func.prep_basis(1.5)[0]
eigvec = np.load('Hg_eigenvectors.npy')
spins = spin_analysis(basis_LS, eigvec, params.R, num)

#fig1 = plt.figure(figsize=(8,6))
#plt.xlim(Rlim)
#plt.ylim(-200, 50)

#plt.title("QD hyperfine")
#for i in range(len(basis_LS)):
"""
plt.xlabel("quantum state (index)")
plt.ylabel("J propability(fraction)")
color = iter(cm.rainbow(np.linspace(0, 1, len(spins))))
for j in range(len(spins)):
    c = next(color)
    plt.scatter(np.arange(0, len(basis_LS), 1), spins[j], s=2, color=c)
"""
E = np.load('Hg_energy_matrix.npy')
plt.figure()
"""
spins_cut = []
energy_cut = []
plt.xlabel("R (Bohr)")
plt.ylabel(num + " propability(fraction)")
for i in range(len(basis_LS)):
    if (abs(E[i][0][0] + 90)  <=  110 and abs(E[i][0][len(E[i][0]) - 1] + 90) <= 110):
        spins_cut.append(spins[i])
        energy_cut.append(E[i][0])
    if len(energy_cut) == 4:
        break
color = iter(cm.prism(np.linspace(0, 1, len(spins_cut) + 1)))
for i in range(len(spins_cut)):
    c = next(color)
    plt.scatter(params.R, spins_cut[i], s=2, color=c)
color = iter(cm.prism(np.linspace(0, 1, len(spins_cut) + 1))) 
plt.savefig("Hg_" + num + "_R_cut.pdf")
fig2 = plt.figure(figsize=(8,6))
plt.xlabel("R (Bohr)")
plt.ylabel("E (GHz")
#plt.xlim(Rlim)
#plt.ylim(Elim)
for i in range(len(energy_cut)):
    c = next(color)
    plt.scatter(params.R, energy_cut[i], s=2, color = c)
plt.savefig("Hg_E_R_cut.pdf")
"""
for i in range(len(basis_LS)):
    plt.plot(params.R, E[i][0])
plt.show()