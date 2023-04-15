# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:04:50 2023

@author: lenovo
"""

import numpy as np
import math as mth
import Hg_energy_params as params
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

"""Nie istotne, wip lepszego wyświetlania i DVR nie testowego.
#from scipy.optimize import fsolve 


#a = params.R_min
#b = params.R_max

#potential_num = 130
#NR = params.NR
"""
h = 6.62607015*10**(-34)
a, b = -1, 1
N = 100
d = (b - a)/(N)
x = np.linspace(a+d, b-d, N - 1)
m = 1
w = 1

def Matrix_prep():
    T = V = np.zeros((N - 1, N - 1))
    potential = np.zeros(N - 1)
    for i in range(N-1):
        V[i, i] = potential[i] = m*w**2*x[i]**2 #potential[potential_num]
        T[i, i] = h**2/(16*m*(b-a)**2)*((2*N**2+1)/3-1/(mth.sin(mth.pi*(i+1)/N))**2)
        for j in range(N-1):
            if j != i:
                T[i][j] = (-1)**(i-j)*h**2/(16*m*(b-a)**2)*(1/(mth.sin(mth.pi*(i-j)/2*N))**2 \
                                                        - 1/(mth.sin(mth.pi*(i+j+2)/2*N))**2)
    H = T + V
    return H, potential


H, potential = Matrix_prep()
eigval, eigvec = np.linalg.eigh(H)
plt.matshow(H)
plt.colorbar()
plt.show()

"""Nie istotne, wip lepszego wyświetlania i DVR nie testowego.
potential = np.zeros(NR)
potentialfunc = interp1d( params.R, potential)
potential = potentialfunc(x)
for i in range(NR):
    potential[i] = -(eigval[0]/100000)*(i-NR/2)**2
potential = np.load('Hg_eigenvalues.npy')[:,potential_num]
prob = np.zeros(N-1)
x_red = np.zeros(N-1)
def diff(x0, i):
    return potentialfunc(x0) - eigval[i]
for i in range(N-1):
    prob[i] = np.round(eigvec[i].dot(eigvec[i]), 5)
    #solve = fsolve(diff, [a + 10*d, b - 10*d], args=(i,))
    #x_red[1] = np.arrange(solve[0], solve[1], d)

#color = 'ocean'
plt.figure()
plt.xlabel("R (bohr)")
plt.ylabel("E (GHz)")
plt.title('DVR results for ' + 'arbitrary potential well')#+ str(potential_num) + 'th potential potential')
plt.scatter(x, potential, s=2, color = 'black')
for i in range(N-1):
    plt.scatter(x_red[i], eigval[i], s = 1.2,  c = np.array(prob[i]), cmap = color, alpha=0.5)
    plt.axhline(y=eigval[i])
plt.savefig("Hg_DVR.pdf")
"""

plt.figure()
plt.xlabel("R (arb. units)")
plt.ylabel("E (arb. units)")
plt.title('DVR test on Harmonic Potential')
for i in range(N-1):
    plt.axhline(y=eigval[i])
    #plt.text(0.5, eigval[i], 'E = ' + str(eigval[i]), fontsize=5, va='center', ha='center', backgroundcolor='w')
plt.plot(x, potential, color = 'black')
#plt.savefig("Hg_DVR.pdf")
plt.show()