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
from scipy.optimize import fsolve

h = 1 #6.62607015*10**(-34)
a = params.R_min
b = params.R_max
state_num = 130
NR = params.NR
N = 100 - 1 #de facto N-1
d = (b - a)/N
x = np.linspace(a+d, b-d, N, endpoint=True)
m = 1
w = 1

def Matrix_prep(a, b, h, N, m, w):
    T = np.zeros((N, N))
    V = np.zeros((N, N))
    for i in range(N):
        V[i, i] = m*w**2*(i+1-(a-b)/2)**2 #state[]
        T[i, i] = h**2/(16*m*(b-a)**2)*((2*N**2+1)/3-1/(mth.sin(mth.pi*(i+1)/N))**2)
        for j in range(N - 1):
            if j == i:
                j += 1
            T[i][j] = (-1)**(i-j)*h**2/(16*m*(b-a)**2)*(1/(mth.sin(mth.pi*(i-j)/N))**2/2/N \
                                                        - 1/(mth.sin(mth.pi*(i+j)/N))**2/2/N)
    H = T + V
    return H

H = Matrix_prep(a, b, h, N, m, w)



eigval, eigvec = np.linalg.eigh(H)
dim = len(eigval)
state = np.zeros(NR)
statefunc = interp1d( params.R, state)
state = statefunc(x)
for i in range(NR):
    state[i] = -(eigval[0]/100000)*(i-NR/2)**2
#state = np.load('Hg_eigenvalues.npy')[:,state_num]
prob = np.zeros(dim)
x_red = np.zeros(dim)
def diff(x0, i):
    return statefunc(x0) - eigval[i]
for i in range(dim):
    prob[i] = np.round(eigvec[i].dot(H.dot(eigvec[i])), 5)
    #solve = fsolve(diff, [a + 10*d, b - 10*d], args=(i,))
    #x_red[1] = np.arrange(solve[0], solve[1], d)

color = 'ocean'
plt.figure()
plt.xlabel("R (bohr)")
plt.ylabel("E (GHz)")
plt.title('DVR results for ' + 'arbitrary potential well')#+ str(state_num) + 'th state potential')
plt.scatter(x, state, s=2, color = 'black')
#plt.title("QD hyperfine")
for i in range(dim):
    #plt.scatter(x_red[i], eigval[i], s = 1.2,  c = np.array(prob[i]), cmap = color, alpha=0.5)
    plt.axhline(y=eigval[i])
plt.savefig("Hg_DVR.pdf")