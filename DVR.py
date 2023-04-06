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

h = 1 #6.62607015*10**(-34)
a = 1
b = 2
state = np.load('Hg_eigenvectors.npy')[:,130]
N = 100 - 1 #de facto N-1
d = (b - a)/N
x = np.linspace(a+d, b-d, N, endpoint=True)
m = 1
w = 1
state = interp1d(x, state)
R = params.NR

def Matrix_prep(a, b, h, N, m, w):
    T = np.zeros((N, N))
    V = np.zeros((N, N))
    for i in range(N):
        V[i, i] = m*w**2*(i+1-(a-b)/2)**2 #state[]
        T[i, i] = h**2/(16*m*(b-a)**2)*((2*N**2+1)/3-1/(mth.sin(mth.pi*(i+1)/N))**2)
        for j in range(N - 1):
            if j == i:
                j += 1
            T[i][j] = (-1)**(i-j)*h**2/(16*m*(b-a)**2)*(1/(mth.sin(mth.pi*(i-j)/N))**2/2/N - 1/(mth.sin(mth.pi*(i+j)/N))**2/2/N)
    H = T + V
    return H

H = Matrix_prep(a, b, h, N, m, w)



eigval, eigvec = np.linalg.eigh(H)
dim = len(eigval)
prob, DVR = np.zeros(dim)
for i in range(dim):
    prob[i] = eigvec[i].dot(H.dot(eigvec[i]))
    DVR[i] = np.zeros(stat)

color = 'ocean'
plt.figure()
plt.xlabel("R (bohr)")
plt.ylabel("E (GHz)")
plt.scatter(x, state, s=2, color = 'black')
#plt.title("QD hyperfine")
for i in range(dim):
    plt.scatter(x, DVR[i], s=1.2, c = np.array(prob[i]), cmap = color, alpha=0.5)