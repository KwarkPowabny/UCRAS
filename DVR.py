# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:04:50 2023

@author: lenovo
"""

import numpy as np
import math as mth
import matplotlib.pyplot as plt


h = 1 #6.62607015*10**(-34)
a, b = -10, 10
N = 1000
d = (b - a)/N
x = np.linspace(a+d, b-d, N)
m = 10
w = 1

v = m*w**2*x**2
V = np.mat(np.diag(v,0))
def Matrix_prep():
    T = np.zeros((N, N))
    for i in range(N-1):
        for j in range(N-1):
            if j == i:
                T[i, i] = h**2/(16*m*(b-a)**2)*((2*N**2+1)/3-1/(mth.sin(mth.pi*(i+1)/(2*N)))**2)
            else:
                T[i][j] = (-1)**(i-j)*h**2/(16*m*(b-a)**2)*(1/(mth.sin(mth.pi*(i-j)/(2*N)))**2- 1/(mth.sin(mth.pi*(i+j)/(2*N)))**2)
    H = T + V
    return H


H = Matrix_prep()
eigval, eigvec = np.linalg.eigh(H)
plt.matshow(H)
plt.colorbar()
plt.show()

sortinds = np.argsort(eigval)
eigvec = eigvec[:,sortinds]
eigval = eigval[sortinds]

plt.figure()
plt.xlabel("State index $n$")
plt.ylabel("E (arb. units)")
plt.plot(eigval[0:99],'o')
plt.show()

plt.figure()
plt.title('DVR test on Harmonic V')
plt.xlabel("R (arb. units)")
plt.ylabel("E (arb. units)")
n = np.linspace(1,999,999)
m = 100
for n in range(m):
    plt.plot(x,np.real(eigvec[:,n]+n))    
plt.plot(x,v)
plt.ylim(-1/2, m)
plt.show()

