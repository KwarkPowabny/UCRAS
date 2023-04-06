# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:40:49 2021

@author: Agata
"""

import numpy as np


R_min = 700
R_max = 900
NR = 60
R = np.linspace(R_min, R_max, NR)
print(R)
R_integral = np.arange(5, 2502, 5) #co 1
thetas = np.linspace(0.0,np.pi/2, 1)
print(thetas)

p = 2
q = 1
nH = 30
n_vals = np.arange(nH-p, nH+q+1,1)
s1_vals = [0.5] #Rydberg
sc_vals = [0.5] #Rydberg core electronic spin
lc_vals = [0] #Rydberg core 6s electron 
s2_vals = [0.5] #ground state
i2_vals = [1.5] #ground state
S_values = np.arange(abs(s1_vals[0]-s2_vals[0]), s1_vals[0]+s2_vals[0]+0.1)

mk_vals = [1]

A = 3417.341 #hyperfine structure constant
EhtoMHz = 6.5796e9
A = A/EhtoMHz

#Quantum Defects for Hg
delta_1S0 = 0.6483684210526317
delta_3S1 = 0.6942857142857142
delta_3P0 = 0.2114444444444446
delta_3P1 = 0.20053124999999986
delta_3P2 = 0.09836065573770472
delta_1P1 = 0.05027272727272727
delta_1D2 = 0.07769230769230792
delta_3D1 = 0.0642222222222223
delta_3D2 = 0.057400000000000076
delta_3D3 = 0.04511111111111136
delta_1F3 = 0.0291
delta_3F2 = 0.0351
delta_3F3 = 0.0332
delta_3F4 = 0.0263

