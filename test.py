import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ik = 480
E_sort = []
for p in range(ik):
    E_sort.append([])
R_min = 600
R_max = 1800
NR = 3998
#R = np.linspace(R_min, R_max, NR)
R_T=np.linspace(R_min, R_max, NR - 2)
Y=pd.read_csv("Hg_E_matrix_g4000_sort")
E_sort = np.array(Y)
E=np.load('Hg_E_matrix_g4000.npy')
spins =  np.load('Hg_S_g800.npy')
R = np.linspace(R_min, R_max, NR)
R_T=np.linspace(R_min, R_max, 3996)
R_spin=np.linspace(R_min, R_max, 800)
dr=(R[3997]-R[0])/len(R)
E_cut = []
E_cut2 = []
spins_cut=[]

xlim = (700,1800)#(656, 658)#
ylim = (-30,0)#(-11.4, -11.33)#
for i in range(0,477,1) :
 plt.plot(R_T,E_sort[i])    
 plt.ylim(ylim[0], ylim[1])
 plt.xlim(xlim[0], xlim[1])
plt.savefig("Hg_states_after_sort.pdf")
plt.show()

for i in range(len(E)-3):
        if np.abs(np.sum(np.array(E[i])) - NR*E[i][len(E[i])-5]) >1e-5:
            E_cut.append(E[i])
            spins_cut.append(spins[i])                

E_cut = np.array(E_cut)
spins_cut=np.array(spins_cut)
derE2 = np.zeros((len(E_cut) , len(R)))
spins_ext = np.zeros((len(E_cut) , len(R)))
test_arr = np.zeros((len(E_cut) , len(R)))
for i in range(len(spins_cut)):
    R_arr=np.array(R)
    R_spin_arr=np.array(R_spin)   
    spins_cut_arr=np.array(spins_cut[i])
    spins_interpolate=np.interp(R_arr,R_spin, spins_cut_arr)    
    spins_ext[i]=spins_interpolate

for i in range(0,478,1) :
    plt.plot(R, E_cut[i])    
    plt.ylim(ylim[0], ylim[1])
    plt.xlim(xlim[0], xlim[1])
plt.savefig("Hg_states_before_sort.pdf")
plt.show()