import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Hg_energy_func as func

ik = 480
E_sort = []
for p in range(ik):
    E_sort.append([])
R_min = 600
R_max = 1800
NR = 3998
#R = np.linspace(R_min, R_max, NR)
R_T=np.linspace(R_min, R_max, NR - 2)
Y=pd.read_csv("Hg_E_matrix_sort_3der.csv")
E_sort = np.array(Y)
E_cut = np.load('Hg_E_matrix_g4000_cut.npy')
E = np.load('Hg_energy_matrix.npy')
spins = np.load('Hg_S_g4000_extrapolated.npy')
R = np.linspace(R_min, R_max, NR)
R_T=np.linspace(R_min, R_max, NR)
R_spin=np.linspace(R_min, R_max, 800)
dr=(R[3997]-R[0])/len(R)

list = [1, 2, 9, 25, 26, 46, 50, 63, 68, 74, 88, 102, 120, 121, 122, 123, 124, 125, 126, 138, 149, 171, 172, 188, 241, 242, 243, 244, 245, 246, 247, 248, 249 ,250 , 251, 269, 292, 320, 362, 363, 364, 365, 366, 367, 368, 369, 370, 374, 378, 381, 386, 400, 406, 417]

xlim = (700,1800)#(656, 658)#
ylim = (-30,0)#(-11.4, -11.33)#

for i in range(0,477,1) :
    #if i in list:
     #   plt.plot(R_T,E_sort[i], marker = 3, color = 'black')
    #else:
        plt.plot(R_T,E_sort[i], marker = 3)    
    #plt.ylim(ylim[0], ylim[1])
    #plt.xlim(xlim[0], xlim[1])
plt.savefig("Hg_states_after_sort.pdf")
plt.show()
             
"""
for i in range(0,len(E),1) :
    plt.plot(R, E[i])    
    plt.ylim(ylim[0], ylim[1])
    plt.xlim(xlim[0], xlim[1])
plt.savefig("Hg_states_before_sort.pdf")
plt.show()

"""

"""
color = 'ocean'
for i in list :
    vmin = np.min(spins_ext) 
    vmax = np.max(spins_ext) 
    plt.scatter(R, E_cut[i], s=10, c = spins_ext[i], cmap = color, alpha=0.5, vmin=vmin, vmax=vmax)    
    #plt.ylim(ylim[0], ylim[1])
    #plt.xlim(xlim[0], xlim[1])
plt.savefig("Hg_states_unchanged_in_sort.pdf")
plt.show()
"""