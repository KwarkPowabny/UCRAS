import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
#import time
import Hg_energy_params as params
import Hg_energy_func as func


E = np.load('Hg_E_cut.npy')
basis = func.prep_basis(params.mk_vals)[0]
spins = []
f1 = np.loadtxt("part1_HgRb_30_1.txt")
f2 = np.loadtxt("part2_HgRb_30_1.txt")
#f3 = np.loadtxt("part1_spin_HgRb_30_1.txt")
#f4 = np.loadtxt("part2_spin_HgRb_30_1.txt")
R = np.transpose(np.array(list(f1[:,0]) + list(f2[:,0])))
newlist = np.zeros((len(E) , len(R)))

E_T=np.array(E).T.tolist()

plt.ylim(-50,200)

plt.plot(E_T)
plt.show()



print(E[0][0])           
dr=(R[799]-R[0])/len(R)
for i in range(len(E)):
 for j in range(len(R)-1):
  dE=(E[i][j+1]-E[i][j])/dr
  newlist[i][j] = dE
  if dE >= 10000:
   print(R[j],dE,i)

    
newlist_transpose = newlist.transpose()  



Hg_crossings = open('Hg_crossings_Ecut.txt', 'w') 
for k in range(len(R)) :
 for j in range(len(E)):
  for i in range(len(E)):
   if j==i:   
    1 
   else :
    if E[i][k]==E[j][k] and 50>E[i][k]>(-200):
     print(R[k],E[i][k],i,j,file=Hg_crossings)  
     
Hg_crossings.close() 
