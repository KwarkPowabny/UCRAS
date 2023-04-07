# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:45:09 2021

@author: Agata
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import Hg_energy_params as params
import Hg_energy_func as func

R_arr, A_S01, A_S13, A_P11, A_P03, A_P13, A_P23 = func.get_a()
E = []
E_R = []
spin = []
N_prev = 0
data = []
eigvect = []

#f = open("part2_spin_HgRb_30_1.txt", 'w')

for mk in params.mk_vals:
    basis_LS, basis_J = func.prep_basis(mk)
    N = len(basis_LS)
    print("mk=", mk, " basis length=", len(basis_LS))
    
    spinM = func.get_spin_matrix(basis_LS)
    
    CGs = func.prep_CG(basis_LS, basis_J)
    print("CGs done")

    wfs = func.read_wf(basis_LS) 
    print("Wave functions done")

    
    E = list(E) + list([np.zeros((len(params.thetas), len(params.R))) for i in range(len(basis_LS))])
    E_R = list(E_R) + list([[] for i in range(len(basis_LS))])
    spin = list(spin) + list([[] for i in range(len(basis_LS))])

    H_hf = np.zeros((len(basis_LS), len(basis_LS)))
    #with open('Hg_energy_output.csv', 'w') as E_out:
    for r in params.R:
    #      E_out.write(str(r) + ' ')
          #f.write(str(r) + " ")
          H_F = np.zeros((len(basis_LS), len(basis_LS)))
          H_diag = np.zeros((len(basis_LS), len(basis_LS)))
          
          
          print(r)
          start = time.time()
          #DIAGONAL PART
          for i in range(len(basis_LS)):
              n = basis_LS[i][0]
              l = basis_LS[i][1]
              L = basis_LS[i][2]
              S = basis_LS[i][3]
              J = basis_LS[i][4]
              if ( L<=3):
                  qd = func.get_qd(L,S,J)
                  H_diag[i][i] += -1/2/(n - (qd-int(qd)))**2
              if( L>3):
                  H_diag[i][i] += -1/2/n**2
  
          #HYPERFINE STRUCTURE
          if(r==min(params.R)): H_hf = func.get_H_hf(basis_LS)   
          
          for theta in params.thetas:
        
              
              #FERMI PSEUDOPOTENTIAL
              A = func.A_matrix(r, theta, basis_LS, basis_J, wfs, CGs)
              H_F = np.matmul(A, np.matmul(func.U_beta(r, basis_J, R_arr, A_S01, A_S13, A_P11, A_P03, A_P13, A_P23), A.T))
              
              #print(A)
      
      
              H = H_F + H_diag + H_hf
              
              
              energy, vect = np.linalg.eigh(H)
              print("Diagonalization time ", time.time()-start)
              
              for i in range(N_prev, N_prev+N):
                  
                  e = energy[i-N_prev]*params.EhtoMHz/1000+ 1/2/params.nH**2*params.EhtoMHz/1000
  
                  E[i][list(params.thetas).index(theta)][list(params.R).index(r)] = e #GHz
                  #E_out.write(str(e) + ' ')
                  if(theta == params.thetas[0]):
                      E_R[i].append(e)
              #E_out.write('\n')
              #f.write("\n")
              eigvect.append(vect)
    print(params.R, E_R[0])
    N_prev+=N
np.save('Hg_energy_matrix', E)
np.save('Hg_eigenvectors', eigvect)
np.save('Hg_eigenstates', energy)
plt.figure(figsize=(8,6))
#plt.xlim(200, 1800)
#plt.ylim(-200, 50)
plt.xlabel("R (bohr)")
plt.ylabel("E (GHz)")
#plt.title("QD hyperfine")
for i in range(N):
    plt.scatter(params.R, E[i], s=2, color='black')

#f.close()
