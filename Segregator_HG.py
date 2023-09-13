#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:40:56 2023

@author: jakubscioch
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

E=np.load('Hg_E_matrix_g4000.npy')
spins =  np.load('Hg_S_g800.npy')
R_min = 600
R_max = 1800
NR = len(E[0])
R = np.linspace(R_min, R_max, NR)
R_T=np.linspace(R_min, R_max, NR - 2)
R_spin=np.linspace(R_min, R_max, 800)
dr=(R[NR - 1]-R[0])/len(R)
E_cut = []
E_cut2 = []
spins_cut=[]

def sort_states_and_interp_spins(E, E_cut, E_cut2, spins_cut):

    for i in range(len(E)-3):
        if np.abs(np.sum(np.array(E[i])) - NR*E[i][len(E[i])-5]) >1e-5:
            E_cut.append(E[i])
            spins_cut.append(spins[i])
            E_cut2.append(E[i])
                
    E_cut = np.array(E_cut)
    E_cut2 = np.array(E_cut)
    spins_cut=np.array(spins_cut)
    derE = np.zeros((len(E_cut) , len(R)))
    spins_ext = np.zeros((len(E_cut) , len(R)))
    #interpolacja spinów z g800 na g4000 (g - grid)
    for i in range(len(spins_cut)):
        R_arr=np.array(R)
        R_spin_arr=np.array(R_spin)   
        spins_cut_arr=np.array(spins_cut[i])
        spins_interpolate=np.interp(R_arr,R_spin, spins_cut_arr)    
        spins_ext[i]=spins_interpolate

sort_states_and_interp_spins(E, E_cut, E_cut2, spins_cut)

ik = 480
E_sort = []
for p in range(ik):
    E_sort.append([])    

#sortowanie:         
for j in range(1,5,1) :
    print(j) 
    used_states=[]
    for k in range(0,480,1) : 
        if len(E_sort[k])==1 :  
            lista3=  []  
            for t in  [x for x in range(480) if x not in used_states]:       
                q= abs((E_cut2[k][j]-E_sort[t][j-1]))
                lista3.append(q)
            r=min(lista3)
            for s in  [x for x in range(480) if x not in used_states]:
                if abs((E_cut2[k][j]-E_sort[s][j-1])) ==r:
                    E_sort[s].append(E_cut2[k][j]) 
                    used_states.append(s) 
             
                    break 
        else:               
     
            if  len(E_sort[k])==2 or len(E_sort[k])==3 or len(E_sort[k])==4 or len(E_sort[k])==5:                              
                lista3=[]  
                for t in  [x for x in range(480) if x not in used_states]:            
                    q= abs((E_cut2[k][j]-2*E_sort[t][j-1]+E_sort[t][j-2]))  
                    lista3.append(q)
                r=min(lista3)
                for s in  [x for x in range(480) if x not in used_states]:
                    if abs((E_cut2[k][j]-2*E_sort[s][j-1]+E_sort[s][j-2])) ==r:
                        E_sort[s].append(E_cut2[k][j]) 
                        used_states.append(s)
              
                        break 
       
             
        
                                        
for j in range(5,3998,1)  :     
    used_states=[]             
    print(j)
    for k in range(0,480,1)  :         
        lista3=[] 
        for t in  [x for x in range(480) if x not in used_states]:               
            q= abs((5*E_cut2[k][j]-18*E_sort[t][j-1]+24*E_sort[t][j-2]-14*E_sort[t][j-3]+3*E_sort[t][j-4]))  
            lista3.append(q)               
        r=min(lista3)
        for s in  [x for x in range(480) if x not in used_states]:
            if abs((5*E_cut2[k][j]-18*E_sort[s][j-1]+24*E_sort[s][j-2]-14*E_sort[s][j-3]+3*E_sort[s][j-4])) ==r:
                E_sort[s].append(E_cut2[k][j]) 
                used_states.append(s) 
                                  
               
                
                break 
             

#pochodna                             
for i in range(250,310,1) :
    for j in range(len(R)-3):
        dE=(E_sort[i][j+1]-E_sort[i][j])/(dr)
        derE[i][j] = dE 

#zapisywanie energii
"""
with open("Hg_E_matrix_g4000_sort", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(E_sort)
"""
 
 #plotowanie
for i in range(260,308,1):

    f, (ax1, ax2,ax3) = plt.subplots(3, 1, sharex=True)
    ax1.set_title('State'+str(i))
    ax1.set_ylabel('E(GHz)')
    ax2.set_ylabel('GHz/Bohr')
    ax3.set_ylabel('Spin')
    plt.xlabel('R(Bohr)')
    
    ax1.scatter(R_T, E_sort[i-1],s=1,color='red',alpha=0.3)
    ax1.scatter(R_T, E_sort[i],s=6,color='blue',alpha=0.3)
    ax1.scatter(R_T, E_sort[i+1],s=3,color='green',alpha=0.3)
   # ax1.set_ylim(-23.3,-23.27)
    
    ax2.scatter(R, derE[i-1],s=4,color='red',alpha=0.3)
    ax2.scatter(R, derE[i],s=4,color='blue',alpha=0.4)
    ax2.scatter(R,derE[i+1],s=4,color='green',alpha=0.3)
   # ax2.set_xlim(808.5,810)
    #ax2.set_ylim(-0.015,0.015)
    
    ax3.scatter(R, spins_ext[i-1],s=2,color='red',alpha=0.4)
    ax3.scatter(R, spins_ext[i],s=2,color='blue',alpha=0.4)
    ax3.scatter(R, spins_ext[i+1],s=2,color='green',alpha=0.4)
   # ax3.set_ylim(0.30,0.80)
   
    
    plt.show()
   # f.savefig('New_Sort_HG_plot_' + str(i) + '.pdf')
