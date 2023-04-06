# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:45:09 2021

@author: Agata
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from sympy.physics.quantum.cg import CG
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d
import math
from mpmath import *
import Hg_energy_params as params
'from numba import jit'    

'@njit'
def prep_basis(mk):
    base_j = []
    base_j_cut = []
    base_LS = []
    base_LS_cut = []
    base_J = []

    #(l1s1)j1
    """for s1 in params.s1_vals:
        for i2 in params.i2_vals:
            mi2_vals = np.arange(-i2, i2+0.1)
            for mi2 in mi2_vals:
                for s2 in params.s2_vals:                                               
                    ms2_vals = np.arange(-s2, s2+0.1)
                    for ms2 in ms2_vals:
                        for n in params.n_vals:
                            for l in range(0,n):
                                for j in np.arange(np.abs(l-s1), np.abs(l+s1)+0.1):
                                    mj_vals = np.arange(-j, j+0.1)
                                    for mj in mj_vals:
                                        if(mj+ms2+mi2==mk):
                                            base_j.append([n, l, j, mj, ms2, mi2])
    
    for vec in base_j:
        if(np.abs(vec[3])<=1.5):
            base_j_cut.append(vec)"""
            
    
    # (LS)J
    for n in params.n_vals:
        for l1 in range(0,n):
            for lc in params.lc_vals:
                L_vals = np.arange(np.abs(l1-lc), np.abs(l1+lc)+0.1)
                for L in L_vals:
                    for s1 in params.s1_vals:
                        for sc in params.sc_vals:
                            S_vals = np.arange(np.abs(s1-sc), np.abs(s1+sc)+0.1)
                            for S in S_vals:
                                J_vals = np.arange(np.abs(L-S), np.abs(L+S)+0.1)
                                for J in J_vals:
                                    mJ_vals = np.arange(-J, J+0.1)
                                    for mJ in mJ_vals:
                                        for s2 in params.s2_vals:
                                            ms2_vals = np.arange(-s2,s2+0.1)
                                            for ms2 in ms2_vals:
                                                for i2 in params.i2_vals:
                                                    mi2_vals = np.arange(-i2,i2+0.1)
                                                    for mi2 in mi2_vals:
                                                        #print(mJ, ms2, mi2)
                                                        if(mJ+ms2+mi2==mk):
                                                            base_LS.append([n, l1, L, S, J, mJ, ms2, mi2])
            
    for vec in base_LS:
        if(np.abs(vec[5])<=2):
            base_LS_cut.append(vec)
            
            
            
    # (LpSp)Jp  
    for i2 in params.i2_vals:
        mi2_vals = np.arange(-i2, i2+0.1)
        for mi2 in mi2_vals:
            for lc in params.lc_vals:
                mlc_vals = np.arange(-lc,lc+0.1)
                for mlc in mlc_vals:
                    for sc in params.sc_vals:
                        msc_vals = np.arange(-sc,sc+0.1) 
                        for msc in msc_vals:
                            for Lp in [0,1]:
                                for Sp in [0, 1]:
                                    Jp_vals = np.arange(np.abs(Lp-Sp), np.abs(Lp+Sp)+0.1)
                                    for Jp in Jp_vals:
                                        mJp_vals = np.arange(-Jp, Jp+0.1)
                                        for mJp in mJp_vals:
                                            if(mJp+mi2+mlc+msc==mk):
                                            #if(mJp==ml
                                                base_J.append([Lp, Sp, Jp, mJp, mlc, msc, mi2])
    
    return base_LS_cut, base_J


def factorial(k): 
    fac = 1
    #print(k)
    for i in range(1,int(k)+1):
        fac=fac*i
    return float(fac)

def prep_file():
    if(params.i2_vals[0]>=0): 
        str1 = "hf"
    else:
        str1 = "nohf"
    f = open("energy_"+str1+"_"+str(min(params.n_vals))+"_"+str(params.nH)+"_"+str(max(params.n_vals))+"_mF"+str(float(params.mk_vals[0]))+".txt", "w")
    return f


def integral(x, y):
    integral = 0
    for i in range(len(x)-1):
        x1 = x[i]
        x2 = x[i+1]
        dx = x2 - x1
        y1 = y[i]
        y2 = y[i+1]
        integral += dx*(y2+y1)/2
    return integral 

def derivative(x0, y):
    dx = 0.001
    dydx = (y(x0 + dx/2) - y(x0-dx/2))/dx
    return dydx

def prep_CG(base_LS, base_J):
    CGs = np.zeros((3, len(base_LS), len(base_J)))
    for i in range(len(base_LS)):
        l1i = base_LS[i][1]
        Li = base_LS[i][2]
        Si = base_LS[i][3]
        Ji = base_LS[i][4]
        mJi = base_LS[i][5]
        ms2i = base_LS[i][6]
        mi2i = base_LS[i][7]
        
        for j in range(len(base_J)):
            Lpj = base_J[j][0]
            Spj = base_J[j][1]
            Jpj = base_J[j][2]
            mJpj = base_J[j][3]
            mlcj = base_J[j][4]
            mscj = base_J[j][5]
            mi2j = base_J[j][6]
            
            if(mi2i==mi2j):
                k=0
                for MLp in [-1,0,1]:
                    CGs[k][i][j] = CG(params.s1_vals[0], mJpj-MLp-ms2i, params.s2_vals[0], ms2i, Spj, mJpj-MLp).doit()*CG(Lpj, MLp, Spj, mJpj-MLp, Jpj, mJpj).doit()*CG(l1i, MLp, params.lc_vals[0], mlcj, Li, MLp).doit()*CG(params.s1_vals[0], mJi-MLp-mscj, params.sc_vals[0], mscj, Si, mJi-MLp).doit()*CG(Li, MLp, Si, mJi-MLp, Ji, mJi).doit()
                    k+=1
    return CGs
    

def get_a():
    a = np.loadtxt("newfort.2001")
    k_arr = a[:,0]
    R_arr = 1/(k_arr**2 /2 + 1/2/params.nH**2)
    print(len(R_arr))
    print(R_arr)
    
    S01 = a[:,1]
    S13 = a[:,2]
    P11 = a[:,3]
    P03 = a[:,4]
    P13 = a[:,5]
    P23 = a[:,6]

    A_S01 = interp1d(R_arr, S01)
    A_S13 = interp1d(R_arr, S13)
    A_P11 = interp1d(R_arr, P11)
    A_P03 = interp1d(R_arr, P03)
    A_P13 = interp1d(R_arr, P13)
    A_P23 = interp1d(R_arr, P23)
    
    return R_arr, A_S01, A_S13, A_P11, A_P03, A_P13, A_P23


def get_qd(L, S, J):
    if(L==0 and S==0 and J==0):
        delta = params.delta_1S0
    if(L==0 and S==1 and J==1):
        delta = params.delta_3S1
        
    if(L==1 and S==0 and J==1):
        delta = params.delta_1P1
    if(L==1 and S==1 and J==0):
        delta = params.delta_3P0
    if(L==1 and S==1 and J==1):
        delta = params.delta_3P1
    if(L==1 and S==1 and J==2):
        delta = params.delta_3P2
    
    if(L==2 and S==0 and J==2):
        delta = params.delta_1D2
    if(L==2 and S==1 and J==1):
        delta = params.delta_3D1
    if(L==2 and S==1 and J==2):
        delta = params.delta_3D2
    if(L==2 and S==1 and J==3):
        delta = params.delta_3D3
        
    if(L==3 and S==0 and J==3):
        delta = params.delta_1F3
    if(L==3 and S==1 and J==2):
        delta = params.delta_3F2
    if(L==3 and S==1 and J==3):
        delta = params.delta_3F3
    if(L==3 and S==1 and J==4):
        delta = params.delta_3F4
        
    return delta


def get_H_hf(base_LS):
    H_hf = np.zeros((len(base_LS), len(base_LS)))
    i2 = params.i2_vals[0]
    s2 = params.s2_vals[0]
    
    for i in range(len(base_LS)):
        ni = base_LS[i][0]
        l1i = base_LS[i][1]
        Li = base_LS[i][2]
        Si = base_LS[i][3]
        Ji = base_LS[i][4]
        mJi = base_LS[i][5]
        ms2i = base_LS[i][6]
        mi2i = base_LS[i][7]

        for j in range(i, len(base_LS)):
            nj = base_LS[j][0]
            l1j = base_LS[j][1]
            Lj = base_LS[j][2]
            Sj = base_LS[j][3]
            Jj = base_LS[j][4]
            mJj = base_LS[j][5]
            ms2j = base_LS[j][6]
            mi2j = base_LS[j][7]

            
            if(i==j):
                H_hf[i][i] += mi2i*ms2i*params.A
            if(ni==nj and l1i==l1j and Li==Lj and Si==Sj and Ji==Jj and mJi==mJj):
                if(mi2i==mi2j+1 and ms2i==ms2j-1):
                    H_hf[i][j] += params.A/2 * np.sqrt(i2*(i2+1) - mi2j*(mi2j+1)) * np.sqrt(s2*(s2+1) - ms2j*(ms2j-1))
                if(mi2i==mi2j-1 and ms2i==ms2j+1):
                    H_hf[i][j] += params.A/2 * np.sqrt(i2*(i2+1) - mi2j*(mi2j-1)) * np.sqrt(s2*(s2+1) - ms2j*(ms2j+1))

            if(i!=j): H_hf[j][i] = H_hf[i][j]
    
    return H_hf


def get_spin_matrix(basis):
    spin_matrix = np.zeros((len(basis), len(basis)))
    for i in range(len(basis)):
        L = basis[i][2]
        S = basis[i][3]
        J = basis[i][4]
        
        if(S==0 and L==0 and J==0):
            spin_matrix[i][i] = 0
        if(S==1 and L==0 and J==1):
            spin_matrix[i][i] = 0.0769
        if(S==1 and L==1 and J==0):
            spin_matrix[i][i] = 2*0.0769
        if(S==1 and L==1 and J==1):
            spin_matrix[i][i] = 3*0.0769
        if(S==1 and L==1 and J==2):
            spin_matrix[i][i] = 4*0.0769
        if(S==0 and L==1 and J==1):
            spin_matrix[i][i] = 5*0.0769
        if(S==0 and L==2 and J==2):
            spin_matrix[i][i] = 6*0.0769
        if(S==1 and L==2 and J==1):
            spin_matrix[i][i] = 7*0.0769
        if(S==1 and L==2 and J==2):
            spin_matrix[i][i] = 8*0.0769
        if(S==1 and L==2 and J==3):
            spin_matrix[i][i] = 9*0.0769
        if(S==1 and L==3 and J==2):
            spin_matrix[i][i] = 10*0.0769
        if(S==1 and L==3 and J==3):
            spin_matrix[i][i] = 11*0.0769
        if(S==1 and L==3 and J==4):
            spin_matrix[i][i] = 12*0.0769
        if(S==0 and L==3 and J==3):
            spin_matrix[i][i] = 13*0.0769
    return spin_matrix
            
def get_spin(vect, spin_matrix):
    exp_spin = np.round(vect.dot(spin_matrix.dot(vect)), 5)
    return exp_spin





def get_spinH(vect, spin_matrix):
    exp_spin = np.round(vect.dot(spin_matrix.dot(vect)), 5)
    return exp_spin






def U_beta(r, base_J, R_arr, A_S01, A_S13, A_P11, A_P03, A_P13, A_P23):
    U_beta = np.zeros((len(base_J), len(base_J)))
    
    for i in range(len(base_J)):
        Lp = base_J[i][0]
        Sp = base_J[i][1]
        Jp = base_J[i][2]
        
        k = np.sqrt(np.abs(-1/(params.nH)**2 + 2/r))
        
        print(Lp, Sp, Jp)
        
        if(Sp==0 and Lp==0 and Jp==0):
            A = -np.tan(A_S01(r))/k
        if(Sp==1 and Lp==0 and Jp==1):
            A = -np.tan(A_S13(r))/k
        if(Sp==0 and Lp==1 and Jp==1):
            A = -np.tan(A_P11(r))/k**3
        if(Sp==1 and Lp==1 and Jp==0):
            A = -np.tan(A_P03(r))/k**3
        if(Sp==1 and Lp==1 and Jp==1):
            A = -np.tan(A_P13(r))/k**3
        if(Sp==1 and Lp==1 and Jp==2):
            A = -np.tan(A_P23(r))/k**3
        U_beta[i][i] = (2*Lp+1)**2 /2 * A
    return U_beta


def Q(l, L, ML, r, theta, wf):
    if(L==0 and ML==0): 
        Q = wf(r) *np.sqrt((2*l+1)/4/np.pi)
    if(L==1 and ML==0):
        Q = np.sqrt((2*l+1)/4/np.pi) * derivative(r, wf)
    if(L==1 and np.abs(ML)==1):
        Q = wf(r)/r * np.sqrt((2*l+1)*(l+1)*l/8/np.pi)
    return float(Q)


def A_matrix(r, theta, base_LS, base_J, wfs, CGs):
    A = np.zeros((len(base_LS), len(base_J)))
    for i in range(len(base_LS)):
        l1i = base_LS[i][1]
        mi2i = base_LS[i][7]
        
        wf = wfs[i]
            
        for j in range(len(base_J)):
            Lpj = base_J[j][0]
            mi2j = base_J[j][6]
            
            if(mi2i==mi2j):
                for ML in np.arange(-Lpj, Lpj+0.1):
                    k = list([-1,0,1]).index(ML)
                    A[i][j] += float(np.sqrt(4*np.pi/(2*Lpj+1))* Q(l1i, Lpj, ML, r, theta, wf)*CGs[k][i][j])


    return A


def read_wf(basis):
    wfs = np.zeros(len(basis), dtype=object)
    r=params.R_integral
    
    for i in range(len(basis)):
        n = basis[i][0]
        l = basis[i][1]
        L = basis[i][2]
        S = basis[i][3]
        J = basis[i][4]
        if(L<=3):
            wf_whit = []
            niu = n-(get_qd(L,S,J) - int(get_qd(L,S,J)))
            for k in r:
                wf_whit.append(whitw(niu, l+0.5, 2*k/(niu))/k/np.sqrt(niu**2*scipy.special.gamma(niu+l+1)*scipy.special.gamma(niu-l)))

            norm = integral(r, r**2 *np.array(wf_whit)**2)
            print(norm)

            wf_whit = np.array(wf_whit)/np.sqrt(norm)
            
            wfs[i] = interp1d(r, wf_whit, kind='cubic')
        else:
            wf =  np.sqrt((2/n)**3 *math.factorial(n-l-1) / 2/n/(math.factorial(n+l))) *np.exp(-r / n) * (2*r /n)**l * scipy.special.genlaguerre(n-l-1, 2*l+1)(2*r /n)
            norm = integral(r, r**2 * np.abs(wf)**2) 
            wf = wf/np.sqrt(norm)
            wfs[i] = interp1d(r, wf)
    return wfs


def plots(N, E, ii, weight):

    plt.figure(figsize=(8,6))
    #plt.xlim(20,600)
    plt.ylim(-200, 30)
    plt.xlabel("$\sqrt{R}$ (bohr)")
    plt.ylabel("E (GHz)")
    for i in range(N):
        #print(np.round(weight[i][0],2))
        #plt.scatter(np.sqrt(params.R), E[i], s=1.4, color='black')

        weight[i][0] = 0
        weight[i][len(weight[i])-1] = 1
        
        plt.scatter(np.sqrt(params.R), E[i], s=14, c=np.array(weight[i]), cmap='nipy_spectral')
    #plt.savefig("Hg_30_triplet.pdf")
    
    plt.figure(figsize=(8,6))
    #plt.xlim(25, 41)
    plt.ylim(-30, 0)
    plt.xlabel("R (bohr)")
    plt.ylabel("E (GHz)")
    #plt.title("QD hyperfine")
    #es = np.loadtxt("30_hf_omega=0.5")
    for i in range(N):
        #e = np.loadtxt("results_RbNOHF_n_28to30to31_M0_501to1000.csv")[:,i+1]
        #e = es[:,i]
        if(weight[i][0]>0.01):
            plt.scatter(np.sqrt(params.R), E[i], s=14, c=weight[i], cmap='nipy_spectral', alpha=0.2)

        #print((e[20:80]*params.EhtoMHz/1000 - E[i])/E[i])

    #plt.savefig("PhD_plot_hf_0.5.pdf")
    
    plt.figure(figsize=(8,6))
    #plt.xlim(200, 1800)
    plt.ylim(-15, -10)
    plt.xlabel("R (bohr)")
    plt.ylabel("E (GHz)")
    #plt.title("QD hyperfine")
    for i in range(N):
        if(i==ii):
            plt.scatter(params.R, E[i], s=2, color='black')
            #f.write(str(params.R) + "\n" + str(E[i]))
        else:
            plt.scatter(params.R, E[i], s=1.4)
    #plt.savefig("Hg_30_triplet_zoom.pdf")
    #f.close()
    
    return 
