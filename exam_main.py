# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 08:57:32 2022

@author: Erlend Johansen
"""
import numpy as np
import matplotlib.pyplot as plt
# from numba import njit
from scipy.optimize import curve_fit
from time import time, localtime, asctime
import os

from simulation import normalize, thermal_spin_heun, spin_heun

###### Physical constants ######
k_b=8.617e-2 #meV/K     Boltzmann constant
mu_B=5.788e-2 #meV/T    Bohr magneton
################################

##### Experiment constants #####
mu_s=mu_B  #meV/T
B_0=0.1/mu_s #T
dt=1e-3  #ps
gamma=1.76e-1  # 1/(T ps)
################################


def randomUnitSphereVectors(N: int) -> np.ndarray:
    randomStates=np.zeros((N,3))
    for n in range(N):
        randomStates[n,2]=np.random.random()*2-1
        phi=np.random.random()*2*np.pi
        randomStates[n,0]=np.sqrt(1-randomStates[n,2]**2)*np.cos(phi)
        randomStates[n,1]=np.sqrt(1-randomStates[n,2]**2)*np.sin(phi)                

    return randomStates


def task_a():
    global dt, gamma, mu_s, k_b
    
    d_z=0.1
    J=1
    alpha=0
    B=np.zeros(3)
    
    spin_off_z=normalize(np.array([1,0,2],dtype=np.float64))
    
    particles_x=1
    particles_y=1
    
    X,Y=particles_x+2,particles_y+2
    
    x_range=np.arange(1,X-1)
    y_range=np.arange(1,Y-1)
    
    S_0=np.zeros((X,Y,3))
    
    for x in x_range:
        for y in y_range:
            S_0[x,y]=spin_off_z
    
    time_steps=100000
    S=spin_heun(time_steps, dt, S_0, d_z, B, alpha, J, mu_s, gamma, k_b, x_range, y_range)[1,1]
    t=np.arange(time_steps)*dt
    plt.figure()
    plt.plot(t,S[:,0],label="x")
    plt.plot(t,S[:,1],label="y")
    plt.plot(t,S[:,2],label="z")
        
def task_b():
    global dt
    d_z=0.1
    J=1
    alpha=0.1
    B=np.zeros(3)
    
    spin_off_axis_z=normalize(np.array([1,0,2],dtype=np.float64))
    
    particles_x=1
    particles_y=1
    X,Y=particles_x+2,particles_y+2
    
    x_range=np.arange(1,X-1)
    y_range=np.arange(1,Y-1)
    
    S_0=np.zeros((X,Y,3))
    
    for x in x_range:
        for y in y_range:
            S_0[x,y]=spin_off_axis_z
    
    time_steps=100000
    S=spin_heun(time_steps, dt, S_0, d_z, B, alpha, J, mu_s, gamma, k_b, x_range, y_range)[1,1]
    t=np.arange(time_steps)*dt
    plt.figure()
    plt.plot(t,S[:,0],label="x")
    plt.plot(t,S[:,1],label="y")
    plt.plot(t,S[:,2],label="z")
    
    if alpha>0:
        x_0=S_0[1,1,0]
        f=lambda t, omega ,tau: x_0*np.cos(omega*t)*np.exp(-t/tau)
        popt,_=curve_fit(f, t,S[:,0])
        print(popt)
        omega=np.abs(popt[0])
        tau=popt[1]
        plt.plot(t,x_0*np.exp(-t/tau),"--")
        plt.plot(t,-x_0*np.exp(-t*omega*alpha),"--")
    
    plt.legend(loc="upper right")

def task_c():
    #global dt
    d_z=0.1
    J=1
    alpha=0.1
    B=np.zeros(3)
    
    spin1=normalize(np.array([1,0,0],dtype=np.float64))
    spin2=normalize(np.array([0,0,1],dtype=np.float64))
    
    time_steps=10000
    particles_x=100
    particles_y=1
    
    X,Y=particles_x+2,particles_y+2
    
    x_range=np.arange(1,X-1)
    y_range=np.arange(1,Y-1)
    
    S_0=np.zeros((X,Y,3))
    
    for x in x_range:
        for y in y_range:
            if x==1:
                S_0[x,y]=spin1
            else:
                S_0[x,y]=spin2
    
    S=spin_heun(time_steps, dt, S_0, d_z, B, alpha, J, mu_s, gamma, k_b, x_range, y_range)[:,1]
    t=np.arange(time_steps)*dt
    
    plt.figure()
    plt.imshow(S[1:(X-1),:,0],aspect="auto")
    
    plt.figure()
    for x in range(0,5):
        plt.plot(t,S[x,:,0],label="x")
     
    plt.figure()
    for x in range(0,5):
        plt.plot(t,S[x,:,1],label="x")
    
    plt.figure()
    for x in range(0,5):
        plt.plot(t,S[x,:,2],label="x")
        
def task_d():
    #global dt
    
    d_z=0.1
    alpha=0
    B=np.zeros(3)
    spin1=normalize(np.array([1,0,0],dtype=np.float64))
    spin2=normalize(np.array([0,0,1],dtype=np.float64))
    
    time_steps=100000
    J=1
    particles_x=100
    particles_y=1
    
    X,Y=particles_x+2,particles_y+2
    S_0=np.zeros((X,Y,3))
    
    x_range=np.arange(1,X-1)
    y_range=np.arange(1,Y-1)
    
    for x in x_range:
        for y in y_range:
            if x==1:
                S_0[x,y]=spin1
            else:
                S_0[x,y]=spin2
    
    S=spin_heun(time_steps, dt, S_0, d_z, B, alpha, J, mu_s, gamma, k_b, x_range, y_range)[:,1]
    t=np.arange(time_steps)*dt
    
    plt.figure()
    plt.imshow(S[1:(X-1),:,0],aspect="auto")
    
    for x in range(0,4):
        plt.figure()
        plt.plot(t,S[x,:,0],label="x")
        plt.plot(t,S[x,:,1],label="y")
        plt.plot(t,S[x,:,2],label="z")
        
    plt.figure()
    for x in range(0,4):
        plt.plot(t,S[x,:,2])
    
    
def task_e():
    global dt
    d_z=0.1
    alpha=0
    B=np.zeros(3)
    
    spin1=normalize(np.array([1,0,0],dtype=np.float64))
    spin2=normalize(np.array([0,0,1],dtype=np.float64))
    
    time_steps=100000
    J=1
    particles_x=100
    particles_y=1
    
    X,Y=particles_x, particles_y+2
    S_0=np.zeros((X,Y,3))
    
    x_range=np.arange(0,X)
    y_range=np.arange(1,Y-1)
    
    for x in x_range:
        for y in y_range:
            if x==1:
                S_0[x,y]=spin1
            else:
                S_0[x,y]=spin2
    
    S=spin_heun(time_steps, dt, S_0, d_z, B, alpha, J, mu_s, gamma, k_b, x_range, y_range)[:,1]
    t=np.arange(time_steps)*dt
    
    plt.figure()
    plt.imshow(S[:,:,0],aspect="auto")
    
    for x in range(1,4):
        plt.figure()
        plt.plot(t,S[x,:,0],label="x")
        plt.plot(t,S[x,:,1],label="y")
        plt.plot(t,S[x,:,2],label="z")
        
    plt.figure()
    for x in range(1,4):
        plt.plot(t,S[x,:,2])

def task_f():
    #global dt
    d_z=0.1
    J=1
    alpha=0.1
    B=np.zeros(3)
    
    time_steps=100000
    particles_x=100
    particles_y=1
    
    randomStates=randomUnitSphereVectors(particles_x*particles_y)
    
    X,Y=particles_x,particles_y+2
    S_0=np.zeros((X,Y,3))
    
    x_range=np.arange(0,X)
    y_range=np.arange(1,Y-1)
    
    i=0
    for x in x_range:
        for y in y_range:
            S_0[x,y]=randomStates[i]
            i+=1
    
    S=spin_heun(time_steps, dt, S_0, d_z, B, alpha, J, mu_s, gamma, k_b, x_range, y_range)[:,1]
    
    plt.figure()
    plt.imshow(S[:,:,0],aspect="auto")
    
    plt.figure()
    plt.imshow(S[:,:,1],aspect="auto")
    
    plt.figure()
    plt.imshow(S[:,:,2],aspect="auto")
    
def task_f2D():
    #global dt, B_0
    alpha=0.1
    d_z=0
    B=np.array([0,0,B_0])
    J=1
    
    time_steps=10000
    particles_x=100
    particles_y=100
    
    randomStates=randomUnitSphereVectors(particles_x*particles_y)
    
    X,Y=particles_x,particles_y
    S_0=np.zeros((X,Y,3))
    
    x_range=np.arange(0,X)
    y_range=np.arange(0,Y)
    
    i=0
    for x in x_range:
        for y in y_range:
            S_0[x,y]=randomStates[i]
            i+=1
    
    S=spin_heun(time_steps, dt, S_0, d_z, B, alpha, J, mu_s, gamma, k_b, x_range, y_range)[:,1]
    
    plt.figure()
    plt.imshow(S[:,:,0],aspect="auto")
    
    plt.figure()
    plt.imshow(S[:,:,1],aspect="auto")
    
    plt.figure()
    plt.imshow(S[:,:,2],aspect="auto")
    
def task_g():
    #global dt, B_0
    alpha=0.1
    d_z=0
    B=np.array([0,0,B_0])
    T=20
    J=1
    
    
    time_steps=20_000
    particles_x=30
    particles_y=30
    
    initial_spin=normalize(np.array([0,0,1],dtype=np.float64))
    
    X,Y=particles_x,particles_y
    S_0=np.zeros((X,Y,3))
    
    x_range=np.arange(0,X)
    y_range=np.arange(0,Y)
    
    for x in x_range:
        for y in y_range:
            S_0[x,y]=initial_spin
    
    print(f"Simulation start: {asctime(localtime())}")
    tic=time()
    S, mag_vals=thermal_spin_heun(time_steps, dt, S_0, d_z, B, alpha, J, mu_s, gamma, k_b, x_range, y_range, T)
    toc=time()
    print(f"{particles_x*particles_y} particles, {time_steps} steps: Simulation time={toc-tic}")
    
    t_avg_mag=np.mean(mag_vals[15000:])
    t_var_mag=np.var(mag_vals[15000:])
    
    t=np.arange(time_steps)*dt
    plt.figure()
    plt.plot(t,mag_vals)
    plt.plot(t,np.ones(time_steps)*t_avg_mag,"--")
    
    plt.figure()
    plt.imshow(S[:,:,0],aspect="auto")
    
    plt.figure()
    plt.imshow(S[:,:,1],aspect="auto")
    
    plt.figure()
    plt.imshow(S[:,:,2],aspect="auto")
    
def task_hi_gen():
    #global dt, gamma, mu_s, k_b
    d_z=0
    alpha=0.5
    B=np.array([0,0,B_0])
    J=1
    
    time_steps=100_000
    particles_x=30
    particles_y=30
    
    initial_spin=np.array([0,0,1],dtype=np.float64)
    
    X,Y=particles_x,particles_y
    S_0=np.zeros((X,Y,3))
    
    x_range=np.arange(0,X)
    y_range=np.arange(0,Y)
    
    for x in x_range:
        for y in y_range:
            S_0[x,y]=initial_spin
            

    B_factors=np.array([0.5,1,2,4])
    T_vals=np.arange(0,30)*2
    
    for B_factor in B_factors:
        for T in T_vals:
            print(f"Simulation B={B_factor}B_0 T={T} start: {asctime(localtime())}")
            tic=time()
            S, M=thermal_spin_heun(time_steps, dt, S_0, d_z, B_factor*B, alpha, J, mu_s, gamma, k_b, x_range, y_range, T)
            toc=time()
            print(f"{particles_x*particles_y} particles, {time_steps} steps: Simulation time={toc-tic}")
        
            S_file_name=f"{B_factor}B0_{T}K_S.npy"
            S_file_path=os.path.join("data2",S_file_name)
            M_file_name=f"{B_factor}B0_{T}K_M.npy"
            M_file_path=os.path.join("data2",M_file_name)
            np.save(S_file_path,S)
            np.save(M_file_path,M)
        

def task_hi_plot():
    #global dt, gamma, mu_s, k_b
    
    
    """
    d_z=0
    alpha=0.1
    B=np.array([0,0,B_0])
    J=1
    
    time_steps=100_000
    particles_x=30
    particles_y=30
    """
    
    
    B_factors=np.array([0.5,1,2,4])
    T_vals=np.arange(0,30)*2

    avg=np.zeros((B_factors.shape[0],T_vals.shape[0]))
    var=np.zeros((B_factors.shape[0],T_vals.shape[0]))
    
    for j,B_factor in enumerate(B_factors):            
        for i,T in enumerate(T_vals):        
            #S_file_name=f"{B_factor}B0_{T}K_S.npy"
            #S_file_path=os.path.join("data",S_file_name)
            M_file_name=f"{B_factor}B0_{T}K_M.npy"
            M_file_path=os.path.join("data",M_file_name)
            
            M=np.load(M_file_path)

            time_steps=M.shape[0]
            t_avg_M=np.mean(M[time_steps//2:])
            t_var_M=np.var(M[time_steps//2:])
            
            if i%10==0:
                t=np.arange(time_steps)*dt
                #plt.figure()
                #plt.plot(t,M)
                #plt.plot(t[time_steps//2:],np.ones(time_steps)[time_steps//2:]*t_avg_M,"--")
        
            avg[j,i]=t_avg_M
            var[j,i]=t_var_M
    
        
    fig, ax=plt.subplots(nrows=2,ncols=2)
    for j,B_factor in enumerate(B_factors):
        ax[j//2,j%2].plot(T_vals,avg[j],label=f"M(T) at B={B_factor}B_0")
        ax[j//2,j%2].fill_between(T_vals,avg[j]+np.sqrt(var[j]),avg[j]-np.sqrt(var[j]),alpha=0.3)
        
        if j!=1:
            ax[j//2,j%2].plot(T_vals,avg[1],"--",label=f"M(T) at B=B_0")
        
        ax[j//2,j%2].set_xlabel("T [K]")
        ax[j//2,j%2].set_ylabel("M [a.u.]")
    
        ax[j//2,j%2].legend()  
    
    fig.tight_layout()
    fig.savefig("Phase_diagrams")
        



if __name__=="__main__":
    
    #task_a()
    #task_b()
    #task_c()
    #task_d()
    #task_e()
    #task_f()
    #task_f2D()
    task_g()
    #task_hi_gen()
    #task_hi_plot()

    
    