# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 08:57:32 2022

@author: Erlend Johansen
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
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
    
    spin_off_z=normalize(np.array([1,0,20],dtype=np.float64))
    
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

    fig,ax=plt.subplots()
    ax.plot(t,S[:,0],label="S_x")
    ax.plot(t,S[:,1],label="S_y")
    ax.plot(t,S[:,2],label="S_z")
    ax.set_xlabel("t [ps]")
    ax.set_ylabel("Spin [unitless]")
    ax.legend()
       
    fig.tight_layout()
    fig.savefig("figures/task_a_time")
    
    fig, ax = plt.subplots()
    t_steps=10000
    points = np.array([S[:t_steps,0], S[:t_steps,1]]).T.reshape(-1,1,2) 
    segments = np.concatenate([points[:-1], points[1:]], axis=1) 
    lc = LineCollection(segments, cmap='cividis', linewidth=3) 
    lc.set_array(t[:t_steps])  
    ax.add_collection(lc) 
    ax.autoscale()
    ax.set_xlabel("S_x [unitless]") 
    ax.set_ylabel("S_y [unitless]") 
    ax.scatter(S[0,0],S[0,1],marker="x",label="Start",c="r",zorder=10)
    ax.legend()
    clb=plt.colorbar(lc)
    clb.ax.set_title('t [ps]',fontsize=15)
    
    fig.tight_layout()
    fig.savefig("figures/task_a_above")
    
        
def task_b():
    global dt
    d_z=0.1
    J=1
    alpha=0.1
    B=np.zeros(3)
    
    spin_off_axis_z=normalize(np.array([1,0,20],dtype=np.float64))
    
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
    
    fig,ax=plt.subplots(nrows=2,sharex=True)
    ax[1].plot(t,S[:,0],label="S_x")
    ax[1].plot(t,S[:,1],label="S_y")
    ax[0].plot(t,S[:,2],label="S_z")
    ax[1].set_xlabel("t [ps]")
    ax[0].set_ylabel("Spin [unitless]")
    ax[1].set_ylabel("Spin [unitless]")
    ax[0].legend()
    ax[1].legend()


    x_0=S_0[1,1,0]
    f=lambda t, omega ,tau: x_0*np.cos(omega*t)*np.exp(-t/tau)
    popt,_=curve_fit(f, t,S[:,0])
    omega=np.abs(popt[0])
    tau=popt[1]
    print(tau, omega, 1/(omega*alpha))
    ax[1].plot(t,x_0*np.exp(-t/tau),"--",label="exp(-t/τ)")
    ax[1].plot(t,-x_0*np.exp(-t*omega*alpha),"--",label="-exp(-tωα)")
    ax[1].legend(loc="upper right")
    
    fig.tight_layout()    
    fig.savefig("figures/task_b_time")
    
    fig, ax = plt.subplots()
    t_steps=30000
    points = np.array([S[:t_steps,0], S[:t_steps,1]]).T.reshape(-1,1,2) 
    segments = np.concatenate([points[:-1], points[1:]], axis=1) 
    lc = LineCollection(segments, cmap='cividis', linewidth=3) 
    lc.set_array(t[:t_steps])  
    ax.add_collection(lc) 
    ax.autoscale()
    ax.set_xlabel("S_x [unitless]") 
    ax.set_ylabel("S_y [unitless]") 
    ax.scatter(S[0,0],S[0,1],marker="x",label="Start",c="r",zorder=10)
    ax.legend()
    
    clb=plt.colorbar(lc)
    clb.ax.set_title('t [ps]',fontsize=15)
    
    fig.tight_layout()    
    fig.savefig("figures/task_b_above")
    

def task_c():
    #global dt
    d_z=0.1
    J=1
    alpha=0.1
    B=np.zeros(3)
    
    spin1=normalize(np.array([1,0,20],dtype=np.float64))
    spin2=normalize(np.array([0,0,1],dtype=np.float64))
    
    time_steps=50000
    particles_x=200
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
    
    S=spin_heun(time_steps, dt, S_0, d_z, B, alpha, J, mu_s, gamma, k_b, x_range, y_range)[1:-1,1]
    t=np.arange(time_steps)*dt
    
    
    
    # plt.figure()
    # plt.imshow(S[:,:,0],aspect="auto")
    # plt.xlabel("t [ps]")
    # plt.ylabel("Spin #")
    # clb=plt.colorbar()
    # clb.ax.set_title('S_x [unitless]', fontsize=10)
    # plt.tight_layout()    
    # plt.savefig("figures/task_c_imshow_x")
    
    # plt.figure()
    # plt.imshow(S[:,:,1],aspect="auto")
    # plt.xlabel("t [ps]")
    # plt.ylabel("Spin #")
    # clb=plt.colorbar()
    # clb.ax.set_title('S_y [unitless]',fontsize=10)
    # plt.tight_layout()    
    # plt.savefig("figures/task_c_imshow_y")
    
    # plt.figure()
    # plt.imshow(1-S[:,:,2],aspect="auto")
    # plt.xlabel("t [ps]")
    # plt.ylabel("Spin #")
    # clb=plt.colorbar()
    # clb.ax.set_title('S_z [unitless]',fontsize=10)
    # plt.tight_layout()    
    # plt.savefig("figures/task_c_imshow_z")
    
    
    #### x over time ####
    plt.figure()
    for x in range(0,3):
        plt.plot(t,S[x,:,0],label=f"Spin nr. {x+1}")
    for x in range(20,23):
        plt.plot(t,S[x,:,0],label=f"Spin nr. {x+1}")    
    plt.xlabel("t [ps]")
    plt.ylabel("S_x [unitless]")
    plt.legend()
    plt.tight_layout()    
    plt.savefig("figures/task_c_time2_x")
    
    plt.figure()
    for x in range(0,3):
        plt.plot(t[20000:30000],S[x,20000:30000,0],label=f"Spin nr. {x+1}")
    for x in range(20,23):
        plt.plot(t[20000:30000],S[x,20000:30000,0],label=f"Spin nr. {x+1}")    
    plt.xlabel("t [ps]")
    plt.ylabel("S_x [unitless]")
    plt.legend()
    plt.tight_layout()    
    plt.savefig("figures/task_c_time3_x")
    
    plt.figure()
    for x in range(0,3):
        plt.plot(t[40000:50000],S[x,40000:50000,0],label=f"Spin nr. {x+1}")
    for x in range(20,23):
        plt.plot(t[40000:50000],S[x,40000:50000,0],label=f"Spin nr. {x+1}")    
    plt.xlabel("t [ps]")
    plt.ylabel("S_x [unitless]")
    plt.legend()
    plt.tight_layout()    
    plt.savefig("figures/task_c_time4_x")
    
    
    #### Close up ####
    t_steps=5000
    plt.figure()
    for x in range(0,3):
        plt.plot(t[:t_steps],S[x,:t_steps,0],label=f"Spin nr. {x+1}")
    plt.xlabel("t [ps]")
    plt.ylabel("S_x [unitless]")
    plt.legend()
    plt.tight_layout()    
    plt.savefig("figures/task_c_time_x")
    
    plt.figure()
    for x in range(0,3):
        plt.plot(t[:t_steps],S[x,:t_steps,1],label=f"Spin nr. {x+1}")
    plt.xlabel("t [ps]")
    plt.ylabel("S_y [unitless]")
    plt.legend()
    plt.tight_layout()    
    plt.savefig("figures/task_c_time_y")
    
    plt.figure()
    for x in range(0,3):
        plt.plot(t[:t_steps],S[x,:t_steps,2],label=f"Spin nr. {x+1}")
    plt.xlabel("t [ps]")
    plt.ylabel("S_z [unitless]")
    plt.legend()
    plt.tight_layout()    
    plt.savefig("figures/task_c_time_z")
    
    
    fig, _ax= plt.subplots(nrows=1,ncols=3,sharey=True, figsize=(16,4))
    ax=_ax.flatten()
    t_steps=5000
    x_max=np.max(np.abs(S[0:4,:t_steps,0]))
    y_max=np.max(np.abs(S[0:4,:t_steps,1]))
    r_max=max(x_max,y_max)*1.05
    
    for x in range(0,3):
        points = np.array([S[x,:t_steps,0], S[x,:t_steps,1]]).T.reshape(-1,1,2) 
        segments = np.concatenate([points[:-1], points[1:]], axis=1) 
        lc = LineCollection(segments, cmap='cividis', linewidth=3) 
        lc.set_array(t[:t_steps])  
        ax[x].add_collection(lc) 
        ax[x].set_xlim(-r_max,r_max)
        ax[x].set_ylim(-r_max,r_max) 
        ax[x].scatter(S[x,0,0],S[x,0,1],marker="x",label="Start",c="r",zorder=10)
        ax[x].set_title(f"Spin nr. {x+1}")
        ax[x].legend()
        ax[x].set_xlabel("S_x [unitless]")
        
    ax[0].set_ylabel("S_y [unitless]")    
    clb=fig.colorbar(lc,ax=ax)
    clb.ax.set_title('t [ps]',fontsize=15)
    
    #fig.tight_layout()    
    fig.savefig("figures/task_c_above")
        
    
def task_d():
    #global dt
    
    d_z=0.1
    alpha=0
    B=np.zeros(3)
    spin1=normalize(np.array([1,0,20],dtype=np.float64))
    spin2=normalize(np.array([0,0,1],dtype=np.float64))
    
    time_steps=50_000
    J=1
    particles_x=200
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
    
    S=spin_heun(time_steps, dt, S_0, d_z, B, alpha, J, mu_s, gamma, k_b, x_range, y_range)[1:-1,1]
    t=np.arange(time_steps)*dt
    
    
    
    plt.figure()
    plt.imshow(S[:,:,0],aspect="auto")
    plt.xlabel("t [ps]")
    plt.ylabel("Spin #")
    clb=plt.colorbar()
    clb.ax.set_title('S_x [unitless]', fontsize=10)
    plt.tight_layout()    
    plt.savefig("figures/task_d_imshow_x")
    
    plt.figure()
    plt.imshow(S[:,:,1],aspect="auto")
    plt.xlabel("t [ps]")
    plt.ylabel("Spin #")
    clb=plt.colorbar()
    clb.ax.set_title('S_y [unitless]',fontsize=10)
    plt.tight_layout()    
    plt.savefig("figures/task_d_imshow_y")
    
    plt.figure()
    plt.imshow(1-S[:,:,2],aspect="auto")
    plt.xlabel("t [ps]")
    plt.ylabel("Spin #")
    clb=plt.colorbar()
    clb.ax.set_title('1-S_z [unitless]',fontsize=10)
    plt.tight_layout()    
    plt.savefig("figures/task_d_imshow_z")
    
    plt.figure()
    for x in range(0,6):
        plt.plot(t,S[x,:,0],label=f"Spin nr. {x+1}")
    plt.xlabel("t [ps]")
    plt.ylabel("S_x [unitless]")
    plt.legend()
    plt.tight_layout()    
    plt.savefig("figures/task_d_time_x")
    
    plt.figure()
    for x in range(0,6):
        plt.plot(t,S[x,:,1],label=f"Spin nr. {x+1}")
    plt.xlabel("t [ps]")
    plt.ylabel("S_y [unitless]")
    plt.legend()
    plt.tight_layout()    
    plt.savefig("figures/task_d_time_y")
    
    plt.figure()
    for x in range(0,6):
        plt.plot(t,S[x,:,2],label=f"Spin nr. {x+1}")
    plt.xlabel("t [ps]")
    plt.ylabel("S_z [unitless]")
    plt.legend()
    
    plt.tight_layout()    
    plt.savefig("figures/task_d_time_z")
    
    
    fig, _ax= plt.subplots(nrows=2,ncols=2,sharey=True, figsize=(12,10))
    ax=_ax.flatten()
    t_steps=5000
    x_max=np.max(np.abs(S[0:4,:t_steps,0]))
    y_max=np.max(np.abs(S[0:4,:t_steps,1]))
    r_max=max(x_max,y_max)*1.1
    
    for x in range(0,4):
        points = np.array([S[x,:t_steps,0], S[x,:t_steps,1]]).T.reshape(-1,1,2) 
        segments = np.concatenate([points[:-1], points[1:]], axis=1) 
        lc = LineCollection(segments, cmap='cividis', linewidth=3) 
        lc.set_array(t[:t_steps])  
        ax[x].add_collection(lc) 
        ax[x].set_xlim(-r_max,r_max)
        ax[x].set_ylim(-r_max,r_max) 
        ax[x].scatter(S[x,0,0],S[x,0,1],marker="x",label="Start",c="r",zorder=10)
        ax[x].set_title(f"Spin nr. {x+1}")
        ax[x].legend()
        ax[x].set_xlabel("S_x [unitless]")
        ax[x].set_ylabel("S_y [unitless]")
        
    clb=fig.colorbar(lc,ax=ax)
    clb.ax.set_title('t [ps]',fontsize=15)
    
    #fig.tight_layout()    
    fig.savefig("figures/task_d_above")
    
    
    
def task_e():
    global dt
    d_z=0.1
    alpha=0
    B=np.zeros(3)
    
    spin1=normalize(np.array([1,0,20],dtype=np.float64))
    spin2=normalize(np.array([0,0,1],dtype=np.float64))
    
    time_steps=10000
    J=1
    particles_x=30
    particles_y=1
    
    X,Y=particles_x, particles_y+2
    S_0=np.zeros((X,Y,3))
    
    x_range=np.arange(0,X)
    y_range=np.arange(1,Y-1)
    
    for x in x_range:
        for y in y_range:
            if x==0:
                S_0[x,y]=spin1
            else:
                S_0[x,y]=spin2
    
    S=spin_heun(time_steps, dt, S_0, d_z, B, alpha, J, mu_s, gamma, k_b, x_range, y_range)[:,1]
    t=np.arange(time_steps)*dt
    
    
    
    plt.figure()
    plt.imshow(S[:,:,0],aspect="auto")
    plt.xlabel("t [ps]")
    plt.ylabel("Spin #")
    clb=plt.colorbar()
    clb.ax.set_title('S_x [unitless]', fontsize=10)
    plt.tight_layout()    
    plt.savefig("figures/task_e_imshow_x")
    
    plt.figure()
    plt.imshow(S[:,:,1],aspect="auto")
    plt.xlabel("t [ps]")
    plt.ylabel("Spin #")
    clb=plt.colorbar()
    clb.ax.set_title('S_y [unitless]',fontsize=10)
    plt.tight_layout()    
    plt.savefig("figures/task_e_imshow_y")
    
    plt.figure()
    plt.imshow(1-S[:,:,2],aspect="auto")
    plt.xlabel("t [ps]")
    plt.ylabel("Spin #")
    clb=plt.colorbar()
    clb.ax.set_title('1-S_z [unitless]',fontsize=10)
    plt.tight_layout()    
    plt.savefig("figures/task_e_imshow_z")
    
    plt.figure()
    for x in range(0,6):
        plt.plot(t,S[x,:,0],label=f"Spin nr. {x+1}")
    plt.xlabel("t [ps]")
    plt.ylabel("S_x [unitless]")
    plt.legend()
    plt.tight_layout()    
    plt.savefig("figures/task_e_time_x")
    
    plt.figure()
    for x in range(0,6):
        plt.plot(t,S[x,:,1],label=f"Spin nr. {x+1}")
    plt.xlabel("t [ps]")
    plt.ylabel("S_y [unitless]")
    plt.legend()
    plt.tight_layout()    
    plt.savefig("figures/task_e_time_y")
    
    plt.figure()
    for x in range(0,6):
        plt.plot(t,S[x,:,2],label=f"Spin nr. {x+1}")
    plt.xlabel("t [ps]")
    plt.ylabel("S_z [unitless]")
    plt.legend()
    
    plt.tight_layout()    
    plt.savefig("figures/task_e_time_z")
    
    
    fig, _ax= plt.subplots(nrows=2,ncols=2,sharey=True, figsize=(12,10))
    ax=_ax.flatten()
    t_steps=5000
    x_max=np.max(np.abs(S[0:4,:t_steps,0]))
    y_max=np.max(np.abs(S[0:4,:t_steps,1]))
    r_max=max(x_max,y_max)*1.1
    
    for x in range(-1,3):
        points = np.array([S[x,:t_steps,0], S[x,:t_steps,1]]).T.reshape(-1,1,2) 
        segments = np.concatenate([points[:-1], points[1:]], axis=1) 
        lc = LineCollection(segments, cmap='cividis', linewidth=3) 
        lc.set_array(t[:t_steps])  
        ax[x].add_collection(lc) 
        ax[x].set_xlim(-r_max,r_max)
        ax[x].set_ylim(-r_max,r_max) 
        ax[x].scatter(S[x,0,0],S[x,0,1],marker="x",label="Start",c="r",zorder=10)
        ax[x].legend()
        ax[x].set_title(f"Spin nr. {x+1}")
        ax[x].set_xlabel("S_x [unitless]")
        ax[x].set_ylabel("S_y [unitless]")
        
    clb=fig.colorbar(lc,ax=ax)
    clb.ax.set_title('t [ps]',fontsize=15)
    
    #fig.tight_layout()    
    fig.savefig("figures/task_e_above")

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
    plt.xlabel("t [ps]")
    plt.ylabel("Spin #")
    clb=plt.colorbar()
    clb.ax.set_title('S_x [unitless]', fontsize=10)
    plt.tight_layout()    
    plt.savefig("figures/task_f_imshow_x")
    
    plt.figure()
    plt.imshow(S[:,:,1],aspect="auto")
    plt.xlabel("t [ps]")
    plt.ylabel("Spin #")
    clb=plt.colorbar()
    clb.ax.set_title('S_y [unitless]',fontsize=10)
    plt.tight_layout()    
    plt.savefig("figures/task_f_imshow_y")
    
    plt.figure()
    plt.imshow(1-S[:,:,2],aspect="auto")
    plt.xlabel("t [ps]")
    plt.ylabel("Spin #")
    clb=plt.colorbar()
    clb.ax.set_title('1-S_z [unitless]',fontsize=10)
    plt.tight_layout()    
    plt.savefig("figures/task_f_imshow_z")

    
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
    alpha=0.5
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
    S, M=thermal_spin_heun(time_steps, dt, S_0, d_z, B, alpha, J, mu_s, gamma, k_b, x_range, y_range, T)
    toc=time()
    print(f"{particles_x*particles_y} particles, {time_steps} steps: Simulation time={toc-tic}")
    
    

    t_avg_M=np.mean(M[time_steps//2:])
    t_var_M=np.var(M[time_steps//2:])
    
    t=np.arange(time_steps)*dt
    plt.figure()
    plt.plot(t,M)
    plt.plot(t[time_steps//2:],np.ones(time_steps)[time_steps//2:]*t_avg_M,"--")
    
    
    
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
                plt.figure()
                plt.plot(t,M)
                plt.plot(t[time_steps//2:],np.ones(time_steps)[time_steps//2:]*t_avg_M,"--")
        
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