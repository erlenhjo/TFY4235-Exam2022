# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 10:08:06 2022

@author: Erlend Johansen
"""
from numba import njit
import numpy as np

@njit (cache=True)   
def normalize(vec: np.ndarray) -> np.ndarray:
    """
    Takes in a vector and normalizes it.
    
    Parameters
    ----------
    vec : np.ndarray
        A vector
        
    Returns
    -------
    np.ndarray
        Normalized vector
    """
    return vec/np.linalg.norm(vec)

@njit(cache=True)
def F_eff_j(S_j: np.ndarray, S_k_vecs: np.ndarray, d_z: float, B: np.ndarray, J: float, mu_s: float) -> np.ndarray:    
    e_z=np.array([0,0,1])
    return J/mu_s*S_k_vecs + 1/mu_s*(2*d_z*S_j*e_z) + B

@njit(cache=True)
def dt_S_j(S_j: np.ndarray, F: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    return -gamma/(1+alpha**2)*(np.cross(S_j, F)+alpha*np.cross(S_j, np.cross(S_j, F)))

@njit(cache=True)
def random3Dgaussians(x_range: np.arange, y_range: np.arange, X: int, Y: int) -> np.ndarray:
    gaussians=np.zeros((X,Y,3))
    for x in x_range:
        for y in y_range:
            gaussians[x,y,0]=np.random.normal()
            gaussians[x,y,1]=np.random.normal()
            gaussians[x,y,2]=np.random.normal()
    return gaussians
    
@njit(cache=True)
def spin_heun(time_steps: int, dt: float, S_0: np.ndarray, d_z: float, B: np.ndarray, alpha: float, J: float, mu_s: float, gamma: float, k_b: float, x_range: np.arange, y_range: np.arange):

    X,Y=np.shape(S_0)[0],np.shape(S_0)[1]
    S=np.zeros((X,Y,time_steps,3))
    
    for x in x_range:
        for y in y_range:
            S[x,y,0]=S_0[x,y]
    
    S_p=np.zeros((X,Y,3))
    f=np.zeros((X,Y,3))
    for n in range(0,time_steps-1):
        #gets euler prediction
        for x in x_range:
            for y in y_range:
                #based on original S_j
                #nearest neighbors
                S_k_vecs=S[x-1,y,n]+S[(x+1)%X,y,n]+S[x,y-1,n]+S[x,(y+1)%Y,n]
                #Effective force and f
                F_eff=F_eff_j(S[x,y,n],S_k_vecs,d_z,B,J,mu_s)
                f[x,y]=dt_S_j(S[x,y,n], F_eff, alpha, gamma)
                #prediction
                S_p[x,y]=normalize(S[x,y,n]+dt*f[x,y])
        #gets heun estimate                
        for x in x_range:
            for y in y_range:
                #based on predicted S_j
                #nearest neighbors
                S_k_vecs_p=S_p[x-1,y]+S_p[(x+1)%X,y]+S_p[x,y-1]+S_p[x,(y+1)%Y]
                #Effective force and  f
                F_eff_p=F_eff_j(S_p[x,y],S_k_vecs_p,d_z,B,J,mu_s)
                f_p=dt_S_j(S_p[x,y], F_eff_p, alpha, gamma)
                #heun estimate
                S[x,y,n+1]=normalize(S[x,y,n]+dt*(f[x,y]+f_p)/2)
    return S

@njit(cache=True)
def thermal_spin_heun(time_steps: int, dt: float, S_0: np.ndarray, d_z: float, B: np.ndarray, alpha: float, J: float, mu_s: float, gamma: float, k_b: float, x_range: np.arange, y_range: np.arange, T: float):
    
    X,Y=np.shape(S_0)[0],np.shape(S_0)[1]
    
    S=np.zeros((X,Y,3))
    
    for x in x_range:
        for y in y_range:
            S[x,y]=S_0[x,y]
            
    M=np.zeros(time_steps)
    M[0]=np.mean(S[:,:,2])
    
    S_p=np.zeros((X,Y,3))
    f=np.zeros((X,Y,3))
    noise_factor=np.sqrt((2*alpha*k_b*T)/(gamma*mu_s*dt))
    
    for n in range(0,time_steps-1):
        F_th=random3Dgaussians(x_range,y_range,X,Y)*noise_factor
        #gets euler prediction
        for x in x_range:
            for y in y_range:
                #based on original S_j
                #nearest neighbors
                S_k_vecs=S[x-1,y]+S[(x+1)%X,y]+S[x,y-1]+S[x,(y+1)%Y]
                #Effective force and f
                F_eff=F_eff_j(S[x,y],S_k_vecs,d_z,B,J,mu_s)
                f[x,y]=dt_S_j(S[x,y], F_eff+F_th[x,y], alpha, gamma)
                #prediction
                S_p[x,y]=normalize(S[x,y]+dt*f[x,y])
        #gets heun estimate                
        for x in x_range:
            for y in y_range:
                #based on predicted S_j
                #nearest neighbors
                S_k_vecs_p=S_p[x-1,y]+S_p[(x+1)%X,y]+S_p[x,y-1]+S_p[x,(y+1)%Y]
                #Effective force and  f
                F_eff_p=F_eff_j(S_p[x,y],S_k_vecs_p,d_z,B,J,mu_s)
                f_p=dt_S_j(S_p[x,y], F_eff_p+F_th[x,y], alpha, gamma)
                #prediction
                S[x,y]=normalize(S[x,y]+dt*(f[x,y]+f_p)/2)
        
        M[n+1]=np.mean(S[:,:,2])
        
    return S, M
