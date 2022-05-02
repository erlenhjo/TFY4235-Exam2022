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
    """
    Calculates the "Force" experienced by the spin of state j

    Parameters
    ----------
    S_j : np.ndarray
        Spin vector of spin j
    S_k_vecs : np.ndarray
        Sum of spin vectors of nearest neighbors to j
    d_z : float
        Anisotropy constant.
    B : np.ndarray
        External magnetic field
    J : float
        Coupling factor between spins
    mu_s : float
        Single spin magnetic moment

    Returns
    -------
    np.ndarray(3)
        "Force" experienced by the spin j

    """
    e_z=np.array([0,0,1])
    return J/mu_s*S_k_vecs + 1/mu_s*(2*d_z*S_j*e_z) + B

@njit(cache=True)
def dt_S_j(S_j: np.ndarray, F: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """
    Calculates the change in spin per time 

    Parameters
    ----------
    S_j : np.ndarray
        Spin vector of spin j
    F : np.ndarray
        Force experienced by S_j
    alpha : float
        Damping factor
    gamma : float
        Gyromagnetic ratio

    Returns
    -------
    np.ndarray
        Change in spin per time

    """
    return -gamma/(1+alpha**2)*(np.cross(S_j, F)+alpha*np.cross(S_j, np.cross(S_j, F)))

@njit(cache=True)
def random3Dgaussians(x_range: np.arange, y_range: np.arange, X: int, Y: int) -> np.ndarray:
    """
    Generates a vector of three random, independent gaussians of zero mean
    and unit variance for each (x,y) looped through by x_range and y_range

    Parameters
    ----------
    x_range : np.arange
        A range of all x with spin.
    y_range : np.arange
        A range of all y with spin
    X : int
        Spin in x direction plus 2 if non-periodic/padded
    Y : int
        Spin in y direction plus 2 if non-periodic/padded

    Returns
    -------
    gaussians : np.ndarray
        A 2D array filled with arrays of length 3 of random gaussian numbers
    """
    gaussians=np.zeros((X,Y,3))
    for x in x_range:
        for y in y_range:
            gaussians[x,y,0]=np.random.normal()
            gaussians[x,y,1]=np.random.normal()
            gaussians[x,y,2]=np.random.normal()
    return gaussians
    
@njit(cache=True)
def spin_heun(time_steps: int, dt: float, S_0: np.ndarray, d_z: float, B: np.ndarray, alpha: float, J: float, mu_s: float, gamma: float, k_b: float, x_range: np.arange, y_range: np.arange):
    """
    Runs the Heun method for a 2D array of spin vectors, no thermal fluctuation
    Saves all states at all t

    Boundary conditions are determined by the initial condition given
    by S_0 and sadly also X,Y,x_range and y_range. Someone should fix
    this and just pass in a bool each for x and y determining the
    boundary conditions.

    Parameters
    ----------
    time_steps : int
        Number of time steps
    dt : float
        Time passed between time steps
    S_0 : np.ndarray
        Initial condition
    d_z : float
        Anisotropy factor
    B : np.ndarray
        External magnetic field
    alpha : float
        Damping factor
    J : float
        Coupling constant
    mu_s : float
        Magnetic moment of a single spin
    gamma : float
        Gyromagnetic ratio
    k_b : float
        Boltzmann constant in apropriate units
    x_range : np.arange
        A range of all x with spin.
    y_range : np.arange
        A range of all y with spin.

    Returns
    -------
    S : np.ndarray
        All spin vectors in t for all spin

    """        
    #Deals with boundary conditions and sets 
    #the initial state
    #Should probably calculate x_range and y_range here
    #and input bool : periodic x and bool : periodic y
    X,Y=np.shape(S_0)[0],np.shape(S_0)[1]
    S=np.zeros((X,Y,time_steps,3))
    for x in x_range:
        for y in y_range:
            S[x,y,0]=S_0[x,y]
    
    #initialize arrays for use in loop
    S_p=np.zeros((X,Y,3))
    f=np.zeros((X,Y,3))
    
    #loops over all time steps
    for n in range(0,time_steps-1):
        #gets euler estimate
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
    """
    Runs the Heun method for a 2D array of spin vectors with thermal fluctuation
    Only saves magnetization M at each t and final state

    Boundary conditions are determined by the initial condition given
    by S_0 and sadly also X,Y,x_range and y_range. Someone should fix
    this and just pass in a bool each for x and y determining the
    boundary conditions.

    Parameters
    ----------
    time_steps : int
        Number of time steps
    dt : float
        Time passed between time steps
    S_0 : np.ndarray
        Initial condition
    d_z : float
        Anisotropy factor
    B : np.ndarray
        External magnetic field
    alpha : float
        Damping factor
    J : float
        Coupling constant
    mu_s : float
        Magnetic moment of a single spin
    gamma : float
        Gyromagnetic ratio
    k_b : float
        Boltzmann constant in apropriate units
    x_range : np.arange
        A range of all x with spin.
    y_range : np.arange
        A range of all y with spin.

    Returns
    -------
    S : np.ndarray
        All spin vectors at last t
    M : np.ndarray
        Magnetization at each t
    """
    
    #Deals with boundary conditions and sets 
    #the initial state
    #Should probably calculate x_range and y_range here
    #and input bool : periodic x and bool : periodic y
    X,Y=np.shape(S_0)[0],np.shape(S_0)[1]
    S=np.zeros((X,Y,3))
    for x in x_range:
        for y in y_range:
            S[x,y]=S_0[x,y]
    
    #initialize magnetization array and calculate at t=0
    M=np.zeros(time_steps)
    M[0]=np.mean(S[:,:,2])
    
    #Initialize arrays for use in for loop
    S_p=np.zeros((X,Y,3))
    f=np.zeros((X,Y,3))
    
    #calculate constant factor for entire loop
    #to avoid doing it within
    noise_factor=np.sqrt((2*alpha*k_b*T)/(gamma*mu_s*dt))
    
    #loops over each time step
    for n in range(0,time_steps-1):
        F_th=random3Dgaussians(x_range,y_range,X,Y)*noise_factor
        #gets euler prediction at each spin
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
        #gets heun estimate at each spin        
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
        
        #calculate magnetization based on calculated spin states
        #technically wrong if non-periodic, but can't be bothered
        #as i will never run it for non-periodic
        M[n+1]=np.mean(S[:,:,2])
        
    return S, M
