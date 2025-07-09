import sys
from scipy import sparse
#set path to the Python_OS_Code directory
sys.path.append('/student/qnj049/Desktop/EPI-SUNDIALS/pythOS')

import fractional_step as fs
import numpy as np
import scipy.linalg as la
import math

#define the parameters
# Monodomain parameters
chi = 1.0  # SurfaceAreaToVolumeRatio
Cm = 1.0   # Capacitance
sigmai = 1.75 # Effective Conductivity = lambda/(1+lambda)*sigma_i where lambda = sigma_e/sigma_i
sigmae = 1.75

# simulation parameters
x1 = 0.
xn = 1.0
Nx = 131
dx=float(xn-x1)/(Nx-1)

# initial condition, may also use a 1 dimensional numpy array here
x = np.linspace(x1, xn, num=Nx)
V0= np.zeros(Nx)
U0 = np.zeros(Nx)
y0 = np.append(V0, U0)


# Create the 1D Laplacian Matrix, M using central differences
# Central difference in x:
firstcol = np.zeros(Nx)
firstcol[0] = -2
firstcol[1] = 1
M = la.toeplitz(firstcol)
M[0,1] = 2
M[-1,-2] = 2
M = 1/(dx**2)*M

def f(t, y, dydt):
    V = y[0:Nx]
    Ue = y[Nx:2*Nx]
    
    f = chi * Cm * 3 * t**2 * np.cos(np.pi * x)*-(sigmai + sigmae)/sigmae + chi * t**3 * -(sigmai + sigmae)/sigmae * np.cos(np.pi * x) - np.pi**2 * sigmae * t**3 * np.cos(np.pi*x)
    result_v = dydt[0:Nx] + V/Cm - sigmai/chi/Cm * (M @ (V + Ue)) - f/chi/Cm
    result = sigmai*(M @ (V + Ue)) + sigmae*(M @ Ue)
    out = np.append(result_v, result)
    #print(t, out[0])
    return out


# list the operators in the order you wish to use them


def f1(t, y, dydt):
    V = y[0:Nx]
    Ue = y[Nx:2*Nx]
    #print(t, V[0], Ue[0], dydt[0], dydt[Nx])
    
    f = chi * Cm * 3 * t**2 * np.cos(np.pi * x)*-(sigmai + sigmae)/sigmae + chi * t**3 * -(sigmai + sigmae)/sigmae * np.cos(np.pi * x) - np.pi**2 * sigmae * t**3 * np.cos(np.pi*x)
    result_v = -dydt[0:Nx]# + sigmai/chi/Cm * (M @ (V + Ue))# - V/Cm# + f/chi/Cm
    result = sigmai*(M @ (V + Ue)) + sigmae*(M @ Ue)
    out = np.append(result_v, result)
    #print(t, out[1])
    return out
def f2(t, y):
    V = y[0:Nx]
    Ue = y[Nx:2*Nx]
    #print(Ue[0], t, V[0])
    
    f = chi * Cm * 3 * t**2 * np.cos(np.pi * x)*-(sigmai + sigmae)/sigmae + chi * t**3 * -(sigmai + sigmae)/sigmae * np.cos(np.pi * x) - np.pi**2 * sigmae * t**3 * np.cos(np.pi*x)
    result_v = np.zeros(Nx) + f/chi/Cm - V / Cm + sigmai/chi/Cm * (M @ (V + Ue))
    result = np.zeros(Nx)
    out = np.append(result_v, result)
    #print(t, out[1])
    return out

def master_function(t,y,dydt, label):
    if label == "f2": return f2(t,y)
    elif label == "f1": return f1(t,y,dydt)
    else:
        raise ValueError("Unknown label: {}".format(label))

tf = 0.005
result = fs.fractional_step([f], tf, y0, 0, tf, 'Godunov', None, {(1,): "ADAPTIVE"}, ivp_methods={1: ("IDA", 1e-4, 1e-8)}, solver_parameters={1: {'id': np.concatenate((np.ones(Nx), np.zeros(Nx))),'max_steps': -1}})

result2 = fs.fractional_step(master_function, 0.005, y0, 0, tf, 'Godunov', ["f2", "f1"], {(1,): "ADAPTIVE", (2,): "ADAPTIVE"}, ivp_methods={2: ("IDA", 1e-4, 1e-8), 1: ("CV_BDF", 1e-4, 1e-8)}, solver_parameters={2: {'id': np.concatenate((np.ones(Nx), np.zeros(Nx))), 'max_steps': -1}})

import matplotlib.pyplot as plt
plt.plot(result[0:Nx],label='IDA')
plt.plot(result[Nx:],label='IDA Ue')
plt.plot(result2[0:Nx],label='split')
plt.plot(result2[Nx:],label='split Ue')
plt.plot(tf**3 * np.cos(np.pi * x),label='V')
plt.plot(-2 * tf ** 3 * np.cos(np.pi * x), label='Ue')
plt.legend()
plt.savefig('bidomain.png')