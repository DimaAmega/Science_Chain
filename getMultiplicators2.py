import numpy as np
import math as mt
import mpmath as mp
from scipy.integrate import odeint
from scipy.linalg import eig as getEigenValues
from InitCinditionSinphaseMode  import getInitCondition as IC
def createPhi_s(X,L,G):
        rside = createRsideOnePendulumsEquationMPMATH(L,G)
        q = mp.odefun(rside,0,mp.matrix([0,X[0]]),tol=0.001,degree=4,method="taylor")
        return lambda t:q(t)[0]
def createQ0(N,i):
    res = np.zeros(N)
    res[i] = 1.0
    return res
def createRsideOnePendulumsEquationMPMATH(L,G):
    return lambda t,q: [ q[1] , G-mp.sin(q[0]) -L*q[1] ] 
def RSlinear(q,t,N,L,K,phi_s_mpf):
    phi_s = float(phi_s_mpf(t))
    X = np.zeros(2*N)
    X[0] = q[1]
    n = 0
    X[1] = -L*q[1] - mt.cos(phi_s)*q[0] + K*(q[2] - q[0] ) 
    while n < 2*N-4:
        X[n+2] = q[n+3] 
        X[n+3] = -L*q[n+3] - mt.cos(phi_s)*q[n+2] + K*( q[n+4] - 2*q[n+2] + q[n] )
        n+=2
    X[2*N-2] = q[2*N-1]
    X[2*N-1] = -L*q[2*N-1] - mt.cos(phi_s)*q[2*N-2] + K*( q[2*N-4] - q[2*N-2] )
    return X
def createMatrixResiliense(t_end,N,L,K,phi_s):
        res = np.zeros((2*N,2*N))
        for i in range(2*N):
            q = odeint(RSlinear, createQ0(2*N,i),np.arange(0,t_end + h_iter,h_iter),args=(N,L,K,phi_s),h0=h_iter,hmax=h_iter)
            res[:,i] =  q[-1]
        return res
h_iter = 0.001
######################
##  IMPORT FUNCTION
######################
def getMultiplicators(phi_s,t_end,L,K,G,N):
    res = createMatrixResiliense(t_end,N,L,K,phi_s)
    EIG = getEigenValues(res)
    return EIG[0]
######################
##  MAIN
######################
L = 0.42
K = 0.754
G = 0.97
N = 14
X = IC(L,G)
phi_s = createPhi_s(X,L,G)
res  = getMultiplicators(phi_s,float(X[1]),L,K,G,N)
print(res)


  