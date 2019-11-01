from InitCinditionSinphaseMode  import getInitCondition as IC
from getMultiplicators2 import getMultiplicators as getMul
import numpy as np
import math as mt
import sys
import mpmath as mp


def up(n=1):
    # My terminal breaks if we don't flush after the escape-code
    for i in range(n):
        sys.stdout.write('\x1b[1A')
    sys.stdout.flush()
def down(n=1):
    # I could use '\x1b[1B' here, but newline is faster and easier
    for i in range(n):
        sys.stdout.write('\n')
    sys.stdout.flush()
def inCircle(eig_s): # Смотрим переход через -1 
    for e_i in eig_s: 
        if np.linalg.norm(e_i)>1 and e_i.real < 0:
            return -1
    return 1
def createPhi_s(X,L,G):
        rside = createRsideOnePendulumsEquationMPMATH(L,G)
        q = mp.odefun(rside,0,mp.matrix([0,X[0]]),tol=0.001,degree=4,method="taylor")
        return lambda t:q(t)[0]
def createRsideOnePendulumsEquationMPMATH(L,G):
    return lambda t,q: [ q[1] , G-mp.sin(q[0]) -L*q[1] ] 
def createMatrix(n):
    x = np.zeros((n,n))
    for i in range(n):
        x[i][i] = -2
    for i in range(n-1):
        x[i][i+1] = 1
        x[i+1][i] = 1
    return x
def binarySearch(arr_range,isRight,phi_s,t_period,L,G,N):
    EPS = 1e-3
    while (arr_range[1] - arr_range[0])>EPS:
        if inCircle(getMul(phi_s,t_period,L,(arr_range[0]+arr_range[1])/2,G,N))==1:
            arr_range[isRight] = (arr_range[0]+arr_range[1])/2
        else:
            arr_range[1-isRight] = (arr_range[0]+arr_range[1])/2
    return (arr_range[1] + arr_range[0])/2
def getRange(eigenValues,g,l):
    Ls = (1/eigenValues)*(1/4)*((g**2)/(l**2) - 2*mt.sqrt(1-g**2) + (1/2)*(l**2)/(g**2))
    Rs = (1/eigenValues)*(1/4)*((g**2)/(l**2) + 2*mt.sqrt(1-g**2) + (1/2)*(l**2)/(g**2))
    res = np.array([Ls,Rs])
    return res
def getStep(k_i,K_s,K_e,eiv_1,eiv_2,G,L):
    dola_k = (k_i-K_s)/(K_e-K_s)
    lambda_i = eiv_2*(1-dola_k)+ eiv_1*dola_k
    res = getRange(lambda_i,G,L)
    return res[1]-res[0]

######################
##      VARIABLES
######################
# L = 0.7
# G = 0.97
# N = 6
# n_thread = 1
# #####################################
# ##              MAIN
# #####################################
# res = calculateLineData(0.86,G,N,n_thread)
# print(res)
