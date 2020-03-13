from scipy.integrate import odeint
import numpy as np
import math as mt
import mpmath as mp
########################
##ТОЧНОСТЬ
########################
mp.mp.dps = 15
mp.mp.pretty = True
#######################
##    GLOBAL VAR
#######################
t_end = 500
h_iter = 0.001
deg = 4
t = np.arange(0,t_end,h_iter)
eps = 1e-14
#####################
##    FUNCTIONS
#####################
def getYacobyMatrix(VF,X):
    h = 0.001
    N = len(X)
    res = mp.matrix(N,N)
    for i in range(N):
        X_1 = X.copy()
        X_2 = X.copy()
        X_1[i]-=h
        X_2[i]+=h
        der = (VF(X_2) - VF(X_1))/(2*h)
        res[:,i] = der
    return res
     
def newtonMethod(VF,X_start):
    X = X_start
    Y = getYacobyMatrix(VF,X)
    Y_1 = Y**-1
    while mp.norm(Y_1*VF(X),2) > eps:
        Y = getYacobyMatrix(VF,X)
        Y_1 = Y**-1
        X = X-(Y_1*VF(X))
    return X
def createVectorFunction(f1,f2):
	return lambda Xi : mp.matrix([f1(Xi),f2(Xi)])
def getIndexOfTime(time):
    return mt.floor(time/h_iter)
def createRsideOnePendulumsEquation(L,G):
    return lambda q,t: np.array( [ q[1] , G-np.sin(q[0]) -L*q[1] ] )
def createRsideOnePendulumsEquationMPMATH(L,G):
    return lambda t,q: mp.matrix([ q[1] , G-mp.sin(q[0]) -L*q[1] ] )
def getSecDerOfRS(Rs):
    return lambda q:Rs(q,None)[1] 
def findFirstMaximum(q,RS,start_index):
    end_index = getIndexOfTime(t_end)
    secDer = getSecDerOfRS(RS)
    prev_sign = secDer(q[start_index-1])
    next_sign = secDer(q[start_index])   #берём начальную частоту
    i = start_index
    while i < end_index:
        if prev_sign > 0 and next_sign < 0:
            return (i,q[i])
        else:
            prev_sign = next_sign
            next_sign = secDer(q[i])
            i+=1
####################################
## IMPORT FUNCTION
####################################
def getApproximateX0(L,G):
    RS = createRsideOnePendulumsEquation(L,G)
    q = odeint(RS,[0,6],t,hmax=h_iter)
    ind_first_max , q_1 = findFirstMaximum(q,RS,getIndexOfTime(t_end*0.5))
    ind_second_max , q_2 = findFirstMaximum(q,RS,ind_first_max)
    period_time = (ind_second_max - ind_first_max)*h_iter
    maximum = (q_2[1] + q_1[1])/2
    return mp.matrix([maximum,period_time])
def getInitCondition(X0_approx,L,G):
    def f1(Xi): # производная по фи 
        q0 = [0,Xi[0]]  # phase and speed
        rside = createRsideOnePendulumsEquationMPMATH(L,G)
        q = mp.odefun(rside,0,q0,tol=h_iter,degree=deg,method="taylor")
        return q(Xi[1])[0] -2*np.pi
    def f2(Xi): # производная по фи 
        q0 = [0,Xi[0]]  # phase and speed
        rside = createRsideOnePendulumsEquationMPMATH(L,G)
        q = mp.odefun(rside,0,q0,tol=h_iter,degree=deg,method="taylor")
        return q(Xi[1])[1] -Xi[0]
    VF = createVectorFunction(f1,f2)
    X = newtonMethod(VF,X0_approx)
    return X


