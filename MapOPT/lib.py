from numba import njit,jit
import numpy as np
import math as mt
from numpy.linalg import inv,norm


def getStateSpecial(X_last):
    X_last = X_last.copy()
    X_last[0] = 0
    eps = 1e-5
    short_res = []
    res = "|"
    N = mt.floor(len(X_last)/2)
    values = []
    for i in range(N):
        values.append((X_last[2*i],i+1,))
    values_np = np.array(values, dtype=[('phase', float), ('index', int),])
    sort_values = np.sort(values_np, kind = "mergesort", order='phase')
    i = 1
    while (i<N):
        chunk = []
        while( i < N and mt.fabs(sort_values[i][0] - sort_values[i - 1][0] ) < eps ):
            chunk.append("{}".format(sort_values[i-1][1]))
            i+=1
        chunk.append("{}".format(sort_values[i-1][1]))
        short_res.append(str(len(chunk)))
        chunk.sort(key = lambda s_i: int(s_i))
        res+="=".join(chunk)+"|"
        i+=1
    if (i==N): 
        res+= "{}|".format(sort_values[i-1][1])
        short_res.append("1")
    res+=" ~ " + ":".join(short_res)
    return res

@njit
def CreateRS(q,t,N,L,G,K):
    X = np.empty(2*N)
    X[0] = q[1]
    X[1] = -L*q[1] - mt.sin(q[0]) + G + K*( mt.sin(q[2] - q[0]) ) 
    n = 0
    while n < 2*N-4: 
        X[n+2] = q[n+3]
        X[n+3] = -L*q[n+3] - mt.sin(q[n+2]) + G + K*( mt.sin(q[n+4] - q[n+2]) + mt.sin(q[n] - q[n+2]) )
        n+=2
    X[2*N-2] = q[2*N-1]
    X[2*N-1] = -L*q[2*N-1] - mt.sin(q[2*N-2]) + G + K*( mt.sin(q[2*N-4] - q[2*N-2]) )
    return X

@njit
def RSlinear(q,t,N,L,G,K,DSomeRotMode_t):
    X = np.empty(2*N)
    X[0] = q[1]
    X[1] = -L*q[1] - mt.cos(DSomeRotMode_t[0])*q[0] + K*mt.cos(DSomeRotMode_t[1]-DSomeRotMode_t[0])*(q[2] - q[0]) 
    n = 0
    while n < 2*N-4:
        X[n+2] = q[n+3] 
        X[n+3] = -L*q[n+3] - mt.cos(DSomeRotMode_t[ int(n/2 + 1) ])*q[n+2] + K*mt.cos( DSomeRotMode_t[ int(n/2 + 2) ]-DSomeRotMode_t[ int(n/2+1) ] )*(q[n+4] - q[n+2])   + K*mt.cos( DSomeRotMode_t[ int(n/2) ]-DSomeRotMode_t[ int(n/2+1) ] )*(q[n] - q[n+2])
        n+=2
    X[2*N-2] = q[2*N-1]
    X[2*N-1] = -L*q[2*N-1] - mt.cos(DSomeRotMode_t[N-1])*q[2*N-2] + K*mt.cos( DSomeRotMode_t[ N-2 ] - DSomeRotMode_t[ N-1 ] )*( q[2*N-4] - q[2*N-2] )
    return X

@njit
def getYacobyMatrix(VF,X,args):
    h = 1e-2
    N = len(X)
    res = np.empty((N,N,))
    X_1 = X.copy()
    X_2 = X.copy()
    for i in range(N):
        X_1[i]-=h
        X_2[i]+=h
        der = (VF(X_2,*args) - VF(X_1,*args))/(2*h)
        res[:,i] = der
        X_1[i]+=h
        X_2[i]-=h
    return res

@njit
def RKMethod(RS,q0,t,h,args):
    X = q0
    for t_i in t:
        X += stepRK4(RS,X,t_i,h,args)
    return X

@njit
def stepRK4(RS,X,t_i,h,args):
    k1 = RS(X,t_i,*args)
    k2 = RS(X+h*k1/2,t_i+h/2,*args)
    k3 = RS(X+h*k2/2,t_i+h/2,*args)
    k4 = RS(X+h*k3,t_i+h,*args)
    return h*(k1 + 2*k2 + 2*k3 + k4)/6

@njit
def VF_find_Regime(X,Period,args):
    # X = [T,der0, ph1,der1, ph2,der2, ph3,der3,...,phN,derN]
    # T = X[0]
    N,L,G,K = args
    res = np.empty(2*N)
    T = X[0]
    q0 = X.copy()
    q0[0] = 0 # first phase eq zero
    t = np.linspace(0,T,mt.floor(T*500)) # IT THE KEY
    h = t[1] - t[0]
    X_last = RKMethod(CreateRS,q0,t,h,args)

    res[0] = (X_last[0] - Period*mt.pi)
    res[1] = (X_last[1] - X[1])
    n = 0
    while n <= 2*N-4: 
        res[n+2] = (X_last[n+2] - X[n+2] - Period*mt.pi)
        res[n+3] = (X_last[n+3] - X[n+3])
        n+=2
    return res

@njit
def mulMatrixOnVec(Mat,vec):
    N = len(vec)
    res = np.zeros(N)
    for i in range(N):
        res += Mat[:,i]*vec[i]
    return res

@njit
def newtonMethod(VF,X_start,args):
    eps = 1e-12
    X = X_start
    Y_1 = inv(getYacobyMatrix(VF,X,args))
    VF_X = VF(X,*args)
    d = mulMatrixOnVec(Y_1,VF_X)
    while norm(d) > eps:
        VF_X = VF(X,*args)
        Y_1 = inv(getYacobyMatrix(VF,X,args))
        d = mulMatrixOnVec(Y_1,VF_X)
        X = X-d
    return X

@njit 
def pushPhases(X,N):
    res = np.empty(N)
    for i in range(N):
        res[i] = X[2*i] 
    return res

@njit
def createEi(N,i):
    res = np.zeros(N)
    res[i] = 1
    return res

@njit
def inCircle(eig_s): # Смотрим переход через левую полуплоскость
    for e_i in eig_s: 
        if np.absolute(e_i)>1 and e_i.real < 0:
            return -1
    return 1

@njit
def getMonodrommyMatrix(InitCond,args):
    N = mt.floor(len(InitCond)/2)
    T = InitCond[0]
    q0 = InitCond.copy()
    q0[0] = 0 # first phase eq zero cause we can make translations on phase
    t = np.linspace(0,T,mt.floor(T*500)) # IT THE KEY
    h = t[1] - t[0]
    t2 = np.arange(0,T+h/4,h/2)
    h2 = t2[1] - t2[0]
    DSomeRotMode_t = np.empty((len(t2)+2,N))
    DSomeRotMode_t[0] = pushPhases(q0,N)
    X = q0
    i = 1
    for t_i in t2:
        X += stepRK4(CreateRS,X,t_i,h2,args)
        DSomeRotMode_t[i] = pushPhases(X,N)
        i+=1
    X += stepRK4(CreateRS,X,t_i+h2,h2,args)
    DSomeRotMode_t[i] = pushPhases(X,N)

    MatrixMonodrommy = np.empty((2*N,2*N))

    for j in range(2*N):
        X = createEi(2*N,j)
        i = 0
        for t_i in t[:-1]:
            k1 = RSlinear(X,t_i,*args,DSomeRotMode_t[i])
            k2 = RSlinear(X+h*k1/2,t_i+h/2,*args,DSomeRotMode_t[i+1])
            k3 = RSlinear(X+h*k2/2,t_i+h/2,*args,DSomeRotMode_t[i+1])
            k4 = RSlinear(X+h*k3,t_i+h,*args,DSomeRotMode_t[i+2])
            X += h*(k1 + 2*k2 + 2*k3 + k4)/6
            i+=2
        MatrixMonodrommy[j] = X
    return MatrixMonodrommy

@njit
def SinphaseParametrSinhronization(X,args):
    N = mt.floor(len(X)/2)
    T = X[0]
    X = X.copy()
    X[0] = 0 # first phase eq zero
    t = np.linspace(0,T,mt.floor(T*150)) # IT THE KEY
    h = t[1] - t[0]
    max_ksi = -100
    for t_i in t:
        X += stepRK4(CreateRS,X,t_i,h,args)
        ksi_i = 0
        for i in range(N):
            for j in range(i+1,N):
                ksi_i+= mt.fabs( X[2*i] - X[2*j])
        max_ksi = max(max_ksi,ksi_i)
    return 2*max_ksi/N/(N-1)