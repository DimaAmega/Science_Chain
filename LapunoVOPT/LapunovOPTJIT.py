from scipy.integrate import odeint
import numpy as np
from multiprocessing import cpu_count
from multiprocessing import Lock
import sys
import pickle
from numba import njit
from numba import vectorize
import math as mt
import random as rndm
from multiprocessing import Pool
from colorama import init
import time
init()

def printMessage(n_thread,message):
    lock.acquire()
    down(n_thread)
    sys.stdout.write("\r \033[K")
    sys.stdout.write(message)
    up(n_thread)
    lock.release()
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
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out
def init2(l):
    global lock
    lock = l

@njit
def getYacobyMatrixExplicit(X,N,L,G,K):
    res = np.zeros((2*N,2*N))
    res[0][1] = 1

    res[1][0] = -( mt.cos(X[0]) + K*mt.cos(X[2] - X[0]) )
    res[1][1] = -L
    res[1][2] = K*mt.cos(X[2] - X[0])
    n = 0
    while n < 2*N-4: 
        res[n+2][n+3]  = 1

        res[n+3][n] = K*mt.cos(X[n] - X[n+2])
        res[n+3][n+2] = -( mt.cos(X[n+2]) + K*mt.cos(X[n+4] - X[n+2]) + K*mt.cos(X[n] - X[n+2]) )
        res[n+3][n+3] = -L
        res[n+3][n+4] = K*mt.cos(X[n+4] - X[n+2])
        n+=2
    res[2*N-2][2*N-1] = 1

    res[2*N-1][2*N-4] = K*mt.cos(X[2*N-4] - X[2*N-2])
    res[2*N-1][2*N-2] = -( mt.cos(X[2*N-2]) + K*mt.cos(X[2*N-4] - X[2*N-2]) )
    res[2*N-1][2*N-1] = -L
    return res

@njit
def CreateRS(q,t,N,L,G,K):
    X = np.zeros(2*N)
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

def createQ0(N):
    q0 = np.zeros(2*N)
    for i in range(2*N):
        q0[i] = 5 + 3*rndm.random()
    return q0

def calcLine(N,L,G,K_arr,num_proc):
    res = []
    rndm.seed(5)    
    start_point = createQ0(N)
    for K_i in K_arr:
        args = (N,L,G,K_i)
        l_exp_data = getLapExp(CreateRS,RSLapunovExp,start_point,args)
        res.append({"Lexp":l_exp_data,"K_i":K_i})
        iterations+=1
        printMessage(num_proc,"Progress of {} Thread, Lexp = {}, Ki - {}  {} of 100".format(num_proc,round(l_exp_data,5),round(K_i,3),round((K_i-K_arr[0])/(K_arr[-1] -K_arr[0])*100,2)))
    return res

@njit
def RSLapunovExp(q,t,VF,N,L,G,K):
    res = np.empty((4*N))
    linear =  np.dot(getYacobyMatrixExplicit(q[:2*N],N,L,G,K),q[2*N:])
    RS = VF(q[:2*N],t,N,L,G,K)
    for i in range(2*N):
        res[i] = RS[i]
        res[i+2*N] = linear[i]
    return res

@vectorize
def vectorisedX(X,k1,k2,k3,k4,h):
    return X + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

@njit
def getLapExp(RS,RSLapExp,q0,args,t_scip = 4000,t_calc = 4000,h=0.015):
    X = q0
    N = mt.floor(len(q0)/2)
    orta = 1/mt.sqrt(2*N)
    t_end = t_scip + t_calc

    for t_i in np.arange(0,t_scip,h):
        k1 = RS(X,0,*args)
        k2 = RS(X+h*k1/2,t_i+h/2,*args)
        k3 = RS(X+h*k2/2,t_i+h/2,*args)
        k4 = RS(X+h*k3,t_i+h,*args)
        X = vectorisedX(X,k1,k2,k3,k4,h)

    pretubrations = np.full(2*N,orta)

    X = np.concatenate((X,pretubrations))

    for t_i in np.arange(t_scip,t_end,h):
        k1 = RSLapExp(X,t_i,RS,*args)
        k2 = RSLapExp(X+h*k1/2,t_i+h/2,RS,*args)
        k3 = RSLapExp(X+h*k2/2,t_i+h/2,RS,*args)
        k4 = RSLapExp(X+h*k3,t_i+h,RS,*args)
        X = vectorisedX(X,k1,k2,k3,k4,h)

    return mt.log(np.linalg.norm(X[2*N:]))/(t_calc)


###################
##     MAIN
###################

if __name__ == '__main__':
    N_CPU = 4 #cpu_count()
    data = []
    tasks = []
    l = Lock()
    K_s = 0 # 0
    K_e = 2.5  # 2.5
    h_K = 0.005
    N = 6     
    L = 0.8 #0.8
    G = 0.97
    K_arr = np.arange(K_s,K_e,h_K)
    Multi_K_arr = chunkIt(K_arr,N_CPU)
    with Pool(processes=N_CPU,initializer=init2, initargs=(l,)) as pool:
        num_proc = 1
        for K_i_arr in Multi_K_arr:
            tasks.append(pool.apply_async(  ,args = (N,L,G,K_i_arr,num_proc),error_callback = lambda e: print(e)))
            num_proc+=1
        for task in tasks:
            task.wait()
            res = task.get()
            data+=res
        pool.close()
        pool.join()
        with open('Lapunov08.pickle', 'wb') as f:
            pickle.dump(data, f)