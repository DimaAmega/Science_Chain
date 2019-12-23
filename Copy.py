from scipy.integrate import odeint
import numpy as np
from multiprocessing import cpu_count
from multiprocessing import Lock
import sys
import pickle
from numba import jit
import math as mt
import random as rndm
from multiprocessing import Pool
from colorama import init
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


def scipTransitionProcess(N,L,G,K_i,q0,t_end,h):
    t = np.arange(0,t_end,h)
    q = odeint(CreateRS,q0,t,args=(N,L,G,K_i),hmax=h)
    q_last = q[-1]
    for i in range(N):
        q_last[2*i] = q_last[2*i]%2*mt.pi
    return q_last

@jit
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
        q0[i] = 10 + 3*rndm.random()
    return q0


def getYacobyMatrix(VF,X):
    h = 0.001
    N = len(X)
    res = np.zeros((N,N))
    for i in range(N):
        X_1 = X.copy()
        X_2 = X.copy()
        X_1[i]-=h
        X_2[i]+=h
        der = (VF(X_2) - VF(X_1))/(2*h)
        res[:,i] = der
    return res


def createVectorFunction(N,L,G,K):
    return lambda Xi: CreateRS(Xi,0,N,L,G,K)


def calcLine(N,L,G,K_arr,num_proc,t_end=1000,h=1e-3):
    res = []
    q0 = createQ0(N)
    for K_i in K_arr:
        q0 = scipTransitionProcess(N,L,G,K_i,q0,t_end,h)
        l_exp_data = getLapExp(N,L,G,K_i,q0)
        res.append(l_exp_data)
        printMessage(num_proc,"Progress of {} Thread, Lexp = {}, {} of 100".format(num_proc,str(res[-1]),round((K_i-K_arr[0])/(K_arr[-1] -K_arr[0])*100,2)))
    return res

# class timeControl:
#     def __init__ (self,t_middle):
#         self.t_middle = t_middle
#         self.flag = False
#     def compare (self,t_i):
#         if self.flag == False and t_i > self.t_middle:
#             self.flag = True
#             return True
#         return False 


def RSLapunovExp(q,t,N,VF,t_pretubr):
    RS = VF(q[:2*N])
    linear = getYacobyMatrix(VF,q[:2*N]).dot(q[2*N:])
    if mt.fabs(t - t_pretubr) < 0.5*1e-3:
        linear = 200*np.full(2*N,1/mt.sqrt(2*N))
    res = np.concatenate([RS,linear])
    return res

def getLapExp(N,L,G,K_i,X0,t_end=1000,t_empty = 500,h_iter=1e-3):
    t = np.arange(0,t_end + t_empty,h_iter)
    q0 = np.concatenate([X0,np.full(2*N,0)])
    VF = createVectorFunction(N,L,G,K_i)
    sol = odeint(RSLapunovExp,q0,t,args=(N,VF,t_empty),hmax=h_iter)
    return {"Lexp":mt.log(np.linalg.norm(sol[-1][2*N:])/np.linalg.norm(sol[round(1000*t_empty)][2*N:]))/(t_end),"Ki":K_i}

###################
##     MAIN
###################
K_i = 0.73 #1.5 -reg #0.73 - ch 
N = 6
L = 0.8
G = 0.97
h = 1e-3
rndm.seed(5)
X0 = scipTransitionProcess(N,L,G,K_i,createQ0(N),100,h)
res = getLapExp(N,L,G,K_i,X0)
print(res)

# if __name__ == '__main__':
#     N_CPU = 4 #cpu_count()
#     data = []
#     tasks = []
#     l = Lock()
#     K_s = 0.575
#     K_e = 0.842
#     h_K = 0.005
#     N = 6
#     L = 0.8
#     G = 0.97
#     K_arr = np.arange(K_s,K_e,h_K)
#     Multi_K_arr = chunkIt(K_arr,N_CPU)
#     print("Calk LapunovEXP, L - ", L)
#     with Pool(processes=N_CPU,initializer=init2, initargs=(l,)) as pool:
#         num_proc = 1
#         for K_i_arr in Multi_K_arr:
#             tasks.append(pool.apply_async(calcLine,args = (N,L,G,K_i_arr,num_proc),error_callback = lambda e: print(e)))
#             num_proc+=1
#         for task in tasks:
#             task.wait()
#             res = task.get()
#             data+=res
#         pool.close()
#         pool.join()
#         with open('Lapunov08.pickle', 'wb') as f:
#             pickle.dump(data, f)
