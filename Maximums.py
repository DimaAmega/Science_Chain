
from scipy.integrate import odeint
import numpy as np
from multiprocessing import cpu_count
from multiprocessing import Lock
import sys
import pickle
import math as mt
import random as rndm
from multiprocessing import Pool
from colorama import init
init()

def init(l):
    global lock
    lock = l
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
def findMax(q,eps=1e-4):
    res = []
    start = q[0]
    i = 1
    N = len(q)
    while i < N-1:
        if q[i] > q[i-1] and q[i] > q[i+1]:
            p = q[i]
            flag = 0
            for r_i in res:
                if mt.fabs(p - r_i)<eps:
                    flag = 1
                    break
            if flag == 0:
                res.append(p)
        i+=1
    return res
def CountMaximums(N,L,G,K_i,t_end=3500,proc=0.95,h=1e-3):
    res = []
    s_t_index = round(t_end*proc/h)
    q_0 = createQ0(N)
    t = np.arange(0,t_end,h)
    q = odeint(CreateRS, q_0,t,args=(N,L,G,K_i),hmax=h)
    for i in range(N):
        res.append(findMax(q[s_t_index:-1][:,2*i+1]))
    return {"K":K_i,"max":res}
###################
##   VARIABLES
###################
# N,L,G,K_i = 6,0.3,0.97,0.74
# CountMaximums(N,L,G,K_i,t_end=200,proc=0.9,h=1e-3)


if __name__ == '__main__':
    N_CPU = cpu_count()
    data = []
    tasks = []
    l = Lock()
    K_s = 0.731
    K_e = 0.745
    h_K = 0.001/3
    N,L,G = 6,0.3,0.97
    K_arr = np.arange(K_s,K_e,h_K)
    print("FIND MAXIMUMS L - ", L)
    with Pool(processes=N_CPU,initializer=init, initargs=(l,)) as pool:
        num_proc = 1
        for K_i in K_arr:
            tasks.append(pool.apply_async(CountMaximums,args = (N,L,G,K_i),error_callback = lambda e: print(e)))
            num_proc+=1
        for task in tasks:
            task.wait()
            res = task.get()
            data.append(res)
            sys.stdout.write("\r \033[K")
            sys.stdout.write("Progress {} of 100".format(round((res["K"]-K_s)/(K_e -K_s)*100,3)))
            sys.stdout.flush()
        pool.close()
        pool.join()
        with open('Data-{}.pickle'.format(L), 'wb') as f:
            pickle.dump(data, f)