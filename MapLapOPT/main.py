import numpy as np
import math as mt
from multiprocessing import cpu_count,Pool,Lock
import sys,pickle
from numba import njit
import random as rndm
from colorama import init
sys.path.append("../LapunoVOPT/")
from LapunovOPTJIT import getLapExp,RSLapunovExp
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
def createQ0(N):
    q0 = np.zeros(2*N)
    for i in range(2*N):
        q0[i] = 10 + 3*rndm.random()
    return q0

def createPretubrations(N):
    q0 = np.zeros(2*N)
    for i in range(2*N):
        q0[i] = 5*1e-2 + 5*1e-2*rndm.random()
    return q0

@njit
def scipTime(RS,h,q0,t_scip,args):
    X = q0
    for t_i in np.arange(0,t_scip,h):
        k1 = RS(X,t_i,*args)
        k2 = RS(X+h*k1/2,t_i+h/2,*args)
        k3 = RS(X+h*k2/2,t_i+h/2,*args)
        k4 = RS(X+h*k3,t_i+h,*args)
        X += h*(k1 + 2*k2 + 2*k3 + k4)/6
    return X

@njit
def modXPhase2Pi(X):
    for i in range(mt.floor(len(X)/2)):
        X[2*i] = X[2*i]%(2*mt.pi)
    return X

def getProgress(K_i,K_s,K_e):
    return round((K_i-K_s)/(K_e -K_s)*100,2)

def getState(X_last):
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

def calcLine(N,L_i,G,K_arr,num_proc,t_scip=2000,h=0.015):
    scip_iterations_to_Print = 1
    iterations = 0
    rndm.seed(2)
    res = {"L":L_i,"Data":[]}
    X_last = scipTime(CreateRS,h,createQ0(N),4*t_scip,(N,L_i,G,K_arr[0]))

    for K_i in K_arr:
        args = (N,L_i,G,K_i)
        X_last = scipTime(CreateRS,h,modXPhase2Pi(X_last) + createPretubrations(N),t_scip,args)
        state_str = getState(modXPhase2Pi(X_last))
        Lexp = getLapExp(CreateRS,RSLapunovExp,modXPhase2Pi(X_last),args,t_scip = 1000,t_calc = 4000,h=0.015)
        res["Data"].append({"K":K_i,"state": state_str,"Lexp":Lexp,"X_last":modXPhase2Pi(X_last)})
        iterations+=1
        ####  PRINT SOME INFO  ####
        if iterations % scip_iterations_to_Print == 0:
            printMessage(num_proc,"{} Th {} of 100,L_i {}, ki {}, state {}, Lexp {}".format(num_proc,getProgress(K_i,K_arr[0],K_arr[-1]),round(L_i,3),round(K_i,3),state_str,round(Lexp,4)))
    return [res]

###################
##     MAIN
###################

if __name__ == '__main__':
    N_CPU = cpu_count()
    data = []
    tasks = []
    l = Lock()
    with open('dataN6.pickle', 'rb') as f:
        MapData = pickle.load(f)
    K_data = MapData["x"]
    L_arr = MapData["y"]

    N = 6
    G = 0.97
    h_k = 1e-3

    with Pool(processes=N_CPU,initializer=init2, initargs=(l,)) as pool:
        num_proc = 1
        print("COUNT OF THREDS WE NEED ",len(L_arr))
        for i in range(len(L_arr)):
            K_arr = np.arange(K_data[i][0],K_data[i][-1],h_k)
            tasks.append(pool.apply_async(calcLine,args = (N,L_arr[i],G,K_arr,num_proc),error_callback = lambda e: print(e)))
            num_proc+=1
        for task in tasks:
            task.wait()
            res = task.get()
            data+=res
        pool.close()
        pool.join()
        with open('LapMapDataN6.pickle', 'wb') as f:
            pickle.dump(data,f)