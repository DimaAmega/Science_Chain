from multiprocessing import Pool
import multiprocessing as mpproc
from multiprocessing import Lock
import numpy as np
import sys
import pickle
from LineDataKRangeTools import *
from colorama import init
init()

def calculateLineData(L,G,N,n_thread):
    DATA = []
    Count_iterations = 0
    N_iterations = 1
    ######################
    ##      ОЦЕНКИ
    ######################
    x = createMatrix(N-1)
    EIG = np.linalg.eig(x)
    eigenValues = -1*EIG[0] # делаем все сч > 0
    EVV = np.array(sorted(eigenValues,reverse=True))
    Data = getRange(EVV,G,L)
    h_k = 1.5
    K_start = Data[0][0] - h_k
    K_end = Data[-1][-1] + h_k
    # print("Запустили поток L - {} Range K [{},{}]".format(L,K_start,K_end))
    l1 = EVV[-1]
    l2 = EVV[0]
    ######################
    ##      СЧЁТ
    ######################
    X = IC(L,G)

    phi_s = createPhi_s(X,L,G)
    t_period =float(X[1])
    k_i = K_start
    Prev = inCircle(getMul(phi_s,t_period,L,k_i,G,N))
    Netx = Prev
    step = getStep(k_i,K_start,K_end,l1,l2,G,L) 
    k_i += step
    while k_i < K_end:
        multiplicators = getMul(phi_s,t_period,L,k_i,G,N)
        Next = inCircle(multiplicators)
        if Next*Prev<0:
            if Prev>0:
                res = binarySearch([k_i-step,k_i],0,phi_s,t_period,L,G,N)
                DATA.append({"type":"Lboard","point":res})
            else:
                res = binarySearch([k_i-step,k_i],1,phi_s,t_period,L,G,N)
                DATA.append({"type":"Rboard","point":res})
        else:
            DATA.append({"type":"point","point":k_i,"multiplicators":multiplicators})
            if  Count_iterations % N_iterations == 0:
                lock.acquire()
                down(n_thread)
                sys.stdout.write("thred - {}; L - {}; Progress - {}/100".format(n_thread,round(L,3),mt.floor((k_i - K_start)/(K_end - K_start)*100)))
                up(n_thread)
                lock.release()
        Prev = Next
        step = getStep(k_i,K_start,K_end,l1,l2,G,L)
        k_i += step
        Count_iterations+=1
    lock.acquire()
    down(n_thread)
    sys.stdout.write("thred - {}; L - {}; Progress - 100/100".format(n_thread,L,mt.floor((k_i - K_start)/(K_end - K_start)*100)))
    up(n_thread)
    lock.release()
    return {"Lambda":L,"Range_K":[K_start,K_end],"data":DATA}    

#######################################
###
#######################################
def init(l):
    global lock
    lock = l

if __name__ == '__main__':
    N_CPU = mpproc.cpu_count()
    # N_CPU = 4
    print("NCPU",N_CPU)
    data = []
    tasks = []
    l = Lock()
    print("START PROCESSES")
    with Pool(processes=N_CPU,initializer=init, initargs=(l,)) as pool:
        num_proc = 1
        G = 0.97
        N = 7
        L_s = 0.1
        L_e = 0.8
        h_L = 0.01
        L_arr = np.arange(L_s,L_e,h_L)
        for L_i in L_arr:
            tasks.append(pool.apply_async(calculateLineData,args = (L_i,G,N,num_proc),error_callback = lambda e: print(e)))
            num_proc+=1
        for task in tasks:
            task.wait()
            data.append(task.get())
        pool.close()
        pool.join()
        down(num_proc)
        with open('Count-{}.pickle'.format(N), 'wb') as f:
            pickle.dump(data, f)
        
