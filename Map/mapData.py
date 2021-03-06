from multiprocessing import Pool
import multiprocessing as mpproc
from multiprocessing import Lock
import numpy as np
import sys
import pickle
from LineDataKRangeTools import *
from colorama import init
init()


def printMessage(n_thread,message):
    lock.acquire()
    down(n_thread)
    sys.stdout.write("\r \033[K")
    sys.stdout.write(message)
    up(n_thread)
    lock.release()

def calculateLineData(t_period,y_point,L,G,N,n_thread):
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
    Right_Board = 40
    if K_end < Right_Board:
        Right_Board = K_end
    # print("Запустили поток L - {} Range K [{},{}]".format(L,K_start,K_end))
    l1 = EVV[-1]
    l2 = EVV[0]
    ######################
    ##      СЧЁТ
    ######################
    phi_s = createPhi_s(y_point,L,G)
    k_i = K_start
    Prev = inCircle(getMul(phi_s,t_period,L,k_i,G,N))
    Netx = Prev
    step = getStep(k_i,K_start,K_end,l1,l2,G,L) 
    k_i += step
    while k_i < Right_Board:
        multiplicators = getMul(phi_s,t_period,L,k_i,G,N)
        Next = inCircle(multiplicators)
        if Next*Prev<0:
            if Prev>0:
                res = binarySearch([k_i-step,k_i],0,phi_s,t_period,L,G,N)
                DATA.append({"type":"Lboard","point":res})
                printMessage(n_thread,"thred - {}; L - {}; Lboard {}".format(n_thread,round(L,3),res))
            else:
                res = binarySearch([k_i-step,k_i],1,phi_s,t_period,L,G,N)
                DATA.append({"type":"Rboard","point":res})
                printMessage(n_thread,"thred - {}; L - {}; Rboard {}".format(n_thread,round(L,3),res))

        else:
            DATA.append({"type":"point","point":k_i,"multiplicators":multiplicators})
            if  Count_iterations % N_iterations == 0:
                printMessage(n_thread,"thred - {}; L - {}; Progress - {}/100 |".format(n_thread,round(L,3),mt.floor((k_i - K_start)/(Right_Board - K_start)*100)))
            else:
                printMessage(n_thread,"thred - {}; L - {}; Point - {}; Range [{},{}]".format(n_thread,round(L,3),k_i,K_start,Right_Board))
        Prev = Next
        step = getStep(k_i,K_start,K_end,l1,l2,G,L)
        k_i += step
        Count_iterations+=1
    printMessage(n_thread,"thred - {}; L - {}; Progress - 100/100".format(n_thread,round(L,3)))
    return {"Lambda":L,"Range_K":[K_start,K_end],"data":DATA}    

#######################################
###
#######################################

def initMultiprocessing(l):
    global lock
    lock = l

if __name__ == '__main__':
    # N_CPU = mpproc.cpu_count()
    N_CPU = 4
    data = []
    tasks = []
    l = Lock()
    num_proc = 1
    G = 0.97
    N = 6
    L_s = 0.91
    L_e = 0.97
    h_L = 0.005
    L_arr = np.arange(L_s,L_e,h_L)
    # print("CHECKING")
    # X = IC(getApproxX0(L_s,G),L_s,G)
    # for L_i in L_arr:
    #     X = IC(X,L_i,G)
    #     print("L -",L_i,"\n Point \n",X)
    print("START PROCESSES")
    with Pool(processes=N_CPU,initializer=initMultiprocessing, initargs=(l,)) as pool:
        X = IC(getApproxX0(L_s,G),L_s,G)
        for L_i in L_arr:
            X = IC(X,L_i,G)
            t_p,y_point = parseX(X)
            tasks.append(pool.apply_async(calculateLineData,args = (t_p,y_point,L_i,G,N,num_proc),error_callback = lambda e: print("ERRROR!",e)))
            num_proc+=1
        for task in tasks:
            task.wait()
            data.append(task.get())
        pool.close()
        pool.join()
        down(num_proc)
        with open('Count-{} {} {}.pickle'.format(N,L_s,L_e), 'wb') as f:
            pickle.dump(data, f)
        
