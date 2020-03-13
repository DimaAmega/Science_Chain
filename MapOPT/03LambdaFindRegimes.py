#######################################################################################################################################################################################################################################
## Lambda 03 - Sinphase X_start = np.array([3,3.5346,-0.2,3.5346,-0.4,3.5346,-0.1,3.5346,-0.1,3.5346,-0.21,3.5346])
## Lambda 03 - 4pi, Ks - 1.33, X_start = np.array([4.01739166e+00, 4.00489756e+00, 6.83138901e-01, 2.81285526e+00,6.83138901e-01, 2.81285526e+00, 0, 4.00489756e+00,0, 4.00489756e+00, 6.83138901e-01, 2.81285526e+00])
#######################################################################################################################################################################################################################################
from numpy.linalg import eig
from lib import  VF_find_Regime,newtonMethod,SinphaseParametrSinhronization,getStateSpecial,getMonodrommyMatrix,inCircle
import numpy as np
import math as mt


##################
##
##################
L = 0.3
G = 0.97
N = 6
K_s = 1.33
K_e = 3
Period = 4
args_sis = (N,L,G,K_s,)
args_for_search = (Period,args_sis,)


X_start = np.array([4.01739166e+00, 4.00489756e+00, 6.83138901e-01, 2.81285526e+00,
6.83138901e-01, 2.81285526e+00, 0, 4.00489756e+00,
0, 4.00489756e+00, 6.83138901e-01, 2.81285526e+00])

X = newtonMethod(VF_find_Regime,X_start,args_for_search)

for K_i in np.linspace(K_s,K_e,mt.floor((K_e - K_s)*150)):
    args_sis = (N,L,G,K_i,)
    args_for_search = (Period,args_sis,)
    X = newtonMethod(VF_find_Regime,X,args_for_search)
    M = getMonodrommyMatrix(X,args_sis)
    EIGVAL, _ = eig(M)
    print("**************************************************\n*** K_i {}\n**************************************************".format(K_i))
    print("Sinchronization ",SinphaseParametrSinhronization(X,args_sis))
    print("State ",getStateSpecial(X))
    print("EIGVAL")
    print(EIGVAL)
    print("FIND\n",X)
    print("IN CIRCLE ",inCircle(EIGVAL))
