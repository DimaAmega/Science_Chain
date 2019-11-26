import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np

t = np.arange(0,11.3354,0.001)
q = odeint(lambda q,t:np.array([q[1],0.97 - np.sin(q[0]) - 0.94*q[1]]),[0,1.5721],t,hmax=0.001)
# plt.plot(t,q[:,1])
plt.plot(t,q[:,0])
plt.show()
