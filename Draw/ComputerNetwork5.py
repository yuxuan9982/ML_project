import math
import random

import matplotlib.pyplot as plt
import numpy

plt.rcParams['font.sans-serif']=['SimHei']
plt.xlabel("n")
plt.ylabel("cwnd")
x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
y=[1,2,4,8,16,32,33,34,35,36,37,38,39,40,41,42,21,22,23,24,25,26,1,2,4,8]
plt.plot(x,y)
plt.scatter(x,y,s=5)
plt.show()