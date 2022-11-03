import csv
# f(x)= w1x1+w2x2+b
import numpy as np
import matplotlib.pyplot as plt
def init(n=5):
    beta = np.zeros((n, 1))
    return beta

def sigmoid(x, beta):
    return 1 / (1 + np.exp(-np.dot(beta.T, x.T)))

# def grad(x, y, p, beta,step=0.05):
#     ans = 0
#     for i in range(len(x)):
#         ans += step*(sigmoid(x[i], beta) - y[i]) * x.item(i, p)
#     return ans / len(x)
#
# def train(x, y, beta, epochs=1000):
#     step = 1
#     for i in range(epochs):
#         g = []
#         for k in range(len(beta)):
#             g.append(grad(x, y, k, beta,0.1))
#         for k in range(len(beta)):
#             beta[k] = beta[k] - step * g[k]
def p1(x,beta):
    #print(beta)
    return np.exp(np.dot(x,beta))/(1+np.exp(np.dot(x,beta)))
def grad(x,y,beta):
    #partial1是l(b)对b的偏导，partial2是l(b)对b的二阶偏导
    partial1,partial2=0,0
    for i in range(len(x)):
        if i==0:
            partial1= -x[i]*(y[i]-p1(x[i],beta)).item(0,0)
            partial2= np.dot(x[i].T,x[i])*p1(x[i],beta).item(0,0)*(1-p1(x[i],beta)).item(0,0)
        else:
            partial1 += -x[i]*(y[i]-p1(x[i],beta)).item(0,0)
            partial2 += np.dot(x[i].T,x[i])*p1(x[i],beta).item(0,0)*(1-p1(x[i],beta)).item(0,0)
    return partial1,partial2
def train(x,y,beta,epochs=10):
    for i in range(epochs):
        g1,g2=grad(x,y,beta)
        beta=beta- np.dot(np.linalg.inv(g2),g1.T)
        #print(beta)
    return beta
#0.05 1000
def evaluate(x,mp,lst):
    cnt,tot=0,0
    for i in range(len(x)):
        best,id=-100000,"0"
        for key in mp.keys():
            beta=mp.get(key)
            v=sigmoid(x[i],beta)
            if v>best:
                best,id=v,key
        if(id==lst[i][5]):
            cnt+=1
        #print(id,lst[i][5])
        tot+=1
    print("accuracy:",cnt/tot)
def solve(s,lst):
    mp={}
    x=[]
    for i in lst:
        tmp=i[1:5];tmp.append(1)
        x.append(tmp)
    for i in s:
        print(i)
        beta= init()
        print(beta)
        y=[]
        for j in range(len(lst)):
            # print(lst[j][5],i)
            if(lst[j][5]==i):y.append(1)
            else:y.append(0)
        #print(y)
        x=np.matrix(x)
        beta=train(x,y,beta)
        print(beta)
        mp[i]=beta
    evaluate(x,mp,lst)

with open("E:/desktop/IRIS/iris.csv","r") as file:
    reader= csv.reader(file)
    lst=list(reader)
    lst.pop(0)
for i in lst:
    for j in range(1,5):
        i[j]=float(i[j])
#for i in lst:print(i)
s=set()
for i in lst:s.add(i[5])
solve(s,lst)