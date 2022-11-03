#f(x)= w1x1+w2x2+b
import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x,beta):
    return 1/(1+np.exp(-np.dot(x,beta)))
def p1(x,beta):
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
def train(x,y,beta,epochs):
    step=0.01
    for i in range(epochs):
        g1,g2=grad(x,y,beta)
        beta=beta- np.dot(np.linalg.inv(g2),g1.T)
        #print(beta)
    return beta
def evaluate(x,y,beta):
    pre_y,cnt1,cnt2=[],0,0
    for i in range(len(y)):
        v= sigmoid(x[i],beta)
        if v>=0.5:
            pre_y.append(1)
        else:
            pre_y.append(0)
    for i in range(len(pre_y) ):
        if pre_y[i]==y[i]:
            cnt1+=1
        cnt2+=1
    print('accuracy:',cnt1/cnt2)
def plot_point(x,y,beta):
    plt.xlabel('Density')
    plt.ylabel('Sugar content')
    label=["line","good","bad"]
    g1,g2,b1,b2=[],[],[],[]
    for i in range(len(x)):
        if y[i]==0:
            b1.append(x.item(i,0))
            b2.append(x.item(i,1))
        else:
            g1.append(x.item(i,0))
            g2.append(x.item(i,1))
    plt.scatter(g1,g2,c='r')
    plt.scatter(b1,b2,c='b')
    x1=np.arange(0,1,0.0001)
    y1=(x1*beta.item(0)+beta.item(2))/(-beta.item(1))
    plt.plot(x1,y1)
    #plt.legend(label,loc=0,ncol=2)
    plt.legend(label)
    plt.show()
x=[
[0.697,0.460,1],[0.774,0.376,1],[0.634,0.264,1],[0.608,0.318,1]
,[0.556,0.215,1],[0.403,0.237,1],[0.481,0.149,1],[0.439,0.211,1]
,[0.666,0.091,1],[0.243,0.267,1],[0.245,0.057,1],[0.343,0.099,1]
,[0.639,0.161,1],[0.657,0.198,1],[0.360,0.370,1],[0.593,0.042,1]
,[0.719,0.103,1]
]
x=np.matrix(x)
y=[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]
beta=np.zeros((3,1))
beta=train(x,y,beta,6)
print('y=',beta.item(0),'x1+',beta.item(1),'x2',beta.item(2),sep='')
evaluate(x,y,beta)
plot_point(x,y,beta)