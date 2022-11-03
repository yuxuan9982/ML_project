# f(x)= w1x1+w2x2+b
import numpy as np


def init(n=4):
    beta = np.zeros((n, 1))
    return beta


def sigmoid(x, beta):
    return 1 / (1 + np.exp(-np.dot(beta.T, x.T)))


def grad(x, y, p, beta):
    ans = 0
    for i in range(len(x)):
        ans += (sigmoid(x[i], beta) - y[i]) * x.item(i, p)
    return ans / len(x)


def train(x, y, beta, epochs):
    step = 1
    for i in range(epochs):
        g = []
        for k in range(len(beta)):
            g.append(grad(x, y, k, beta))
        for k in range(len(beta)):
            beta[k] = beta[k] - step * g[k]


def printf(beta):
    for i in beta:
        print(i, sep='')

def evaluate(x,beta,y):
    cnt,tot=0,0
    for i in range(len(x)):
        pred= sigmoid(x,beta)
        if(pred>0.5):pred=1;
        else: pred=0;
        if pred==y[i]:
            cnt+=1
        tot+=1
    return cnt/tot





# x = [
#     [0.697, 0.460, 1], [0.774, 0.376, 1], [0.634, 0.264, 1], [0.608, 0.318, 1]
#     , [0.556, 0.215, 1], [0.403, 0.237, 1], [0.481, 0.149, 1], [0.439, 0.211, 1]
#     , [0.666, 0.091, 1], [0.243, 0.267, 1], [0.245, 0.057, 1], [0.343, 0.099, 1]
#     , [0.639, 0.161, 1], [0.657, 0.198, 1], [0.360, 0.370, 1], [0.593, 0.042, 1]
#     , [0.719, 0.103, 1]
# ]
# x = np.matrix(x)
# print(x.item(1,1))
# y = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#beta = init()
# print(beta)
# train(x, y, beta, 100)
# printf(beta)
# printf(evaluate(x,beta,y))



