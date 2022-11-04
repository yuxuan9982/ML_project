import csv
import numpy as np
import scipy
import numpy.random
import sklearn.model_selection
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
from sklearn import preprocessing
import xlrd
def sigmoid(x):
    return scipy.special.expit(x)
def par_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
class ANN:
    input_num,hidden_num,output_num,learning_rate,epochs=100,100,100,0.1,200
    y,beta,w,b,alpha,v,x=[],[],[],[],[],[],[]
    theta,gamma,hat_y,e=[],[],[],[]
    loss,wg=[],[]
    # def g(self,k):
    #     return self.hat_y[k]*(1-self.hat_y[k])*(self.y[k]-self.hat_y[k])

    def __init__(self,x,y,learning_rate,input_num,hidden_num,output_num,epochs):
        self.x=x #x should be (num,d)
        self.y=y #y should be (1,l)
        self.learning_rate=learning_rate
        self.input_num=input_num
        self.hidden_num=hidden_num
        self.output_num=output_num
        self.epochs=epochs
        self.b=np.zeros(hidden_num)
        #阈值的初始化
        self.theta=np.random.rand(output_num)
        self.gamma=np.random.rand(hidden_num)

        self.hat_y=np.zeros(output_num)
        #权重初始化
        self.v=np.zeros((input_num,hidden_num))
        self.w=np.zeros((hidden_num,output_num))
        self.e=np.zeros((1,hidden_num))
    def train_std(self):
        for _ in range(self.epochs):
            loss=[]
            for j in range(len(self.x)):
                x1,y1=self.x[j],self.y[j]
                #前向传播
                self.alpha= np.dot(x1,self.v)
                self.b= sigmoid(self.alpha-self.gamma)
                self.beta= np.dot(self.b,self.w)
                self.hat_y= sigmoid(self.beta-self.theta)
                ##
                #求得E均方误差
                E=0
                for i in range(len(y1)):
                    #print(self.hat_y[i]  )
                    E+=(self.hat_y[0,i]-y1[i])**2
                E/=2
                loss.append(E)
                #反向传播
                #g=np.multiply(self.hat_y,1-self.hat_y,self.y[j]-self.hat_y)
                g=np.zeros((1,self.output_num))
                for i in range(self.output_num):
                    g[0,i]= self.hat_y[0,i]*(1-self.hat_y[0,i])*(self.y[0,i]-self.hat_y[0,i])
                wg=np.dot(self.w,g.T)
                for k in range(self.hidden_num):
                    self.e[0,k]=self.b[0,k]*(1-self.b[0,k])*wg[k,0]

                self.w+=self.learning_rate*np.dot(self.b.T,g)

                for i in range(self.output_num):
                    self.theta[i]+=-self.learning_rate*g[0,i]
                self.v+=self.learning_rate*np.dot(self.x[j].T,self.e)
                for i in range(self.hidden_num):
                    self.gamma[i]+=-self.learning_rate*self.e[0,i]
            print('epochs-->',_)
            print('loss:',loss)
    def predict(self,x1):
        self.alpha = np.dot(x1, self.v)
        self.b = sigmoid(self.alpha - self.gamma)
        self.beta = np.dot(self.b, self.w)
        self.hat_y = sigmoid(self.beta - self.theta)
        return self.hat_y
def read_iris():
    with open("E:/desktop/3rdgrade-1/机器学习/IRIS/iris.csv", "r") as file:#打开CSV文件
        reader = csv.reader(file)#读取
        lst = list(reader)#获取list
        lst.pop(0)#第一行不要
    data,label=[],[]
    for i in lst:
        for j in range(1, 5):
            i[j] = float(i[j])#转换为浮点类型
        data.append(i[1:5])#data增加
        label.append(i[5])#tag增加
    return data,label
    # for i in lst:print(i)
def read_wine():
    path= "E:/desktop/3rdgrade-1/机器学习/winequality_data.xlsx"
    book = xlrd.open_workbook(path)#获取xlsx文件
    sheet1 = book.sheets()[0]#第一页
    row,col=sheet1.nrows,sheet1.ncols#行数列数
    #print(row,col)
    data,label=[],[]
    for i in range(1,row):
        tmp=[]
        for j in range(col-1):
            tmp.append(sheet1.cell(i,j).value)#获取到某行某列值
        label.append(sheet1.cell(i,col-1).value)#标签更新
        data.append(tmp)#数据更新
    #print(data);print(label)
    return data,label
img,label=read_iris()
enc=preprocessing.LabelEncoder()
label=enc.fit_transform(label)
s=set()
for i in label:s.add(i)
tot_img,tot_label=np.matrix(img),np.matrix(label)
tot_label=tot_label.reshape(-1,1)#维度不对，应该设置为n行1列
tot_label=np_utils.to_categorical(tot_label)
input,hidden,output=len(img[0]),125,len(s)#设置输入层，隐藏层，输出层的神经元个数 125best now

ANN1=ANN(x=tot_img,y=tot_label,learning_rate=0.2,input_num=input,hidden_num=hidden,output_num=output,epochs=20)
ANN1.train_std()