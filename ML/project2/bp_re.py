import csv
import gzip
import struct
import random
import keras.models
import numpy as np
import os
import scipy
import matplotlib.pyplot as plt
import numpy.random
import sklearn.model_selection
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
from sklearn import preprocessing
import xlrd
def sigmoid(x):
    return scipy.special.expit(x)
class ANN:#人工神经网络
    def __init__(self,input,hidd1,hidd2,output,learn_rate):
        self.i=input#输入层的神经元个数
        self.h1=hidd1#隐藏层的神经元个数
        self.h2=hidd2
        self.o=output#输出层的神经元个数
        self.lr=learn_rate#学习率/步长
        self.wih= numpy.random.normal(0,pow(self.h1,-0.5),size=(self.i,self.h1))
        self.whh= numpy.random.normal(0,pow(self.h2,-0.5),size=(self.h1,self.h2))
        #初始化输入层和隐藏层神经元之间的参数
        self.who = numpy.random.normal(0, pow(self.o, -0.5), size=(self.h2, self.o))
        #初始化输出层和隐藏层神经元之间的参数
        self.activation= lambda x:sigmoid(x)#确定激活函数

    def train(self,img,lab):#训练
        train_x,train_y=numpy.array(img,ndmin=2).T,numpy.array(lab,ndmin=2).T
        #训练数据格式标准化
        hidden_inputs1= numpy.dot(self.wih.T,train_x)#得到隐藏层的输入
        hidden_outputs1=self.activation(hidden_inputs1)#得到隐藏层的输出
        hidden_inputs2= numpy.dot(self.whh.T,hidden_outputs1)
        hidden_outputs2=self.activation(hidden_inputs2)

        final_inputs=numpy.dot(self.who.T,hidden_outputs2)#得到输出层的输入
        final_outputs=self.activation(final_inputs)#得到输出层的输出

        output_dif=(lab-final_outputs)*final_outputs*(1.0-final_outputs)#计算差值
        hidden_dif2=numpy.dot(self.who,output_dif)*hidden_outputs2*(1.0-hidden_outputs2)
        hidden_dif=numpy.dot(self.whh,hidden_dif2)*hidden_outputs1*(1.0-hidden_outputs1)

        self.who += self.lr * numpy.dot(output_dif,hidden_outputs2.T).T
        self.whh += self.lr * numpy.dot(hidden_dif2,hidden_outputs1.T).T
        self.wih += self.lr * numpy.dot(hidden_dif,train_x.T).T

    def predict(self, img):  # 预测
        pred_x = img.reshape(-1, 1)  # 输入数据格式标准化
        hidden_inputs = numpy.dot(self.wih.T, pred_x)  # 得到隐藏层的输入
        hidden_outputs = self.activation(hidden_inputs)  # 得到隐藏层的输出
        hidden_input2 = numpy.dot(self.whh.T, hidden_outputs)
        hidden_outputs2 = self.activation(hidden_input2)
        final_inputs = numpy.dot(self.who.T, hidden_outputs2)  # 得到输出层的输入
        final_outputs = self.activation(final_inputs)  # 得到输出层的输出
        return final_outputs  # 返回输出层的结果
def read_iris():
    with open("E:/desktop/3rdgrade-1/机器学习/IRIS/iris.csv", "r") as file:
        reader = csv.reader(file)
        lst = list(reader)
        lst.pop(0)
    data,label=[],[]
    for i in lst:
        for j in range(1, 5):
            i[j] = float(i[j])
        data.append(i[1:5])
        label.append(i[5])
    return data,label
    # for i in lst:print(i)
def read_wine():
    path= "E:/desktop/3rdgrade-1/机器学习/winequality_data.xlsx"
    book = xlrd.open_workbook(path)
    sheet1 = book.sheets()[0]
    row,col=sheet1.nrows,sheet1.ncols
    print(row,col)
    data,label=[],[]
    for i in range(1,row):
        tmp=[]
        for j in range(col-1):
            tmp.append(sheet1.cell(i,j).value)
        label.append(sheet1.cell(i,col-1).value)
        data.append(tmp)
    #print(data);print(label)
    return data,label
img,label=read_wine()
enc=preprocessing.LabelEncoder()
label=enc.fit_transform(label)
s=set()
for i in label:s.add(i)
input,hidden,output=len(img[0]),200,len(s)#设置输入层，隐藏层，输出层的神经元个数
#print(test_img,test_img)
mmx=MinMaxScaler()#输入数据归一化
tot_img,tot_label=np.matrix(img),np.matrix(label)
tot_label=tot_label.reshape(-1,1)#维度不对，应该设置为n行1列
tot_label=np_utils.to_categorical(tot_label)

tot=np.concatenate([tot_img,tot_label],axis=1)#把数据和标签按列拼接到一起
tot=mmx.fit_transform(tot)#不归一化已经100了
tot_img,tot_label=tot[:,:input],tot[:,input:]#归一化以后的数据和tag
train_x,test_x,train_y,test_y=sklearn.model_selection.train_test_split(tot_img,tot_label,train_size=0.8)#抽取80%作为训练集
################使用手工搭建的神经网络##################################
learning_rate=0.15#确定学习率
ANN1=ANN(input,hidden,100,output,learning_rate)#实例化ANN人工神经网络
epochs=200
for k in range(epochs):#迭代
    for j in range(len(train_x)):#遍历所有训练数据
        ANN1.train(train_x[j],train_y[j].reshape(-1,1))#人工神经网络进行训练
    accuracy=0#计算准确性
    sum1,sum2=0,0
    for i in range(len(test_x)):#遍历预测数据
        #wa_x=[]
        outputs = ANN1.predict(test_x[i])#得到对应的输出
        pred_label=numpy.argmax(outputs)#得到输出的值最大的位置，也就是判断结果
        #print(pred_label,test_label[i])
        if(test_y[i][pred_label]==1):#如果判断正确
            sum1+=1
        sum2+=1
    if k%10==9:print('epoch:----->',k+1,'使用手工搭建的神经网络的准确率:',sum1/sum2)#输出准确性
################使用手工搭建的神经网络##################################
