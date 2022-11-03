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


    def predict(self,img):#预测
        pred_x=img.reshape(-1,1)#输入数据格式标准化
        hidden_inputs=numpy.dot(self.wih.T,pred_x)#得到隐藏层的输入
        hidden_outputs=self.activation(hidden_inputs)#得到隐藏层的输出
        hidden_input2=numpy.dot(self.whh.T,hidden_outputs)
        hidden_outputs2=self.activation(hidden_input2)
        final_inputs=numpy.dot(self.who.T,hidden_outputs2)#得到输出层的输入
        final_outputs=self.activation(final_inputs)#得到输出层的输出
        return final_outputs#返回输出层的结果
def read_iris():
    with open("E:/desktop/IRIS/iris.csv", "r") as file:
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

img,label=read_iris()
enc=preprocessing.LabelEncoder()
label=enc.fit_transform(label)
#不同的标签种类数
s=set()
for i in label:s.add(i)

#train_img,test_img,train_label,test_label=sklearn.model_selection.train_test_split(img,label,test_size=0.3)
#数据集不多，暂时设置为全集
train_img,test_img,train_label,test_label=img,img,label,label

input,hidden,output=len(img[0]),200,len(s)#设置输入层，隐藏层，输出层的神经元个数
cut_sz=150
#print(test_img,test_img)
################使用keras来搭建的神经网络##################################

model=keras.models.Sequential()#创建sequential
model.add(Dense(200,input_dim=input,activation='relu'))#第一层激活函数选择relu能提升很大的准确率
model.add(Dense(100,activation='sigmoid'))#第二层单元设置为100比较合适，过大会导致过拟合，过小拟合效果不好
model.add(Dense(output,activation='softmax'))#输出结果
model.compile(optimizer='adam',loss='categorical_crossentropy')#选择adam优化，损失函数选择二元交叉熵准确率会提升一些（大概1%左右吧）
mmx=MinMaxScaler()#输入数据归一化
tot_img=np.concatenate([train_img,test_img])#把train和test放到一起弄成一个总的tot
tot_label=np.concatenate([train_label,test_label])#把train和test放到一起弄成一个总的tot
tot_label=tot_label.reshape(-1,1)#维度不对，应该设置为n行1列
tot_label=np_utils.to_categorical(tot_label)

tot=np.concatenate([tot_img,tot_label],axis=1)#把数据和标签按列拼接到一起
scaled=mmx.fit_transform(tot)#归一化


train,test=scaled[:cut_sz,:],scaled[cut_sz:,:]#把归一化的总数据还原成训练数据和测试数据
scaled_x,scaled_y=train[:,:input],train[:,input:]#再把训练数据拆分成输入和标签
# print(scaled_y)
test_x,test_y=test[:,:input],test[:,input:]
model.fit(scaled_x,scaled_y,batch_size=50,epochs=500)#进行训练，迭代20轮

predict_y=model.predict(test_x)#预测test数据的结果
sum1,sum2,acc=0,0,0
for i in range(len(predict_y)):
    maxindex=np.argmax(predict_y[i])#找到最大值所在的索引
    if(maxindex==test_label[i]):#如果预测结果正确
        sum1+=1
    #else:
        #show_num(test_img[i])
    sum2+=1
acc=sum1/sum2#计算准确率
print("使用keras搭建的神经网络的准确率:",acc)#输出观察得到的准确率

################使用keras来搭建的神经网络##################################

################使用手工搭建的神经网络##################################
learning_rate=0.05#确定学习率
ANN1=ANN(input,hidden,100,output,learning_rate)#实例化ANN人工神经网络
epochs=7#迭代次数设置为3
cnt=0
for k in range(epochs):#迭代
    cnt += 1#记录迭代次数
    for j in range(len(train_img)):#遍历所有训练数据
        target=np.zeros(10)+0.01#输出的处理
        target[int(train_label[j])]=0.99
        #print(train_y[j].reshape(-1,1))
        ANN1.train(scaled_x[j],target.reshape(-1,1))#人工神经网络进行训练
    print('epoch:----->',cnt)#输出迭代过程

    accuracy=0#计算准确性
    sum1,sum2=0,0
    for i in range(len(test_img)):#遍历预测数据
        #wa_x=[]

        outputs = ANN1.predict(test_x[i])#得到对应的输出
        pred_label=numpy.argmax(outputs)#得到输出的值最大的位置，也就是判断结果
        #print(pred_label,test_label[i])
        if(pred_label==test_label[i]):#如果判断正确
            sum1+=1
        #else :
            #wa_x.append(test_x)
        sum2+=1
        #show_num(wa_x)
    print('使用手工搭建的神经网络的准确率:',sum1/sum2)#输出准确性
################使用手工搭建的神经网络##################################
