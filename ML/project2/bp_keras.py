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
input,hidden,output=len(img[0]),200,len(s)#设置输入层，隐藏层，输出层的神经元个数
#print(test_img,test_img)
mmx=MinMaxScaler()#输入数据归一化
# tot_img=np.concatenate([train_img,test_img])#把train和test放到一起弄成一个总的tot
# tot_label=np.concatenate([train_label,test_label])#把train和test放到一起弄成一个总的tot
tot_img,tot_label=np.matrix(img),np.matrix(label)
tot_label=tot_label.reshape(-1,1)#维度不对，应该设置为n行1列
tot_label=np_utils.to_categorical(tot_label)

# tot=np.concatenate([tot_img,tot_label],axis=1)#把数据和标签按列拼接到一起
# tot=mmx.fit_transform(tot)#不归一化已经100了
# tot_img,tot_label=tot[:,:input],tot[:,input:]#归一化以后的数据和tag
train_x,test_x,train_y,test_y=sklearn.model_selection.train_test_split(tot_img,tot_label,train_size=0.8)#抽取80%作为训练集

################使用keras来搭建的神经网络##################################

model=keras.models.Sequential()#创建sequential
model.add(Dense(200,input_dim=input,activation='relu'))#第一层激活函数选择relu能提升很大的准确率
model.add(Dense(100,activation='sigmoid'))#第二层单元设置为100比较合适，过大会导致过拟合，过小拟合效果不好
model.add(Dense(output,activation='softmax'))#输出结果
model.compile(optimizer='adam',loss='categorical_crossentropy')#选择adam优化，损失函数选择二元交叉熵准确率会提升一些（大概1%左右吧）

model.fit(train_x,train_y,batch_size=30,epochs=500)#进行训练，迭代20轮

predict_y=model.predict(test_x)#预测test数据的结果
sum1,sum2,acc=0,0,0
for i in range(len(predict_y)):
    maxindex=np.argmax(predict_y[i])#找到最大值所在的索引
    if(test_y[i][maxindex]==1):#如果预测结果正确
        sum1+=1
    sum2+=1
acc=sum1/sum2#计算准确率
print("使用keras搭建的神经网络的准确率:",acc*100,"%")#输出观察得到的准确率

################使用keras来搭建的神经网络##################################
