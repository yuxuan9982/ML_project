import csv
from math import log, log2
from random import shuffle

import xlrd
from sklearn.model_selection import train_test_split


def get_ant(data):
    label,ent={},0.0
    for i in data:
        if i[-1] not in label.keys():label[i[-1]]=0
        label[i[-1]]+=1
    for key in label:
        posi= float(label.get(key))/len(data)
        ent -= posi*log2(posi)
    return ent
def gain(data,a):
    gain = get_ant(data)
    feature= [item[a] for item in data]
    feature=set(feature)
    for fea in feature:
        n_data=[]
        for i in data:
            if(i[a]==fea):n_data.append(i)
        posi = len(n_data)/len(data)
        gain-= posi*get_ant(n_data)
    return gain


def handle_continuous(data,a):
    candidate,lst=[],[sample[a] for sample in data]
    for i in range(1,len(lst)):
        candidate.append((lst[i]+lst[i-1])/2)
    best_ent,best_candi,base_ent=-0x3f3f3f3f,0,get_ant(data)
    for candi in candidate:
        l_data,r_data=[],[]
        for i in data:
            if i[a]>=candi:
                r_data.append(i)
            else:l_data.append(i)
        ent1,posi1=get_ant(l_data),len(l_data)/len(data)
        ent2,posi2=get_ant(r_data),len(r_data)/len(data)
        tmp_ent= base_ent-posi1*ent1-posi2*ent2
        if tmp_ent >=best_ent:
            best_ent=tmp_ent
            best_candi=candi
    return best_ent,best_candi


def find_best(data):
    best_ent,best_fea=-0x3f3f3f3f,0
    for i in range(len(data[0])-1):
        if type(data[0][i]).__name__=='float':continue
        if gain(data,i) > best_ent:
            best_ent,best_fea=gain(data,i),i
    return best_ent,best_fea

def find_best_continuous(data):
    best_ent, best_fea,best_val = -0x3f3f3f3f,0,0
    for i in range(len(data[0]) - 1):
        if type(data[0][i]).__name__ != 'float': continue
        candi_ent,candi_val=handle_continuous(data,i)
        if candi_ent > best_ent:
            best_ent, best_fea,best_val = candi_ent, i,candi_val
    return best_ent, best_fea,best_val

def majority(label):
    cnt={}
    for i in label:
        if i not in cnt:cnt[i]=0
        cnt[i]+=1
    major,idx=0,0
    for i in cnt.keys():
        if cnt.get(i) >=major:
            major,idx=cnt.get(i),i
    return idx
# label=["T","T","F"]
# print(majority(label))
def dfs(data,name):
    label = [i[-1] for i in data]
    if(label.count(label[0])==len(label)):
        return label[0]
    if len(data[0])==1:
        return majority(label)

    best_ent,best = find_best(data)
    best_ent2,best2,best2_val=find_best_continuous(data)

    #print(best_ent2,best2,best2_val)
    #print(data[0],best,name)
    if best_ent2<best_ent:
        best_label = name[best]
        Tree ={best_label:{}}
        del(name[best])
        dif_fea=set([i[best] for i in data])
        for fea in dif_fea:
            n_data,n_name= [],name[:]#without n_name origin name will be changed
            for i in data:
                if (i[best] == fea):
                    tmp=[]
                    for j in range(len(i)):
                        if j!=best:tmp.append(i[j])
                    n_data.append(tmp)
            Tree[best_label][fea]=dfs(n_data,n_name)
        return Tree
    else:
        best_label=name[best2]
        Tree={best_label:{}}
        n_data,n_name,n_data2,n_name2=[],name[:],[],name[:]
        for i in data:
            if i[best2]<best2_val:
                n_data.append(i)
            else: n_data2.append(i)
        Tree[best_label]["<"+str(best2_val)]=dfs(n_data,n_name)
        Tree[best_label][">="+str(best2_val)]=dfs(n_data2,n_name2)
        return Tree



def predict(Tree,name,test):
    feature,key= 0,Tree.keys()
    for i in key:
        feature=i
        break
    dic2= Tree[feature]
    #print(dic2)
    index= name.index(feature)
    if(type(test[index]).__name__=='float'):
        val= list(dic2.keys())[0]
        val = float(val[1:])
        if(test[index]<val):
            res='<'+str(val)
        else:
            res='>='+str(val)
        if(type(dic2[res]).__name__=='dict'):
            return predict(dic2[res],name,test)
        else:
            return dic2[res]
    for key in dic2.keys():
        if test[index]==key:
            if(type(dic2[key]).__name__=='dict'):
                return predict(dic2[key],name,test)
            else:
                return dic2[key]
def get_water():
    path = "E:/desktop/3rdgrade-1/机器学习/water2.xlsx"
    book = xlrd.open_workbook(path)
    sheet1 = book.sheets()[0]
    row, col = sheet1.nrows, sheet1.ncols
    #print(row, col)
    data, label = [], []
    for i in range(1, row):
        tmp = []
        for j in range(col):
            tmp.append(sheet1.cell(i, j).value)
        data.append(tmp)

    tmp = []
    for j in range(col):tmp.append(sheet1.cell(0, j).value)
    label=tmp
    # print(data);print(label)
    return data, label

def read_iris():
    with open("E:/desktop/3rdgrade-1/机器学习/IRIS/iris.csv", "r") as file:#打开CSV文件
        reader = csv.reader(file)#读取
        lst = list(reader)#获取list
    label=lst[0][1:]
    lst.pop(0)#第一行不要
    data=[]
    for i in lst:
        for j in range(1, 5):
            i[j] = float(i[j])#转换为浮点类型
        data.append(i[1:])#data增加
    #print(data)
    #print(label)
    return data,label
    # for i in lst:print(i)

def read_wine():
    path= "E:/desktop/3rdgrade-1/机器学习/winequality_data.xlsx"
    book = xlrd.open_workbook(path)
    sheet1 = book.sheets()[0]
    row, col = sheet1.nrows, sheet1.ncols
    # print(row, col)
    data, label = [], []
    for i in range(1, row):
        tmp = []
        for j in range(col):
            tmp.append(sheet1.cell(i, j).value)
        data.append(tmp)
    tmp = []
    for j in range(col): tmp.append(sheet1.cell(0, j).value)
    label = tmp
    #print(data);print(label)
    return data, label


data,label=read_wine()
#print(data)
#print(label)
###divide
shuffle(data)
train,test=[],[]
for i in range(len(data)):
    if(i<len(data)*0.8):
        train.append(data[i])
    else:
        test.append(data[i])
###devide
lab2=label[:]
Tree = dfs(train,lab2)
print(Tree)
cnt=0
for d in test:
    if (predict(Tree,label,d[:-1])==d[-1]):
        cnt+=1
print('accuracy:',cnt/len(test))
