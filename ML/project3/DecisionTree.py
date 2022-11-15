from math import log, log2

import xlrd


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

def find_best(data):
    best_ent,best_fea=0,0
    for i in range(len(data[0])-1):
        if gain(data,i) > best_ent:
            best_ent,best_fea=gain(data,i),i
    return best_fea

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
    best = find_best(data)
    #print(data[0],best,name)
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

def predict(Tree,name,test):
    feature,key= 0,Tree.keys()
    for i in key:
        feature=i
        break
    dic2= Tree[feature]
    #print(dic2)
    index= name.index(feature)
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

data,label=get_water()
#print(data)
#print(label)
lab2=label[:]
Tree = dfs(data,lab2)
#print(Tree)
a=0
for d in data:
    print(predict(Tree,label,d))
