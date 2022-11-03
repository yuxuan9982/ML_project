import csv
with open("E:/desktop/IRIS/iris.csv","r") as file:
    reader= csv.reader(file)
    lst=list(reader)
for i in lst:print(i)