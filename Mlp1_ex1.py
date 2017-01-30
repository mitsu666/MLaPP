import numpy as np
from sklearn.datasets import fetch_mldata
mnist=fetch_mldata('MNIST original')
#print mnist.data.shape
print type(mnist.data)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)

# 学習用データN個，検証用データを残りの個数に設定
N = 60000
x_train, x_test = np.split(mnist.data,      [N])
y_train, y_test = np.split(mnist.target,    [N])
N_test = y_test.size

neigh.fit(x_train, y_train)

list=[]

#print(neigh.predict(x_test[100]))
for x,y in zip(x_test,y_test):
    a=neigh.predict(x)
    b=y
    if a[0]==b:
        print a[0],b
        x=1
    else:
        x=0
    list.append(x)
#print (y_test[100])
#list.append(if a==b)
print list

print 1-(float(list.count(1))/len(x_test))