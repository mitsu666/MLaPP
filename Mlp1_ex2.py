from pyflann import *
from numpy import *
from numpy.random import *
from sklearn.datasets import fetch_mldata
mnist=fetch_mldata('MNIST original')
# 学習用データN個，検証用データを残りの個数に設定
N = 60000
x_train, x_test = np.split(mnist.data,      [N])
y_train, y_test = np.split(mnist.target,    [N])
N_test = y_test.size
dataset = x_train
testset = x_test
flann = FLANN()
params = flann.build_index(dataset, target_precision=0.9, log_level = "info"); print params
result, dists = flann.nn_index(testset,1, checks=params["checks"]);
list=[]
for i,y in enumerate(y_test):
    a=y_train[result[i]]
    b=y
    if a==b:
        #print a,b
        x=1
    else:
        x=0
    list.append(x)
print list.count(1)
print 1-(float(list.count(1))/len(x_test))