import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression  #产生一个随机回归问题

def gradient_descent(w,b,lr):
    for x,y in zip(X,Y):
        w -= lr * ((w*x + b) - y) * x
        b -= lr * ((w*x + b) - y)
    return w,b

def train(X,Y,rg=(150,170)):
    plt.ion()  #打开交互模式

    loss = lambda w,b: np.mean([(y - (w*x + b)) ** 2 for x,y in zip(X,Y)])
    # zip() 将X, Y中每一组数据打包成元组 (x,y)

    # train
    w,b = .0,.0
    for i in range(rg[0],rg[1]):
        print("%d loss: " %i, loss(w,b))
        w,b = gradient_descent(w,b,1/i)
    
    return w,b,loss



def display_plot(X,Y,w,b):
    plt.ion()
    plt.title("Linear Regression")
    plt.scatter(X,Y)
    plt.plot(X, w*X+b,color="red")
    plt.pause(0.1)


#test
X,Y,coef = make_regression(n_features=1, noise=9, bias=10., coef=True)
# X: 样本特征    Y: 输出    coef: 回归系数

X = X.flatten()  #等同于 X.reshape(-1)

w,b,loss = train(X,Y)

print("coef:",coef,", w:",w,", b:",b,", loss:",loss)

display_plot(X,Y,w,b)
