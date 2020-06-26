import numpy as np
import random
import matplotlib.pyplot as plt



def inference(w, b, x): # w:斜率  b:常数
    return w*x+b


def gradient(pred_y,gt_y,x): #计算单个
    diff = pred_y - gt_y
    dw = diff*x
    db = diff
    return dw,db


def cal_step_gradient(batch_x_list, batch_gt_y_list, w, b, lr):
    avg_dw, avg_db = 0,0
    batch_size = len(batch_x_list)
    for i in range(batch_size):
        pred_y = inference(w,b,batch_x_list[i])
        dw,db = gradient(pred_y, batch_gt_y_list[i],batch_x_list[i])
        avg_dw += dw
        avg_db += db
    avg_dw /= batch_size
    avg_db /= batch_size
    
    w -= lr*avg_dw
    b -= lr*avg_db
    return w,b


def loss_func(w,b,x_list,gt_y_list): #ground truth: 真实标签值
    avg_loss = 0
    for i in range(len(x_list)):
        avg_loss += 0.5*(w*x_list[i]+b - gt_y_list[i])**2  #方均差 h(x) - y
    avg_loss /= len(gt_y_list)
    return avg_loss


def train(x_list,gt_y_list,batch_size,lr,max_iterations): 
    w,b = 0,0
    num_sample = len(x_list)
    for i in range(max_iterations):
        #梯度更新
        batch_idxs = np.random.choice(len(x_list),batch_size)
        batch_x = [x_list[j] for j in batch_idxs]
        batch_y = [gt_y_list[j] for j in batch_idxs]
        w,b = cal_step_gradient(batch_x,batch_y,w,b,lr)
        print('w:{0}, b: {1}'.format(w,b))
        print('loss: {}'.format(loss_func(w,b,x_list,gt_y_list)))
    return w,b


# test
def gen_sample_data():
    w = random.randint(0,10) + random.random() #0-10的整数加上小于一的小数
    b = random.randint(0,5) + random.random()
    
    num_sample = 100
    x_list = []
    y_list = []
    print(w,b)
    for i in range(num_sample):
        x = random.randint(0,100)*random.random()
        y = w*x + b + random.random()*random.randint(-1,100)  #后者为填加的噪声
        x_list.append(x)
        y_list.append(y)
    return x_list,y_list

x_list,y_list = gen_sample_data()
w,b = train(x_list,y_list,batch_size=100,lr=0.0001,max_iterations=100)  #batch size可大可小，learn rate学习率应与其同步改变以防数据爆炸（batch_size调大，lr应该调小）

# display the diagram
x = np.linspace(0,100)
plt.plot(x,w*x+b,color = 'red')
plt.scatter(x_list,y_list)
plt.show()