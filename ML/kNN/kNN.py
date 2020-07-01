import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from math import sqrt

def kNN_classify(k, X_train, y_train, x):
    assert 1 <= k <= X_train.shape[0], "k must be valid"
    assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must equal to the size of y_train"
    assert X_train.shape[1] == x.shape[0], \
            "the  feature number of x must be equal to X_train"
    
    distances = [sqrt(np.sum((x_train - x)**2)) for x_train in X_train]
    nearest = np.argsort(distances)
    
    topK_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(topK_y)
    return votes.most_common(1)[0][0]



# sample data
raw_data_x = [[3.3935,2.3312],
              [3.1101,1.7815],
              [1.3438,3.3683],
              [3.5822,4.6791],
              [2.2803,2.8669],
              [7.4234,4.6965],
              [5.7450,3.5339],
              [9.1721,2.5111],
              [7.7923,3.4240],
              [7.9398,0.7916]
             ]
raw_data_y = [0,0,0,0,0,1,1,1,1,1]

x_train = np.array(raw_data_x)
y_train = np.array(raw_data_y)

x = np.array([8.0936,3.3657])

# display the image
plt.scatter(x_train[y_train==0,0],x_train[y_train==0,1],color='g')
plt.scatter(x_train[y_train==1,0],x_train[y_train==1,1],color='r')
plt.scatter(x[0],x[1],color='b')
plt.show()

# test
y = kNN_classify(6,x_train,y_train,x)
print(f"label: {y}")