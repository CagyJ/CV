import numpy as np 
from math import sqrt
from collections import Counter


class KNNClassifier:

    def __init__(self, k):

        assert k >= 1, "k must be valid"
        self.k = k 
        self._X_train = None
        self._y_train = None
    
    def fit(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must equal to the size of y_train"
        assert self.k <= X_train.shape[0], \
            "the size of X_train must be at least k."
        
        self._X_train = X_train
        self._y_train = y_train
        return self
    
    def predict(self, X_predict):
        assert self._X_train is not None and self._y_train is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "the feature number of X_predict must be equal to X_train"
        
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)
    
    def _predict(self, x):
        assert x.shape[0] == self._X_train.shape[1], \
            "the feature number of x must be equal to X_train"
        
        distances = [sqrt(np.sum((x_train - x)**2)) for x_train in X_train]
        nearest = np.argsort(distances)
        
        topK_y = [y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]
    
    def __repr__(self):
        return f"KNN(k={self.k})"

# test
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

X_train = np.array(raw_data_x)
y_train = np.array(raw_data_y)

x = np.array([8.0936,3.3657])
x_predict = x.reshape(1,-1)

knn_clf = KNNClassifier(k=6)
knn_clf.fit(X_train, y_train)
y_predict = knn_clf.predict(x_predict)
print(y_predict[0])