import numpy as np 
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from Metrics.metrics import r2_score

class SimpleLinearRegression1:

    def __init__(self):
        self.a_ = None
        self.b_ = None
    
    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, \
            "It should be single feature training data."
        assert len(x_train) == len(y_train), \
            "the length of feature training data should be equal to the target training data."
        
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0

        for x_i, y_i in zip(x_train, y_train):
            num += (x_i - x_mean) * (y_i - y_mean)
            d += (x_i - x_mean) ** 2
        
        self.a_ = num/d 
        self.b_ = y_mean - self.a_ * x_mean

        return self
    
    def predict(self, x_predict):
        assert x_predict.ndim == 1,\
            "It should be single feature training data."
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"
        
        return np.array([self._predict(x) for x in x_predict])
    
    def _predict(self, x_single):
        return self.a_ * x_single + self.b_
    
    def __repr__(self):
        return "SimpleLinearRegression1()"


class SimpleLinearRegression2:

    def __init__(self):
        self.a_ = None
        self.b_ = None
    
    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, \
            "It should be single feature training data."
        assert len(x_train) == len(y_train), \
            "the length of feature training data should be equal to the target training data."
        
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)

        
        self.a_ = num/d 
        self.b_ = y_mean - self.a_ * x_mean

        return self
    
    def predict(self, x_predict):
        assert x_predict.ndim == 1,\
            "It should be single feature training data."
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"
        
        return np.array([self._predict(x) for x in x_predict])
    
    def _predict(self, x_single):
        return self.a_ * x_single + self.b_
    
    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)
    
    def __repr__(self):
        return "SimpleLinearRegression2()"


if __name__ == "__main__":
    x = np.array([1., 2., 3., 4., 5.])
    y = np.array([1., 3., 2., 3., 5.])
    reg1 = SimpleLinearRegression2()
    reg1.fit(x, y)
    # print(reg1.predict(np.array([6, 7])))
    plt.scatter(x, y)
    plt.plot(x, reg1.predict(x), color='r')
    plt.axis([0, 6, 0, 6])
    plt.show()
    print(reg1.score(x, y))