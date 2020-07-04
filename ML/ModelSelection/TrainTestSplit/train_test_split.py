import numpy as np 

def train_test_split(X, y, test_ratio=0.2, seed=None):
    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, \
        "test_ration must be valid"
    
    if seed:
        np.random.seed(seed)
    
    # 打乱索引
    shuffle_indexes = np.random.permutation(len(X))
    
    # 设定测试数据集的大小
    test_ratio = 0.2
    test_size = int(len(X) * test_ratio)
    
    # 确定训练数据集和测试数据集对应的索引
    test_indexes = shuffle_indexes[:test_size]
    train_indexes = shuffle_indexes[test_size:]

    # 训练数据集和测试数据集
    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, y_train, X_test, y_test