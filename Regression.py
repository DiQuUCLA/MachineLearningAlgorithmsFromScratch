import numpy as np
from util import *
class LinearRegression:
    def __init__(self, X=None, y=None, random_state=0):
        if isinstance(X, list):
            X = np.array(X)
        
        np.random.seed(random_state)

        self.params = {}
        if X:
            self.init_paramaters(X)
            self.fit(X)

    def get_input_size_(self, X):
        # This function provide the number of instance and instance size
        if len(X.shape) == 1:
            return len(X), 1
        elif len(X.shape) == 2:
            return X.shape[0], X.shape[1]
        raise Exception('X has invalid dimension {}'.format(len(X.shape)))

    def init_paramaters(self,X):
        # Init parameters Weight vectors and bias constant for the model

        # The shape of X should be (m, n) where m is the number of 
        # instance and n is the vector size of each instance
        
        if isinstance(X, list):
            X = np.array(X)
        
        _, vector_length = self.get_input_size_(X)

        self.params['W'] = np.random.normal(size=vector_length)
        self.params['b'] = np.random.normal(size=1)


    def predict(self, X):
        if isinstance(X, list):
            X = np.array(X)
    
        if 'W' not in self.params:
            self.init_paramaters(X)
    
        instance_num, vector_length = self.get_input_size_(X)
        X = X.reshape(instance_num, vector_length)
        W = self.params['W']
        b = self.params['b']
        
        # weight_sum = W * X
        weight_sum = np.matmul(X,W)
        assert weight_sum.shape[0] == instance_num

        # prediction = weight_sum + bias
        pred_y = weight_sum + b
        assert pred_y.shape[0] == instance_num
        
        return pred_y

    def update(self, X, y, metrics="MSE")
        pred_y = self.prediction(X)
        if metrics == "MSE":
            loss = MSE(pred_y, y)
        elif metrics == "MAE":
            loss = MAE(pred_y, y)

