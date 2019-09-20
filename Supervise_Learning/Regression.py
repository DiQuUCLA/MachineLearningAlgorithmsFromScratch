import numpy as np
import sys

sys.path.insert(1, '../')

from util import *
class LinearRegression:
    def __init__(self, X=None, y=None, random_state=0):
        if isinstance(X, list):
            X = np.array(X)
        
        np.random.seed(random_state)

        self.params = {}
        if X:
            self.fit(X)

    def _get_input_size(self, X):
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


    def predict(self, X):
        if isinstance(X, list):
            X = np.array(X)
    
        if 'W' not in self.params:
            self.init_paramaters(X)
    
        instance_num, vector_length = self.get_input_size_(X)
        X = X.reshape(instance_num, vector_length)
        W = self.params['W']
        
        # weight_sum = W * X
        weight_sum = np.matmul(X,W)
        assert weight_sum.shape[0] == instance_num

        # prediction = weight_sum + bias
        pred_y = weight_sum + b
        assert pred_y.shape[0] == instance_num
        
        return pred_y

    def _compute_loss(self, X, y, metrics="MSE"):

        pred_y = self.prediction(X)
        if metrics == "MSE":
            loss = MSE(pred_y, y)
        elif metrics == "MAE":
            loss = MAE(pred_y, y)
        
        return loss

    def _compute_gradient(self, X, y, y_pred):
        diff_of_predict_and_true_label = y_pred - y
        sum_of_gradient = np.multiply(diff_of_predict_and_true_label.T, X)
        return sum_of_gradient.mean(axis=1)

    def fit(self, X, y, batch_size=32, iters=30, learn_rate=0.001):
        X = np.array(X)
        data_size = X.shape[0]
        X = np.concatenate((np.ones((data_size,1)), X), axis=1)
        self.init_paramaters(X)
        for it in range(iters):
            for i in range(0, data_size, batch_size):
                X_batch = X[i * batch_size : (i + 1) * batch_size]
                y_batch = y[i * batch_size : (i + 1) * batch_size]
                y_pred = self.predict(X)
                gradient_w = self._compute_gradient(X_batch, y_batch, y_pred)
                self.params['W'] += gradient_w * learn_rate 
            
        
        

