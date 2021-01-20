import torch
import torch.nn as nn
import torch.nn.functional as F

def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

class LogisticRegression:
    def __init__(self, dim):
        self.W = torch.rand(dim, 1,dtype=torch.float32)
        self.b = torch.rand(1, dtype=torch.float32)

    def forward(self, X):
        return torch.mm(X, self.W) + self.b

    def predict(self, X):
        return sigmoid(self.forward(X))

    def update(self, X, y, learning_rate):
        z = torch.mul(self.forward(X), y)
        
        grad_W = torch.divide(torch.mul(X, y), (1 + torch.exp(z)))
        grad_b = torch.divide(y, 1 + torch.exp(z))

        grad_W = -1 * torch.mean(grad_W, dim=0).unsqueeze(dim=1)
        grad_b = -1 * torch.mean(grad_b)

        self.W -= learning_rate * grad_W
        self.b -= learning_rate * grad_b
        
        return grad_W, grad_b

if __name__ == '__main__':
    lr = LogisticRegression(10)
    X = torch.rand(5, 10)
    y = torch.rand(5, 1)

    pred = lr.forward(X)
    gW, gb = lr.update(X, y, 0.001)
    print(pred)

    print(gW)
    print(gb)
