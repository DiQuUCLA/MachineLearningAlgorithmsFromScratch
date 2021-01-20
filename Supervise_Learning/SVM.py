import torch
import torch.nn as nn
import torch.nn.functional as F

class SVM:
    def __init__(self, dim):
        self.W = torch.rand(dim, 1, dtype=torch.float32);
        self.b = torch.rand(1, dtype=torch.float32)

    def forward(self, X):
        return torch.mm(X, self.W) + self.b

    def predict(self, X):
        return self.forward(X) > 0

    def update(self, X, y, C, lr):
        z = self.forward(X)
        clf = torch.mul(y, z) < 1

        self.W -= lr * (self.W - 
                C * torch.mean(torch.mul(y*1.0, X), dim=0).unsqueeze(dim=1))
        self.b -= lr * (self.b - C * torch.mean(y))
        return self.W, self.b

if __name__ == '__main__':
    svm = SVM(10)
    X = torch.rand(5, 10)
    y = torch.randint(-1, 2, (5,1))

    print(svm.forward(X))

    print(svm.update(X, y.float(), 0.5, 0.01))
