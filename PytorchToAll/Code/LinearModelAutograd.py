import torch
from torch.autograd import Variable
import numpy as np

w = Variable(torch.Tensor([1.0]), requires_grad=True)

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w_list = []
mse_list = []

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)

def gradient(x,y):
    return 2*x*(w*x-y)

print("(Before training)", 4, forward(4).data)
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()
        w.data = w.data - 0.01*w.grad.data
        print ("\t", x_val, y_val, w.grad.data)
        w.grad.data.zero_()

    print("Progress: ", epoch, "w= ", w.data[0], "loss= ", l.data[0])

print("(After training)", 4, forward(4).data)
print(w.data)
print(w.grad) #same
print(w.grad.data) #same