import numpy as np
import matplotlib.pyplot as plt

w = 1.0

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

print("(Before training)", 4, forward(4))
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w - 0.01*grad
        l = loss(x_val, y_val)
        print ("\t", x_val, y_val, l)

    print("Progress: ", epoch, "w= ", round(w,2), "loss= ", round(l,2))
    #print("MSE=", l_sum/3)
    #w_list.append(w)
    #mse_list.append(l_sum/3)

print("(After training)", 4, forward(4))

#plt.plot(w_list, mse_list)
#plt.ylabel('Loss')
#plt.xlabel('w')
#plt.show()