import numpy as np
import matplotlib.pyplot as plt

w = 1.0

x_data = np.array([1,2,3])
y_data = np.array([2,4,6])

w_list = []
mse_list = []

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)

for w in np.arange(0.0, 4.1, 0.1):
    print("w = ", w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        l = loss(x_val, y_val)
        l_sum += l
        print ("\t", x_val, y_val, y_pred_val, l)

    print("MSE=", l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)


plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()