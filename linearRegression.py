import numpy as np 
import matplotlib.pyplot as plt

x = np.array([0.3, 0.5, 0.6, 0.72, 0.9, 1.4, 1.7])
y = np.array([3, 4.5, 7.6, 8.72, 9.1, 14.3, 22.7])

def designMatrix(x, degree):
    matrix_list = []
    for d in range(1, degree+1):
        matrix_list.append(np.power(x, d))
    ones = []
    ones = np.ones((x.shape[0],1))
    matrix_list.append(ones)
    phi = np.concatenate(matrix_list, axis = 1)
    w = np.linalg.pinv(phi).dot(y)
    return phi, w

def MSE(y, y_hat):
    return np.square(y-y_hat).mean(axis=0) #try 1

x = x[..., np.newaxis]
y = y[..., np.newaxis]

phi, w = designMatrix(x,2)
y_hat = phi.dot(w)

plt.plot(x,y,'ro')
plt.plot(np.squeeze(x), y_hat)
plt.show()

print(MSE(y, y_hat))