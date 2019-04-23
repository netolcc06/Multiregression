#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
import assignment1

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:] #(195,33)
#x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:] #(100,33)
x_test = x[N_TRAIN:,:] #(95,33)
t_train = targets[0:N_TRAIN] #(100,1)
t_test = targets[N_TRAIN:] #(95,1)
cv = np.random.rand(10,10)
print(t_train.shape)
#cv = t_train.view()
cv = np.reshape(t_train,(10,10))
print(cv.shape)
print(t_train.shape)
print('------------------')
#cv[0,:] = t_train[:10, 0].view()


# TO DO:: Complete the linear_regression and evaluate_regression functions of the assignment1.py

# (w, tr_err) = a1.linear_regression()
# (t_est, te_err) = a1.evaluate_regression()


#countries
print(countries.shape)
#features
print(features.shape)
#values 
print(values.shape)
print('------------------')
print(str(max(values[:,0])))
c0 = countries[np.argmax(values[:,0])]
print(c0)

print(str(max(values[:,1])))
c1 = countries[np.argmax(values[:,1])]
print(c1)
print('------------------')

print(x.shape)
print(x_train.shape)
print(x_test.shape)
print(t_train.shape)
print(t_test.shape)


train_err = { } 
test_err = { }
for degree in range(1, 7):
    (w, t) = assignment1.linear_regression(x_train, t_train, 'polynomial', 0, degree)
    train_err[degree] = t
    print(t)

    (test_e, error) = assignment1.evaluate_regression(x_test, t_test, w , 'polynomial', degree)
    test_err[degree] = error
    print('we love money')


# Produce a plot of results.

plt.plot(test_err.keys(), test_err.values())
plt.plot(train_err.keys(), train_err.values())
plt.ylabel('RMS')
plt.legend(['Test error','Training error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()
