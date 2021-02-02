import os
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time as time

data = np.loadtxt('C:/Users/moham/OneDrive/Desktop/5/PMDL/Assig.1/CCPP/fold1.txt')
order = 7
kf = KFold(n_splits = 5, shuffle = True)
x_axis = ["AT, AP", "AT, V, AP", "AT, V, AP, RH"]
Erms = np.zeros(len(x_axis))
Erms_test = np.zeros(len(x_axis))

for i in range(len(x_axis)):
    for train_index, test_index in kf.split(data):
        data_train, data_test = data[train_index], data[test_index]
        PE_train, PE_test = data[train_index, 4], data[test_index, 4]

        if i == 0:
            X = np.concatenate((data_train[:,0], data_train[:,2]), axis = 0)
            X = np.reshape(X, (data_train.shape[0], -1))
            a = np.ones(X.shape[0])
            X = np.c_[X,a]

            X_test = np.concatenate((data_test[:,0], data_test[:,2]), axis = 0)
            X_test = np.reshape(X_test, (data_test.shape[0], -1))
            a = np.ones(X_test.shape[0])
            X_test = np.c_[X_test,a]

        elif i == 1:
            X = np.concatenate((data_train[:,0], data_train[:,1], data_train[:,2]), axis = 0)
            X = np.reshape(X, (data_train.shape[0], -1))
            a = np.ones(X.shape[0])
            X = np.c_[X,a]

            X_test = np.concatenate((data_test[:,0], data_test[:,1], data_test[:,2]), axis = 0)
            X_test = np.reshape(X_test, (data_test.shape[0], -1))
            a = np.ones(X_test.shape[0])
            X_test = np.c_[X_test,a]

        elif i == 2:
            X = np.concatenate((data_train[:,0], data_train[:,1], data_train[:,2], data_train[:,3]), axis = 0)
            X = np.reshape(X, (data_train.shape[0], -1))
            a = np.ones(X.shape[0])
            X = np.c_[X,a]

            X_test = np.concatenate((data_test[:,0], data_test[:,1], data_test[:,2], data_test[:,3]), axis = 0)
            X_test = np.reshape(X_test, (data_test.shape[0], -1))
            a = np.ones(X_test.shape[0])
            X_test = np.c_[X_test,a]

        w = np.matmul(np.linalg.pinv(X), PE_train)
        Erms[i] += np.matmul(np.transpose(PE_train - np.matmul(X, w)), (PE_train - np.matmul(X, np.transpose(w)))) # use the testing set
        Erms_test[i] += np.matmul(np.transpose(PE_test - np.matmul(X_test, w)), (PE_test - np.matmul(X_test, np.transpose(w)))) # use the testing set

for i in range(len(x_axis)):
    Erms[i] /= 5
    Erms[i] = np.sqrt(2*Erms[i]/(PE_train.shape))
    Erms_test[i] /= 5
    Erms_test[i] = np.sqrt(2*Erms_test[i]/(PE_test.shape))
plt.plot(x_axis, Erms, 'x-', label = "Train")
plt.plot(x_axis, Erms_test, 'o-', label = "Test")
plt.legend(loc = "upper left")
plt.grid()
plt.xlabel('Set')
plt.ylabel('Erms')
plt.show()
