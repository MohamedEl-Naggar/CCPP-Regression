import os
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time as time
import itertools

data = np.loadtxt('C:/Users/moham/OneDrive/Desktop/5/PMDL/Assig.1/CCPP/fold1.txt')
kf = KFold(n_splits = 5, shuffle = True)
orders = [1, 2, 3, 4]
Erms = np.zeros(len(orders))
Erms_test = np.zeros(len(orders))

for order in orders:
    for train_index, test_index in kf.split(data):
        data_train, data_test = data[train_index], data[test_index]
        PE_train, PE_test = data[train_index, 4], data[test_index, 4]

        if order == 1:
            X = np.concatenate((data_train[:,0], data_train[:,1], data_train[:,2], data_train[:,3]), axis = 0)
            X = np.reshape(X, (data_train.shape[0], -1))
            a = np.ones(X.shape[0])
            X = np.c_[X,a]

            X_test = np.concatenate((data_test[:,0], data_test[:,1], data_test[:,2], data_test[:,3]), axis = 0)
            X_test = np.reshape(X_test, (data_test.shape[0], -1))
            a = np.ones(X_test.shape[0])
            X_test = np.c_[X_test,a]

        elif order == 2:
            X = np.concatenate((data_train[:,0], data_train[:,1], data_train[:,2], data_train[:,3]), axis = 0)
            length = [2]
            for r in length:
                comb = itertools.combinations_with_replacement([0, 1, 2, 3], r)
                for i in comb:
                        X = np.concatenate((X, data_train[:, i[0]] * data_train[:, i[1]]), axis = 0)

            X = np.reshape(X, (data_train.shape[0], -1))
            a = np.ones(X.shape[0])
            X = np.c_[X,a]

            X_test = np.concatenate((data_test[:,0], data_test[:,1], data_test[:,2], data_test[:,3]), axis = 0)
            length = [2]
            for r in length:
                comb = itertools.combinations_with_replacement([0, 1, 2, 3], r)
                for i in comb:
                        X_test = np.concatenate((X_test, data_test[:, i[0]] * data_test[:, i[1]]), axis = 0)
            X_test = np.reshape(X_test, (data_test.shape[0], -1))
            a = np.ones(X_test.shape[0])
            X_test = np.c_[X_test,a]

        elif order == 3:
            X = np.concatenate((data_train[:,0], data_train[:,1], data_train[:,2], data_train[:,3]), axis = 0)
            length = [2, 3]
            for r in length:
                comb = itertools.combinations_with_replacement([0, 1, 2, 3], r)
                for i in comb:
                    if r == 2:
                        X = np.concatenate((X, data_train[:, i[0]] * data_train[:, i[1]]), axis = 0)
                    if r == 3:
                        X = np.concatenate((X, data_train[:, i[0]] * data_train[:, i[1]] * data_train[:, i[2]]), axis = 0)
            X = np.reshape(X, (data_train.shape[0], -1))
            a = np.ones(X.shape[0])
            X = np.c_[X,a]

            X_test = np.concatenate((data_test[:,0], data_test[:,1], data_test[:,2], data_test[:,3]), axis = 0)
            length = [2, 3]
            for r in length:
                comb = itertools.combinations_with_replacement([0, 1, 2, 3], r)
                for i in comb:
                    if r == 2:
                        X_test = np.concatenate((X_test, data_test[:, i[0]] * data_test[:, i[1]]), axis = 0)
                    if r == 3:
                        X_test = np.concatenate((X_test, data_test[:, i[0]] * data_test[:, i[1]] * data_test[:, i[2]]), axis = 0)

            X_test = np.reshape(X_test, (data_test.shape[0], -1))
            a = np.ones(X_test.shape[0])
            X_test = np.c_[X_test,a]

        elif order == 4:
            X = np.concatenate((data_train[:,0], data_train[:,1], data_train[:,2], data_train[:,3]), axis = 0)
            length = [2, 3, 4]
            for r in length:
                comb = itertools.combinations_with_replacement([0, 1, 2, 3], r)
                for i in comb:
                    if r == 2:
                        X = np.concatenate((X, data_train[:, i[0]] * data_train[:, i[1]]), axis = 0)
                    if r == 3:
                        X = np.concatenate((X, data_train[:, i[0]] * data_train[:, i[1]] * data_train[:, i[2]]), axis = 0)
                    if r == 4:
                        X = np.concatenate((X, data_train[:, i[0]] * data_train[:, i[1]] * data_train[:, i[2]] * data_train[:, i[3]]), axis = 0)                      
            X = np.reshape(X, (data_train.shape[0], -1))
            a = np.ones(X.shape[0])
            X = np.c_[X,a]

            X_test = np.concatenate((data_test[:,0], data_test[:,1], data_test[:,2], data_test[:,3]), axis = 0)
            length = [2, 3, 4]
            for r in length:
                comb = itertools.combinations_with_replacement([0, 1, 2, 3], r)
                for i in comb:
                    if r == 2:
                        X_test = np.concatenate((X_test, data_test[:, i[0]] * data_test[:, i[1]]), axis = 0)
                    if r == 3:
                        X_test = np.concatenate((X_test, data_test[:, i[0]] * data_test[:, i[1]] * data_test[:, i[2]]), axis = 0)
                    if r == 4:
                        X_test = np.concatenate((X_test, data_test[:, i[0]] * data_test[:, i[1]] * data_test[:, i[2]] * data_test[:, i[3]]), axis = 0)                      
            X_test = np.reshape(X_test, (data_test.shape[0], -1))
            a = np.ones(X_test.shape[0])
            X_test = np.c_[X_test,a]

        w = np.matmul(np.linalg.pinv(X), PE_train)
        Erms[orders.index(order)] += np.matmul(np.transpose(PE_train - np.matmul(X, w)), (PE_train - np.matmul(X, np.transpose(w)))) 
        Erms_test[orders.index(order)] += np.matmul(np.transpose(PE_test - np.matmul(X_test, w)), (PE_test - np.matmul(X_test, np.transpose(w)))) # use the testing set

for i in range(len(orders)):
    Erms[i] /= 5
    Erms[i] = np.sqrt(2*Erms[i]/(PE_train.shape))
    Erms_test[i] /= 5
    Erms_test[i] = np.sqrt(2*Erms_test[i]/(PE_test.shape))
    
plt.plot(orders, Erms, 'x-', label = "Train")
plt.plot(orders, Erms_test, 'o-', label = "Test")
plt.legend(loc = " upper left")
plt.grid()
plt.xlabel('Order')
plt.ylabel('Erms')
plt.show()
