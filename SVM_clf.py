from sklearn.model_selection import StratifiedKFold
from libsvm.commonutil import *
from libsvm.svmutil import *
from libsvm.svm import *
#from sklearn.model_selection import KFold
import numpy as np
import math


y, x = svm_read_problem('DogsVsCats.train')
yi, xi = svm_read_problem('DogsVsCats.test')

def get_idx(data, label, idx):
    data_t = []
    label_t = []
    for i in idx:
        data_t.append(data[i])
        label_t.append(label[i])

    return data_t, label_t

#10-fold cv
cv = StratifiedKFold(n_splits=10)
train_fold = []
val_fold = []


for t, val in cv.split(x, y):
    train_fold.append(t)
    val_fold.append(val)

acc = []
for i in range(len(train_fold)):

    train_idx = train_fold[i]
    val_idx = val_fold[i]

    x_train, y_train = data_index(x, y, train_idx)
    x_val, y_val = data_index(x, y, val_idx)
    # for linear kernel
    model = svm_train(y_train, x_train, '-t 0')
    # for polynomial kernel
    # model = svm_train(y_train, x_train, '-t 1 -d 5')
    p_label, p_acc, p_val = svm_predict(y_val, x_val, model)
    acc.append(p_acc[0])

print(acc)
# print(np.mean(acc))

'''
# test accuracy
model = svm_train(y, x, '-t 0')
# model = svm_train(y_train, x_train, '-t 1 -d 5')
p_label, p_acc, p_val = svm_predict(yi, xi, model)
'''
