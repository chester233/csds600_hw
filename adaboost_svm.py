import numpy as np
import pandas as pd

from libsvm.svmutil import *
from libsvm.svm import *
from libsvm.commonutil import *

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier

train_y, train_x = svm_read_problem('DogsVsCats.train')
test_y, test_x = svm_read_problem('DogsVsCats.test')

x_train = []
for i in train_x:
    i = list(i.values())
    x_train.append(i)
x_train = np.array(x_train)

x_test = []
for i in test_x:
    i = list(i.values())
    x_test.append(i)
x_test = np.array(x_test)

y_train = np.array(train_y).astype(float)
y_test = np.array(test_y).astype(float)

adaboost=AdaBoostClassifier(n_estimators=10, base_estimator=LinearSVC())
# adaboost=AdaBoostClassifier(n_estimators=20, base_estimator=LinearSVC())
adaboost.fit(x_train,y_train)
pred = adaboost.predict(x_test)
print("Accuracy :",accuracy_score(y_test,pred))


