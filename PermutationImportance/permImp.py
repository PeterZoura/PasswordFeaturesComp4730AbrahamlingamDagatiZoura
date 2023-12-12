import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import pandas as pd

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

data_raw = pd.read_csv("../Data/data2.tsv", sep="\t", index_col=0)
data_XY = data_raw.drop(columns=data_raw.columns[0:1])#drop the first column which contains the password itself
data_XY = data_XY.drop(columns=data_XY.columns[-1:])#drop the last column which contains a worthless attribute, t hat is "most common character"
X = data_XY.drop(columns=data_XY.columns[0:2])
y = data_XY.drop(columns=data_XY.columns[1:])

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=43)

clf = DecisionTreeClassifier(random_state=47, criterion="entropy")
clfLog = LogisticRegression(random_state=472, solver="newton-cholesky")
clfNaiveBayes = GaussianNB()
clfKNN = KNeighborsClassifier(n_neighbors=2)


clf.fit(X_train, y_train)
clfLog.fit(X_train, y_train.values.ravel())
clfNaiveBayes.fit(X_train, y_train.values.ravel())
clfKNN.fit(X_train, y_train.values.ravel())

resultLogistic = permutation_importance(clf, X, y, n_repeats=10, random_state=72)

importanceRanking = {k: v for k, v in sorted(dict(enumerate(resultLogistic.importances_mean, 0)).items(), key=lambda item: item[1], reverse=True)}
x = 0
for i in importanceRanking:
    c = data_XY.columns[i+2]
    print(c + " has importance " + str(importanceRanking[i]))
    x+=1