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
X = data_XY.drop(columns=data_XY.columns[0:1])#Drop strength
y = data_XY.drop(columns=data_XY.columns[1:])#drop everything else except strength

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=43)

clf = DecisionTreeClassifier(random_state=47, criterion="entropy")
clf2 = LogisticRegression(random_state=472, solver="newton-cholesky")
clf3 = GaussianNB()
clf4 = KNeighborsClassifier(n_neighbors=2)


clf.fit(X_train, y_train)
clf2.fit(X_train, y_train.values.ravel())
clf3.fit(X_train, y_train.values.ravel())
clf4.fit(X_train, y_train.values.ravel())

#resultLogistic = permutation_importance(clf4, X_test, y_test, n_repeats=10, random_state=72)

# importanceRanking = {k: v for k, v in sorted(dict(enumerate(resultLogistic.importances_mean, 0)).items(), key=lambda item: item[1], reverse=True)}
# x = 0
# for i in importanceRanking:
#     c = data_XY.columns[i+1]
#     print(c + " has importance " + str(importanceRanking[i]))
#     x+=1

print("Score with all features: ")
print("DT: " + str(clf.score(X_test, y_test)))
print("LR: " + str(clf2.score(X_test, y_test)))
print("GNB: " + str(clf3.score(X_test, y_test)))
print("KNN: " + str(clf4.score(X_test, y_test)))

print("Score with top three features for that classifier dropped: ")

X_trainO = pd.DataFrame(X_train, columns=['Length','#Uppers','#Lowers','#Numbers','#Symbols','SLG_uppers','SLG_lowers','SLG_numbers','SLG_symbols','FCT_Up','FCT_Lo','FCT_Nu','FCT_Sy','LCT_Up','LCT_Lo','LCT_Nu','LCT_Sy','#words','ave_size_of_words','palindrome'])
X_testO = pd.DataFrame(X_test, columns=['Length','#Uppers','#Lowers','#Numbers','#Symbols','SLG_uppers','SLG_lowers','SLG_numbers','SLG_symbols','FCT_Up','FCT_Lo','FCT_Nu','FCT_Sy','LCT_Up','LCT_Lo','LCT_Nu','LCT_Sy','#words','ave_size_of_words','palindrome'])

#'Length','#Uppers','#Lowers','#Numbers','#Symbols','SLG_uppers','SLG_lowers','SLG_numbers','SLG_symbols','FCT_Up','FCT_Lo','FCT_Nu','FCT_Sy','LCT_Up','LCT_Lo','LCT_Nu','LCT_Sy','#words','ave_size_of_words','palindrome'
X_train1 = X_trainO.drop(columns=['#Uppers','#Symbols','SLG_uppers','SLG_lowers','SLG_numbers','SLG_symbols','FCT_Up','FCT_Lo','FCT_Nu','FCT_Sy','LCT_Up','LCT_Lo','LCT_Nu','LCT_Sy','#words','ave_size_of_words','palindrome'])
X_test1 = X_testO.drop(columns=['#Uppers','#Symbols','SLG_uppers','SLG_lowers','SLG_numbers','SLG_symbols','FCT_Up','FCT_Lo','FCT_Nu','FCT_Sy','LCT_Up','LCT_Lo','LCT_Nu','LCT_Sy','#words','ave_size_of_words','palindrome'])
clf.fit(X_train, y_train)
print("DT without the worthless features: " + str(clf.score(X_test, y_test)))

X_train2 = X_trainO.drop(columns=['#Lowers','#Symbols','SLG_uppers','SLG_lowers','SLG_numbers','SLG_symbols','FCT_Up','FCT_Lo','FCT_Nu','FCT_Sy','LCT_Up','LCT_Lo','LCT_Nu','LCT_Sy','#words','ave_size_of_words','palindrome'])
X_test2 = X_testO.drop(columns=['#Lowers','#Symbols','SLG_uppers','SLG_lowers','SLG_numbers','SLG_symbols','FCT_Up','FCT_Lo','FCT_Nu','FCT_Sy','LCT_Up','LCT_Lo','LCT_Nu','LCT_Sy','#words','ave_size_of_words','palindrome'])
clf2.fit(X_train2, y_train.values.ravel())
print("LR without the worthless features: " + str(clf2.score(X_test2, y_test)))

X_train3 = X_trainO.drop(columns=['Length','#Lowers','#Numbers','#Symbols','SLG_uppers','SLG_lowers','SLG_numbers','SLG_symbols','FCT_Lo','FCT_Nu','FCT_Sy','LCT_Lo','LCT_Nu','LCT_Sy','#words','ave_size_of_words','palindrome'])
X_test3 = X_testO.drop(columns=['Length','#Lowers','#Numbers','#Symbols','SLG_uppers','SLG_lowers','SLG_numbers','SLG_symbols','FCT_Lo','FCT_Nu','FCT_Sy','LCT_Lo','LCT_Nu','LCT_Sy','#words','ave_size_of_words','palindrome'])
clf3.fit(X_train3, y_train.values.ravel())
print("GNB without the worthless features: " + str(clf3.score(X_test3, y_test)))

X_train4 = X_trainO.drop(columns=['#Uppers','#Symbols','SLG_uppers','SLG_lowers','SLG_numbers','SLG_symbols','FCT_Up','FCT_Lo','FCT_Nu','FCT_Sy','LCT_Up','LCT_Lo','LCT_Nu','LCT_Sy','#words','ave_size_of_words','palindrome'])
X_test4 = X_testO.drop(columns=['#Uppers','#Symbols','SLG_uppers','SLG_lowers','SLG_numbers','SLG_symbols','FCT_Up','FCT_Lo','FCT_Nu','FCT_Sy','LCT_Up','LCT_Lo','LCT_Nu','LCT_Sy','#words','ave_size_of_words','palindrome'])
clf4.fit(X_train4, y_train.values.ravel())
print("KNN without the worthless features: " + str(clf4.score(X_test4, y_test)))
