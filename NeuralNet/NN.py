import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance


data_raw = pd.read_csv("../Data/data2.tsv", sep="\t", index_col=0)
data_XY = data_raw.drop(columns=data_raw.columns[0:1])
data_XY = data_XY.drop(columns=data_XY.columns[-1:])
X = data_XY.drop(columns=data_XY.columns[0:1])
Y = data_XY.drop(columns=data_XY.columns[1:])

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42)

#print(X_train)
#print(Y_train)

# Train the neural network with all features
nn = MLPClassifier(max_iter=1000, random_state=1, early_stopping=True, hidden_layer_sizes=(
    100, 50, 30), activation='tanh', solver='adam', alpha=0.001, learning_rate='constant')
nn.fit(X_train, Y_train.values.ravel())

#y_pred = nn.predict(X_test)
#accuracy = accuracy_score(Y_test, y_pred)
#print(f'Accuracy: {accuracy}')  # Accuracy: 0.9986666666666667

# Determine feature importance
importance = permutation_importance(nn, X_test, Y_test, n_repeats=30)

#print("Importance:")
#print(importance)

# Get the feature importances and their corresponding standard deviations
importances = importance['importances_mean']
std = importance['importances_std']
sorted_indices = np.argsort(importances)[::-1]

# Determine the number of features you want to select, say X best features
X_top_features = 3  # For example, if you want to select the top 10 features

# Get the names of the top X most important features
top_X_feature_names = X_train.columns[sorted_indices][:X_top_features]

# Print out the names and importance of the top X features
for i in range(X_top_features):
    print(
        f"{top_X_feature_names[i]}: {importances[sorted_indices[i]]} +/- {std[sorted_indices[i]]}")

# Length: 0.33054141414141414 +/- 0.004145322619831608
# Numbers: 0.10765252525252529 +/- 0.0023344446170851275
# Uppers: 0.07904242424242429 +/- 0.0018460755950232776
# FCT_Lo: 0.06306666666666669 +/- 0.0019346279823413504
# FCT_Nu: 0.05582626262626267 +/- 0.0020235971470955573
# Lowers: 0.05339797979797982 +/- 0.002277311650215885
# LCT_Lo: 0.04742626262626268 +/- 0.0017012740024178676
# LCT_Nu: 0.04640808080808086 +/- 0.0017648011241656373
# SLG_numbers: 0.02227070707070711 +/- 0.001707595400805405
# SLG_uppers: 0.01908686868686873 +/- 0.001347121847386852


# Select only the top X features for training and testing
X_train_reduced = X_train[top_X_feature_names]
X_test_reduced = X_test[top_X_feature_names]

# Retrain the neural network with the top X features
nn_reduced = MLPClassifier(max_iter=1000, random_state=1, early_stopping=True, hidden_layer_sizes=(100, 50, 30),
                           activation='tanh', solver='adam', alpha=0.001, learning_rate='constant')

# Fit the model to the reduced training data
nn_reduced.fit(X_train_reduced, Y_train.values.ravel())

# Predict the outputs on the reduced testing data
y_pred_reduced = nn_reduced.predict(X_test_reduced)

# Evaluate the accuracy
accuracy_reduced = accuracy_score(Y_test, y_pred_reduced)
print(f'Reduced Model Accuracy: {accuracy_reduced}')
# Reduced Model Accuracy: 0.9993939393939394
