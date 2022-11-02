# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

# read from cdv
ds_hun = pd.read_csv('Xvector_embedding_hun.csv')
ds_dutch = pd.read_csv('Xvector_embedding_dutch.csv')

# get the data
# X_train = ds_hun.iloc[:, :511].values
# y_train = ds_hun.columns['grade'].values
X_train = ds_hun.iloc[:, :511].values
y_train = ds_hun.label.values
X_test = ds_dutch.iloc[:, :511].values
y_test = ds_dutch.label.values

from sklearn import svm

clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))




