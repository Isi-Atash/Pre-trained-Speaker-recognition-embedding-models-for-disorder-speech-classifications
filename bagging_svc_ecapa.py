import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

ds_org_hun = pd.read_csv('Ecapa_embedding_hun.csv')
ds_org_dutch = pd.read_csv('Ecapa_embedding_dutch.csv')

ds_hun = ds_org_hun[(ds_org_hun.label == 'OD') | (ds_org_hun.label == 'HC')]
ds_dutch = ds_org_dutch[(ds_org_dutch.label == 'OD') | (ds_org_dutch.label == 'HC')]

ds_hun = ds_hun.drop(['filename', 'grade'], axis=1)
ds_dutch = ds_dutch.drop(['filename', 'grade'], axis=1)

labelencoder = LabelEncoder()
ds_hun['label'] = labelencoder.fit_transform(ds_hun['label'])
ds_dutch['label'] = labelencoder.fit_transform(ds_dutch['label'])

# Scaling features to 0-1 range
scaler = MinMaxScaler()
ds_hun = scaler.fit_transform(ds_hun)
ds_dutch = scaler.fit_transform(ds_dutch)

ds_hun = pd.DataFrame(ds_hun)
ds_dutch = pd.DataFrame(ds_dutch)

# split the data to X and y
X_hun = ds_hun.iloc[:, 1:192]
y_hun = ds_hun.iloc[:, 193]
X_dutch = ds_dutch.iloc[:, 1:192]
y_dutch = ds_dutch.iloc[:, 193]
X = pd.concat([X_hun, X_dutch])
y = pd.concat([y_hun, y_dutch])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)
# M-H-D
train_x = X_dutch
train_y = y_dutch
test_x = X_hun
test_y = y_hun

from sklearn.model_selection import GridSearchCV
C_range = np.logspace(-2, 2, 50)
gamma_range = np.logspace(-9, 3, 50)
param_grid = {'C': C_range, 'gamma': gamma_range, 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(train_x, train_y)
grid_predictions = grid.predict(test_x)
print("Accuracy:", metrics.accuracy_score(test_y, grid_predictions))
# print the best parameters
print(grid.best_params_)
# print the best estimator
print(grid.best_estimator_)
# print the best score
print(grid.best_score_)

#use the best parameters
from sklearn.ensemble import BaggingClassifier

bag_model = BaggingClassifier(base_estimator=SVC(kernel='rbf', C=grid.best_estimator_.C, gamma=grid.best_estimator_.gamma),
                              n_estimators=100,
                              random_state=0,
                              max_samples=0.8).fit(train_x, train_y)

prediction = bag_model.predict(test_x)

#confusion matrix
cm = confusion_matrix(test_y, prediction)
#acurracy score round to 2 decimals
acc = round(metrics.accuracy_score(test_y, prediction), 2)
print('Accuracy:', acc)
#senstivity round to 2 decimals
sens = round(cm[0, 0] / (cm[0, 0] + cm[0, 1]), 2)
print('Sensitivity:', sens)
#specificity round to 2 decimals
spec = round(cm[1, 1] / (cm[1, 0] + cm[1, 1]), 2)
print('Specificity:', spec)
#f1 score round to 2 decimals
f1 = round(metrics.f1_score(test_y, prediction), 2)
print('F1 score:', f1)