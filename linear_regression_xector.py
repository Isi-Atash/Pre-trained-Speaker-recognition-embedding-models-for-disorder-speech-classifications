import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

# get th data from csv
# 2 datasets, 1 for Hungary, 1 for Dutch
ds_org_hun = pd.read_csv('Xvector_embedding_hun.csv')
ds_org_dutch = pd.read_csv('Xvector_embedding_dutch.csv')

ds_hun = ds_org_hun[(ds_org_hun.label == 'OD') | (ds_org_hun.label == 'HC')]
ds_dutch = ds_org_dutch[(ds_org_dutch.label == 'OD') | (ds_org_dutch.label == 'HC')]

ds_hun = ds_hun.drop(['filename', 'label'], axis=1)
ds_dutch = ds_dutch.drop(['filename', 'label'], axis=1)

# scaler = MinMaxScaler()
# ds_hun = scaler.fit_transform(ds_hun)
# ds_dutch = scaler.fit_transform(ds_dutch)
#
# ds_hun = pd.DataFrame(ds_hun)
# ds_dutch = pd.DataFrame(ds_dutch)

X_hun = ds_hun.iloc[:, :513]
y_hun = ds_hun.iloc[:, 513]
X_dutch = ds_dutch.iloc[:, :513]
y_dutch = ds_dutch.iloc[:, 513]
X = pd.concat([X_hun, X_dutch])
y = pd.concat([y_hun, y_dutch])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.8, random_state=0)

#scale X values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = pd.DataFrame(sc.fit_transform(X_train))
X_test = pd.DataFrame(sc.transform(X_test))
X_hun = pd.DataFrame(sc.transform(X_hun))
X_dutch = pd.DataFrame(sc.transform(X_dutch))

#M-D-H
train_x =X_hun
train_y = y_hun
test_x = X_dutch
test_y = y_dutch

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

C_range = np.logspace(-2, 2, 50)
gamma_range = np.logspace(-9, 3, 50)
param_grid = {'C': C_range, 'gamma': gamma_range, 'kernel': ['rbf']}
grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=3)
grid.fit(train_x, train_y)
grid_predictions = grid.predict(test_x)
# print the best parameters
print(grid.best_params_)
# print the best estimator
print(grid.best_estimator_)
# print the best score
print(grid.best_score_)

# SVR grade prediction
model = SVR(kernel='rbf', C =grid.best_estimator_.C, gamma= grid.best_estimator_.gamma).fit(train_x, train_y)

#root_mean_squared_error round to 2 decimals
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(test_y, model.predict(test_x)))
print("Root mean squared error: ", round(rms, 2))

#spearman correlation round to 2 decimals
from scipy.stats import spearmanr

corr, _ = spearmanr(test_y, model.predict(test_x))

print("Spearman correlation: ", round(corr, 2))

#pearson correlation round to 2 decimals
from scipy.stats import pearsonr

corr, _ = pearsonr(test_y, model.predict(test_x))

print("Pearson correlation: ", round(corr, 2))



