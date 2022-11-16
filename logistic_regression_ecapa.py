import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# 2 datasets, 1 for Hungary, 1 for Dutch
ds_org_hun = pd.read_csv('Ecapa_embedding_hun.csv')
ds_org_dutch = pd.read_csv('Ecapa_embedding_dutch.csv')

# Normalize the data
# 1st, we need to save only OD and HC labels from the dataset
# 2nd, we need to label_encoder the labels

# find OD and HC from ds and create a new dataset
ds_hun = ds_org_hun[(ds_org_hun.label == 'OD') | (ds_org_hun.label == 'HC')]
ds_dutch = ds_org_dutch[(ds_org_dutch.label == 'OD') | (ds_org_dutch.label == 'HC')]

# drop filename and grade columns
ds_hun = ds_hun.drop(['filename', 'grade'], axis=1)
ds_dutch = ds_dutch.drop(['filename', 'grade'], axis=1)

# label_encoder OD and HC to 0 and 1
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

# param_grid = {'C': [0.1, 1, 10, 100, 1000],
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#               'kernel': ['rbf']}
# # create a grid search object
# grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
# grid.fit(X_hun, y_hun)
# grid_predictions = grid.predict(X_dutch)
# # print the results
#
# print(confusion_matrix(y_dutch, grid_predictions))
# print(classification_report(y_dutch, grid_predictions))
# print("Accuracy:", metrics.accuracy_score(y_dutch, grid_predictions))


#logisic regression
logreg = LogisticRegression()
logreg.fit(X_hun, y_hun)
y_pred = logreg.predict(X_dutch)
print("Accuracy:", metrics.accuracy_score(y_dutch, y_pred))