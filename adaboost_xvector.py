import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

ds_org_hun = pd.read_csv('Xvector_embedding_hun.csv')
ds_org_dutch = pd.read_csv('Xvector_embedding_dutch.csv')

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
X_hun = ds_hun.iloc[:, :512]
y_hun = ds_hun.iloc[:, 513]
X_dutch = ds_dutch.iloc[:, :512]
y_dutch = ds_dutch.iloc[:, 513]

# merge the datasets
X = pd.concat([X_hun, X_dutch])
y = pd.concat([y_hun, y_dutch])

#adaboost with svc
from sklearn.ensemble import AdaBoostClassifier
#decision tree
from sklearn.tree import DecisionTreeClassifier

ada = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=100, learning_rate=1)
ada.fit(X_hun, y_hun)

print('H Adaboost score: ', ada.score(X_hun, y_hun))
print('D Adaboost score: ', ada.score(X_dutch, y_dutch))
print('H+D Adaboost score: ', ada.score(X, y))

#cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(ada, X, y, cv=10)
print('Cross validation scores: ', scores)
print('Cross validation mean score: ', scores.mean())






