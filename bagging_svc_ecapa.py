import pandas as pd
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

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier

bag_model = BaggingClassifier(base_estimator= SVC(kernel='rbf', C=1000, gamma=0.001),
                              n_estimators=100,
                              random_state=0,
                              max_samples=0.8)

bag_model.fit(X_dutch, y_dutch)

# prediction accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

prediction = bag_model.predict(X_hun)
test = y_hun
# confusion matrix
cm = confusion_matrix(test, prediction)
# accuracy score round to 2 decimals
acc = round(accuracy_score(test, prediction), 2)
print('Accuracy:', acc)
# sensitivity round to 2 decimals
sens = round(cm[0, 0] / (cm[0, 0] + cm[0, 1]), 2)
print('Sensitivity:', sens)
# specificity round to 2 decimals
spec = round(cm[1, 1] / (cm[1, 0] + cm[1, 1]), 2)
print('Specificity:', spec)
# f1 score round to 2 decimals
f1 = round(f1_score(test, prediction), 2)
print('F1 score:', f1)
