import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split

# read from cvs
ds_hun = pd.read_csv('Xvector_embedding_hun.csv')


X_train, X_test, y_train,y_test = train_test_split(ds_hun.iloc[:, :511] .values, ds_hun.label, test_size=0.3, random_state=0)

clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# print(ds_hun.loc[:, 1:512] .values)

