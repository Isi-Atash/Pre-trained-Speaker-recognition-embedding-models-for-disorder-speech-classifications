import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# get th data from csv
# 2 datasets, 1 for Hungary, 1 for Dutch
ds_org_hun = pd.read_csv('Xvector_embedding_hun.csv')
ds_org_dutch = pd.read_csv('Xvector_embedding_dutch.csv')

ds_hun = ds_org_hun[(ds_org_hun.label == 'OD') | (ds_org_hun.label == 'HC')]
ds_dutch = ds_org_dutch[(ds_org_dutch.label == 'OD') | (ds_org_dutch.label == 'HC')]

ds_hun = ds_hun.drop(['filename', 'label'], axis=1)
ds_dutch = ds_dutch.drop(['filename', 'label'], axis=1)

scaler = MinMaxScaler()
ds_hun = scaler.fit_transform(ds_hun)
ds_dutch = scaler.fit_transform(ds_dutch)

ds_hun = pd.DataFrame(ds_hun)
ds_dutch = pd.DataFrame(ds_dutch)

X_hun = ds_hun.iloc[:, :512]
y_hun = ds_hun.iloc[:, 513]
X_dutch = ds_dutch.iloc[:, :512]
y_dutch = ds_dutch.iloc[:, 513]
X = pd.concat([X_hun, X_dutch])
y = pd.concat([y_hun, y_dutch])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.8, random_state=0)

# Linear regression grade prediction
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X_train, y_train)

#root_mean_squared_error round to 2 decimals
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, model.predict(X_test)))
print("Root mean squared error: ", round(rms, 2))

#spearman correlation round to 2 decimals
from scipy.stats import spearmanr

corr, _ = spearmanr(y_test, model.predict(X_test))

print("Spearman correlation: ", round(corr, 4))

#pearson correlation round to 2 decimals
from scipy.stats import pearsonr

corr, _ = pearsonr(y_test, model.predict(X_test))

print("Pearson correlation: ", round(corr, 4))



