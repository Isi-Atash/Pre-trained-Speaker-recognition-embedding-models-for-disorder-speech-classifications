import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# get th data from csv
# 2 datasets, 1 for Hungary, 1 for Dutch
ds_org_hun = pd.read_csv('Ecapa_embedding_hun.csv')
ds_org_dutch = pd.read_csv('Ecapa_embedding_dutch.csv')

ds_hun = ds_org_hun[(ds_org_hun.label == 'OD') | (ds_org_hun.label == 'HC')]
ds_dutch = ds_org_dutch[(ds_org_dutch.label == 'OD') | (ds_org_dutch.label == 'HC')]

ds_hun = ds_hun.drop(['filename', 'label'], axis=1)
ds_dutch = ds_dutch.drop(['filename', 'label'], axis=1)

scaler = MinMaxScaler()
ds_hun = scaler.fit_transform(ds_hun)
ds_dutch = scaler.fit_transform(ds_dutch)

ds_hun = pd.DataFrame(ds_hun)
ds_dutch = pd.DataFrame(ds_dutch)

X_hun = ds_hun.iloc[:, 1:192]
y_hun = ds_hun.iloc[:, 193]
X_dutch = ds_dutch.iloc[:, 1:192]
y_dutch = ds_dutch.iloc[:, 193]

# Linear regression grade prediction
model = LinearRegression().fit(X_hun, y_hun)
# y_pred = model.predict(X_dutch)

# The coefficients
print('Coefficients: \n', model.coef_)

print(f"intercept: {model.intercept_}")
print('Coefficient of determination: %.2f'
      % model.score(X_hun, y_hun))


