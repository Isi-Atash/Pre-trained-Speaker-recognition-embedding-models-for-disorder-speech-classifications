import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor


ds_org_hun = pd.read_csv('Xvector_embedding_hun.csv')
ds_org_dutch = pd.read_csv('Xvector_embedding_dutch.csv')

ds_hun = ds_org_hun[(ds_org_hun.label == 'OD') | (ds_org_hun.label == 'HC')]
ds_dutch = ds_org_dutch[(ds_org_dutch.label == 'OD') | (ds_org_dutch.label == 'HC')]

ds_hun = ds_hun.drop(['filename', 'label'], axis=1)
ds_dutch = ds_dutch.drop(['filename', 'label'], axis=1)

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

X = pd.concat([X_hun, X_dutch])
y = pd.concat([y_hun, y_dutch])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.8, random_state=0)


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor

bag_model = BaggingRegressor(LinearRegression(),
                              n_estimators=100,
                              max_samples=0.8,
                              bootstrap=True,
                              oob_score=True).fit(X_dutch,y_dutch)

#root_mean_squared_error round to 2 decimals
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_hun, bag_model.predict(X_hun)))
print("Root mean squared error: ", round(rms, 2))

#spearman correlation round to 2 decimals
from scipy.stats import spearmanr

corr, _ = spearmanr(y_hun, bag_model.predict(X_hun))

print("Spearman correlation: ", round(corr, 4))

#pearson correlation round to 2 decimals
from scipy.stats import pearsonr

corr, _ = pearsonr(y_hun, bag_model.predict(X_hun))

print("Pearson correlation: ", round(corr, 4))