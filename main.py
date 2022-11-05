import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


#get th data from csv
#2 datasets, 1 for Hungary, 1 for Dutch
ds_org_hun = pd.read_csv('Xvector_embedding_hun.csv')
ds_org_dutch = pd.read_csv('Xvector_embedding_dutch.csv')

#Normalize the data
#1st, we need to save only OD and HC labels from the dataset
#2nd, we need to label_encoder the labels

#find OD and HC from ds and create a new dataset
ds_hun = ds_org_hun[(ds_org_hun.label == 'OD') | (ds_org_hun.label == 'HC')]
ds_dutch = ds_org_dutch[(ds_org_dutch.label == 'OD') | (ds_org_dutch.label == 'HC')]

#drop filename and grade columns
ds_hun = ds_hun.drop(['filename','grade'], axis=1)
ds_dutch = ds_dutch.drop(['filename','grade'], axis=1)

#label_encoder OD and HC to 0 and 1
labelencoder = LabelEncoder()
ds_hun['label'] = labelencoder.fit_transform(ds_hun['label'])
ds_dutch['label'] = labelencoder.fit_transform(ds_dutch['label'])

#Scaling features to 0-1 range
scaler = MinMaxScaler()
ds_hun = scaler.fit_transform(ds_hun)
print(ds_hun)

