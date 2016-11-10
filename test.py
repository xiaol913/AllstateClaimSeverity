import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv('train.csv')

# print train_data.head()

# print("Number of missing values", train_data.isnull().sum().sum())

# print train_data.describe()

contFeatureslist = []
for colName, x in train_data.iloc[1, :].iteritems():
    if (not str(x).isalpha()):
        contFeatureslist.append(colName)

contFeatureslist.remove("id")
contFeatureslist.remove("loss")

# print(contFeatureslist)

contFeatureslist.append("loss")

catFeatureslist = []
for colName,x in train_data.iloc[1,:].iteritems():
    if(str(x).isalpha()):
        catFeatureslist.append(colName)

# print(train_data[catFeatureslist].apply(pd.Series.nunique))

for cf1 in catFeatureslist:
    le = LabelEncoder()
    le.fit(train_data[cf1].unique())
    train_data[cf1] = le.transform(train_data[cf1])

print train_data.head(5)