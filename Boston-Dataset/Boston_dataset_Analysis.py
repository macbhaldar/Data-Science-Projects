import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston

boston =load_boston()

boston.keys()
boston.data
boston.target
boston

boston.feature_names

print(boston.DESCR)

boston.filename

df= pd.DataFrame(boston.data, columns=boston.feature_names)
df
df.shape

df['Target']= boston.target
df.head()
df.dtypes

# check null values in the data
df.isnull().sum()

# Basic Stats of data
df.describe()

# Find the correlation between target and other features
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),annot=True)

'''
# Target is positively corelated with ZN, rooms per dwelling(RM), 
# Proportion of blacks in town(B), distance from employment centres(DIS) and CHAS
we infer that with rise in any of these features the target or prices of the houses will increase.
'''
 
''' 
# Target is negatively corelated with crime(CRIM),INDUS,pollution (NOX),
# Age,RAD,TAX,Parent Teacher ratio (PTRATIO) and Lower status of population (LSTAT)
with the increase in any of the above features the prices of the houses will decrease
'''

'''
we will apply Linear Regression since we have to predict prices, 
since train and test data are not given separately, 
we split the data into train and test data.
the test data will be predicted prices, 
train data we will train it and fit it to be able to predict approximate prices
'''

df.columns

# index of accessibility to radial highways
df['RAD'].value_counts()
df['B'].value_counts()
df['RM'].value_counts()
df['CRIM'].value_counts()

# Visualizing data

# Histogram
plt.figure(figsize=(12,6))
plt.hist(df['CRIM'], color='b');
plt.xlabel('Crime')
plt.ylabel('Frequency')

# Violin plot
sns.violinplot(df['RAD'], data=df,palette='rocket')

# Count plot
sns.countplot(df['TAX'], data=df)
plt.xticks(rotation=90);
plt.figure(figsize=(30,10))
plt.xticks(fontsize=17)

sns.distplot(df['RM'], color='r')
sns.distplot(df['Target'])

sns.countplot(df['Target'],data=df,palette='cubehelix')
sns.countplot('CHAS',data=df, palette='husl')

sns.jointplot(x='RM', y='Target', data=df,kind='hex',color='g')
sns.jointplot(x='PTRATIO',y='Target',data=df,kind='reg',color='r')
sns.jointplot(x='B',y='Target', kind='hex', color='m',data=df)
sns.jointplot(x='NOX',y='Target',data=df,kind='kde',color='Y')

sns.boxplot('Target',data=df)

sns.relplot(x="Target", y="RAD", kind="line", data=df)
sns.regplot(x='DIS', y='Target',data=df)

# Training the model

import sklearn
from sklearn.model_selection import train_test_split

X=df.drop('Target', axis=1)
y= df['Target']

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.33,random_state=5)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Linear Regression

from sklearn.linear_model import LinearRegression

lin_reg= LinearRegression()
lin_reg.fit(X_train, y_train)
predictions= lin_reg.predict(X_test)

plt.scatter(y_test, predictions)
plt.xlabel('Prices')
plt.ylabel('Predicted Prices')
plt.title('Prices vs Predicted prices')

lin_reg.score(X_test, y_test)

'''
# The score comes out to be 69.56% indicates that the accuracy of the model is low.
'''

error= y_test-predictions
sns.distplot(error)

mse= sklearn.metrics.mean_squared_error(y_test, predictions)
print(mse)

'''
# The mean squared error tells you how close a regression line is to a set of points.
# the lower mse the better the model
# we have a mse of 28.5 which is more i.e it is not a great model.
'''
