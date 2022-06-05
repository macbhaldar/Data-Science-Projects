import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = sns.load_dataset('penguins')
df.head(10)

df.describe().T
df.isnull().sum()

df.describe(include='all')

df.corr()
sns.heatmap(df.corr(), cmap= 'Wistia', annot=True);

# Histogram
df.hist(figsize=(12,8));
plt.show()

# Box plot
df.plot(kind= 'box', subplots=True, layout=(3,2), sharex=False, sharey= False , figsize=(8,12))
plt.show()

# Pairplots
sns.pairplot(data=df, hue='species',palette='mako');

# Violin Plot
ax = sns.violinplot(x= 'sex', y= 'flipper_length_mm', data=df, size=8)
ax = sns.violinplot(x= 'species', y= 'body_mass_g', data=df, size=8, palette='YlGn');

# Value Counts
df.sex.value_counts()
sns.countplot(data=df, x='sex', palette='summer');

df.species.value_counts()
sns.countplot(data=df, x='species', palette='YlOrRd');

df.island.value_counts()
sns.countplot(data=df, x='island', palette='RdPu');

sns.countplot(data= df, x='sex', palette='rocket', hue='species');
sns.countplot(data= df, x= 'island', hue='species', palette='husl');
sns.countplot(data= df, x= 'island', hue='sex', palette='spring');

# Drop The Null Values
df.isnull().sum()
df.dropna(inplace=True)
df.head()

# 
df1 = df.copy() # Create a copy of the dataset
df1.head()

Y = df1.species
Y.head()

Y.unique()

Y = Y.map({'Adelie': 0, 'Chinstrap':1, 'Gentoo':2})
Y.head()

# Drop the species column from df1
df1.drop('species', inplace=True, axis=1)
df1.head()

# One Hot Encoding on sex variable
pd.get_dummies(df1['sex']).head()
se=pd.get_dummies(df1['sex'], drop_first=True) 
se.head(3)

# Label Encoding
df1.island.unique()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1['island']= le.fit_transform(df1['island']) 
df1['island']
df1.head()

# Concatenation
df2 = pd.concat([df1, se], axis=1)
df2.head()

# drop the sex column
df2.drop('sex', axis=1, inplace=True);
df2.head()

X = df2

# Split data to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=40)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

model1 = rfc.fit(X_train, y_train)
prediction1 = model1.predict(X_test)

print("Acc on training data: {:,.3f}".format(rfc.score(X_train, y_train)))
print("Acc on test data: {:,.3f}".format(rfc.score(X_test, y_test)))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print(confusion_matrix(y_test, prediction1))
print(classification_report(y_test, prediction1))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
models=[]

models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('rfc', RandomForestClassifier()))
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# evaluate each model
results =[]
names=[]
for name , model in models:
    kfold=KFold(n_splits=10, random_state=40)
    cv_results= cross_val_score(model, X_train, y_train, cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    
    msg= '%s:, %f, (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
# make predictions on test datasets
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
predictions= rfc.predict(X_test)
print(accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

NB = GaussianNB()
NB.fit(X_train, y_train)
predcitions = NB.predict(X_test)
print(accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)
predictions= KNN.predict(X_test)
print(accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
