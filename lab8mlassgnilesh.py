

import numpy as np
import pandas as pd
df = pd.read_csv("lab-8.csv")
df.head()

df.describe()

df.info()

def numeric_value(value):

  if value == "Yes":

    return 1
  if value == "No":
    
    return 0

df['attrition'] = df['Attrition'].apply(numeric_value)
df = df.drop('Attrition', 1)
df.head()



df = df.drop(['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','Over18'], axis = 1)

from pandas.plotting import scatter_matrix
scatter_matrix(df, alpha=0.9, figsize=(25,25), diagonal='hist')

df.head()

df.info()

pair_corr_coeff = df.corr()
print(pair_corr_coeff)
#pair_corr_coeff.abs().style.background_gradient()
import matplotlib.pyplot as plt
plt.matshow(np.abs(pair_corr_coeff))
plt.colorbar()
plt.xticks(range(len(pair_corr_coeff.columns)), pair_corr_coeff.columns, rotation='vertical');
plt.yticks(range(len(pair_corr_coeff.columns)), pair_corr_coeff.columns);

X = df.iloc[:,0:26].values
print(X)

y = df.iloc[:,26].values
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Logistic reg
from sklearn.linear_model import LogisticRegression
model_1= LogisticRegression(random_state = 0)
model_1.fit(X_train, y_train)
y_pred = model_1.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

y_pred = model_1.predict(X_train)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
accuracy_score(y_train, y_pred)

from sklearn.metrics import f1_score
f1_score(y_train, y_pred, average='macro')

f1_score(y_train, y_pred, average='micro')

from sklearn.metrics import precision_score
precision_score(y_train, y_pred, average='macro')

precision_score(y_train, y_pred, average='micro')

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
y_p1 = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_p1)
print(cm)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_p1)

from sklearn.metrics import f1_score
f1_score(y_test, y_p1, average='macro')

precision_score(y_test, y_p1, average='macro')

#KNN
from sklearn.neighbors import KNeighborsClassifier
classifier1 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier1.fit(X_train, y_train)

y_p2 = classifier1.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_p2)
print(cm)

accuracy_score(y_test, y_p2)

f1_score(y_test, y_p2, average='macro')

precision_score(y_test, y_p2, average='macro')