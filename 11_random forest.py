# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:20:47 2024

@author: rajendra
"""

import pandas as pd 
from sklearn.datasets import load_iris
iris=load_iris()
dir(iris)

df=pd.DataFrame(iris.data)
df.head()

df['target']=iris.target
df[0:12]

X=df.drop('target',axis='columns')
y=df.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=20)
#n_estimator or no of trees in the forest
model.fit(X_train,y_train)

model.score(X_test, y_test)
y_predicted=model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_predicted)
cm

import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
plt.xlabel('Predicted')
plt.ylabel('Truth')






