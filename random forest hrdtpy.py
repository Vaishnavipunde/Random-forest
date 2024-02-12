# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:08:21 2024

@author: rajendra
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import LabelEncoder

# Read the CSV file
df = pd.read_csv('HR_DT.csv')
df.columns
df.dtypes

# Assuming 'target' is the column you want to predict
X = df.drop(' monthly income of employee', axis=1)  # Use axis=1 to drop the 'target' column
y = df[' monthly income of employee']

# Perform label encoding for categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for column in categorical_columns:
    X[column] = label_encoder.fit_transform(X[column])

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
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['NO', 'YES'], yticklabels=['NO', 'YES'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()



