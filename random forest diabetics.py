# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:28:47 2024

@author: rajendra
"""

import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv("Diabetes.csv")
data.head()

df=pd.DataFrame(data)
df.head()

# Features and target variable
features = df.drop(df.columns[-1], axis=1)
target = df[df.columns[-1]]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create and train the Random Forest model
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, predictions)
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
