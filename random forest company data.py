"""

1.	Business Problem
1.1.	What is the business objective?
1.1.	Are there any constraints?

Objective: Understand and predict factors influencing income levels to assist in targeted marketing strategies.
Constraints: Limited computational resources, potential class imbalance in the target variable.




2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image:

Feature	Data Type	Description	Relevant to Model
Undergrad	Categorical	Educational qualification (Yes/No)	Yes
Marital.Status	Categorical	Marital status (e.g., Single, Married, Divorced)	Yes
Taxable.Income	Numeric	Taxable income level	Yes
City.Population	Numeric	Population of the city	Yes
Work.Experience	Numeric	Number of years of work experience	Yes
Urban	Categorical	Urban or non-urban residency	Yes







2.1 Make a table as shown above and provide information about the features such as its data type and its relevance to the model building. And if not relevant, provide reasons and a description of the feature.

3.	Data Pre-processing
3.1 Data Cleaning, Feature Engineering, etc.

4.	Exploratory Data Analysis (EDA):
4.1.	Summary.
4.2.	Univariate analysis.
4.3.	Bivariate analysis.

5.	Model Building
5.1	Build the model on the scaled data (try multiple options).
5.2	Perform Decision Tree and Random Forest on the given datasets.
5.3	Train and Test the data and perform cross validation techniques, compare accuracies, precision and recall and explain about them.
5.4	Briefly explain the model output in the documentation. 

		 
6.	Write about the benefits/impact of the solution - in what way does the business (client) benefit from the solution provided?

Targeted marketing strategies based on predicted income levels.
Resource optimization through more efficient marketing spending.
Improved understanding of factors influencing income levels.
Enhanced decision-making for business planning.


"""




# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Read the CSV file
df = pd.read_csv('Company_Data.csv')

# Display column names and data types
df.columns
df.dtypes

# Assuming 'Income' is the column you want to predict
X = df.drop('Income', axis=1)  # Use axis=1 to drop the 'Income' column
y = df['Income']

# Perform label encoding for categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for column in categorical_columns:
    X[column] = label_encoder.fit_transform(X[column])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a Random Forest Classifier with 20 trees
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)

# Evaluate the model on the test set and print accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Make predictions on the test set
y_predicted = model.predict(X_test)

# Generate and print the confusion matrix
cm = confusion_matrix(y_test, y_predicted)
print("Confusion Matrix:\n", cm)

# Plot the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['NO', 'YES'], yticklabels=['NO', 'YES'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
