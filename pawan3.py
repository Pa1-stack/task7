import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = datasets.load_iris()

# Create a DataFrame with the data
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Display first few rows of the dataset
print(df.head())

# Split the dataset into features (X) and target (y)
X = df.drop('species', axis=1)
y = df['species']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)

# Make predictions using Logistic Regression
y_pred_lr = lr_model.predict(X_test)

# Evaluate the Logistic Regression model
print(f'Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred_lr)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred_lr)}')

# Random Forest Classifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions using Random Forest
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
print(f'Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred_rf)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred_rf)}')

# Plot confusion matrix for Random Forest
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Random Forest')
plt.show()
