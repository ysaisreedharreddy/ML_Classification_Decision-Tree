# Import necessary libraries for data handling and visualization
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset from a CSV file into a pandas DataFrame. This dataset likely includes information such as Age and Estimated Salary, which are used to predict whether an individual has purchased a product.
dataset = pd.read_csv(r"C:\Users\prasu\DS2\git\classification\5. Decision tree\5. DECESSION TREE CODE\Social_Network_Ads.csv")

# Extract features (columns 2 and 3) and the target variable (last column) from the dataset. The features represent attributes like Age and Estimated Salary, while the target indicates purchasing behavior.
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Split the dataset into training and testing sets to prepare for model training and evaluation. 20% of the data is reserved for testing the model's performance.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Decision trees do not require feature scaling to perform well; thus, this step is optional and commented out.

# Instantiate and train a Decision Tree classifier. This model will be used to predict the target variable based on the input features.
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()  # Initialize the Decision Tree classifier
classifier.fit(X_train, y_train)       # Fit the classifier on the training data

# Use the trained model to predict the outcomes for the test set.
y_pred = classifier.predict(X_test)  # Predicting the labels for the test set

# Create a confusion matrix to evaluate the predictions against the actual outcomes. The matrix helps in visualizing the performance of the classification model.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculate the accuracy of the model on the test data to understand how often the model is correct.
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print('accuracy: ', ac)

# Display the model's accuracy on the training data (Bias) and on the test data (Variance) to assess overfitting or underfitting.
bias = classifier.score(X_train, y_train)
variance = classifier.score(X_test, y_test)
print('Bias (Training Score):', bias)
print('Variance (Test Score):', variance)
