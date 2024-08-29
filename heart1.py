import streamlit as st

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the datasets
heart_df = pd.read_csv('/workspaces/codespaces-blank/heart.csv')
o2_saturation_df = pd.read_csv('/workspaces/codespaces-blank/o2Saturation.csv')

"""### Initial Data Exploration"""

# Display the first few rows of the heart dataset
heart_df.head()

# Display the first few rows of the O2 saturation dataset
o2_saturation_df.head()

"""### Data Cleaning and Preparation"""

# Check for missing values in heart dataset
heart_df.isnull().sum()

# Check for missing values in O2 saturation dataset
o2_saturation_df.isnull().sum()

"""### Exploratory Data Analysis (EDA)"""

# Plot the distribution of age
plt.figure(figsize=(10, 6))
sns.histplot(heart_df['age'], kde=True, bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Plot the correlation heatmap
numeric_df = heart_df.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

"""### Predictive Modeling"""

# Define features and target variable
X = heart_df.drop('output', axis=1)
y = heart_df['output']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
accuracy, conf_matrix, class_report

"""### Conclusion and Future Work

In this notebook, we explored the heart attack analysis and prediction dataset. We performed initial data exploration, cleaning, and visualization. We also built a predictive model using Logistic Regression and evaluated its performance.

For future analysis, it would be interesting to:
- Explore other machine learning models and compare their performance
- Perform feature engineering to create new features that might improve the model
- Investigate the impact of O2 saturation levels on heart attack prediction

What do you think would be useful to explore next? Let me know in the comments.

## Credits
This notebook was created with the help of [Devra AI data science assistant](https://devra.ai/ref/kaggle)
"""