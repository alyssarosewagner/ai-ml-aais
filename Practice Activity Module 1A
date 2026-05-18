import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
boston = fetch_openml(name="boston", version=1, as_frame=True)

# Features (X) and target (y)
X = boston.data
y = boston.target

# View first few rows
print(X.head())

# Basic statistics
print(X.describe())

# Check dataset shape
print("Shape:", X.shape)

plt.hist(y, bins=30)
plt.xlabel("House Prices ($1000's)")
plt.ylabel("Frequency")
plt.title("Distribution of House Prices")
plt.show()

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

plt.scatter(y_test, y_pred, color='green')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# Example: Predict using first 5 rows from test set
sample_data = X_test.iloc[:5]

predictions = model.predict(sample_data)

print("Predicted Prices:", predictions)
