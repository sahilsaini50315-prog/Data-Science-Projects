import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("Advertising.csv")

print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nColumns:")
print(df.columns)

print("\nShape:")
print(df.shape)
# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate Rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove Duplicates
df.drop_duplicates(inplace=True)

# Remove Unnecessary Column
df.drop("Unnamed: 0", axis=1, inplace=True)

print("\nShape after Cleaning:")
print(df.shape)
print("\nDataset Statistics:")
print(df.describe())
plt.figure(figsize=(8,5))

sns.histplot(df["Sales"], bins=15, kde=True)

plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Count")

plt.savefig("screenshots/sales_distribution.png")
plt.show()
plt.figure(figsize=(8,5))

sns.scatterplot(x="TV", y="Sales", data=df)

plt.title("TV Advertising vs Sales")

plt.savefig("screenshots/tv_vs_sales.png")
plt.show()
plt.figure(figsize=(8,5))

sns.scatterplot(x="Radio", y="Sales", data=df)

plt.title("Radio Advertising vs Sales")

plt.savefig("screenshots/radio_vs_sales.png")
plt.show()
plt.figure(figsize=(8,6))

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.savefig("screenshots/heatmap.png")
plt.show()
# Features and Target

X = df.drop("Sales", axis=1)
y = df["Sales"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)
# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
# Linear Regression Model

model = LinearRegression()

model.fit(X_train, y_train)

print("Model Trained Successfully!")
# Prediction

y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred[:5])
# Model Performance

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("MAE :", mae)
print("MSE :", mse)
print("R2 Score :", r2)
# Actual vs Predicted Sales
plt.figure(figsize=(8,5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.savefig("screenshots/actual_vs_predicted.png")
plt.show()