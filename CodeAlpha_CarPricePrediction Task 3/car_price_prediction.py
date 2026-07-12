import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Load Dataset
df = pd.read_csv("car data.csv")

# First 5 rows
print(df.head())

# Dataset Information
print("\nDataset Information:")
print(df.info())

# Column Names
print("\nColumns:")
print(df.columns)

# Dataset Shape
print("\nShape:")
print(df.shape)
# Missing Values Check
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate Values Check
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove Duplicates (if any)
df.drop_duplicates(inplace=True)

print("\nShape after removing duplicates:")
print(df.shape)
# Selling Price Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Selling_Price'], bins=20, kde=True)
plt.title("Selling Price Distribution")
plt.xlabel("Selling Price (Lakhs)")
plt.ylabel("Count")
plt.savefig("screenshots/selling_price_distribution.png")
plt.show()
# Correlation Heatmap
plt.figure(figsize=(8,6))

numeric_df = df.select_dtypes(include=['int64', 'float64'])

sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")
plt.savefig("screenshots/correlation_heatmap.png")
plt.show()
# Present Price vs Selling Price
plt.figure(figsize=(8,5))

sns.scatterplot(
    x='Present_Price',
    y='Selling_Price',
    data=df
)

plt.title("Present Price vs Selling Price")
plt.savefig("screenshots/present_vs_selling.png")
plt.show()

# Encode Categorical Columns
encoder = LabelEncoder()

categorical_columns = ['Car_Name', 'Fuel_Type', 'Selling_type', 'Transmission']

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

print("\nEncoded Dataset:")
print(df.head())
# Features and Target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
# Train Model
model = LinearRegression()

model.fit(X_train, y_train)

print("Model Trained Successfully!")
# Prediction
y_pred = model.predict(X_test)

print("Predictions:")
print(y_pred[:5])
# Model Evaluation
print("\nModel Performance")

print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("R2 Score :", r2_score(y_test, y_pred))
# Actual vs Predicted Graph
plt.figure(figsize=(8,5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")

plt.savefig("screenshots/actual_vs_predicted.png")
plt.show()
# Feature Importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nFeature Importance:")
print(importance.sort_values(by='Coefficient', ascending=False))