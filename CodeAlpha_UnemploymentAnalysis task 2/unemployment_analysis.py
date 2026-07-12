import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print(df.head())
print(df.info())
print(df.columns)
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)
print("\nDataset Statistics:")
print(df.describe())
df['Date'] = pd.to_datetime(df['Date'])

df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Date'] = pd.to_datetime(df['Date'])

df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
plt.figure(figsize=(8,5))

sns.histplot(df['Estimated Unemployment Rate (%)'], bins=20, kde=True)

plt.title("Distribution of Unemployment Rate")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Count")

plt.show()
plt.figure(figsize=(15,6))

sns.barplot(
    data=df,
    x='Region',
    y='Estimated Unemployment Rate (%)'
)

plt.xticks(rotation=90)
plt.title("State-wise Unemployment Rate")

plt.show()
plt.figure(figsize=(8,6))

sns.heatmap(
    df.select_dtypes(include='number').corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.show()
top_states = df.groupby("Region")["Estimated Unemployment Rate (%)"].mean().sort_values(ascending=False)

print("\nTop 10 States by Average Unemployment Rate:\n")
print(top_states.head(10))
plt.figure(figsize=(12,5))

sns.lineplot(
    data=df,
    x='Date',
    y='Estimated Unemployment Rate (%)'
)

plt.title("Impact of COVID-19 on Unemployment")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)

plt.show()
plt.savefig("histogram.png")
plt.show()
plt.savefig("statewise_bar_chart.png")
plt.show()
plt.savefig("covid_trend.png")
plt.show()
plt.savefig("heatmap.png")
plt.show()
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)