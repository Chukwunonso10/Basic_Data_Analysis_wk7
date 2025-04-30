import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
except FileNotFoundError:
    print("Dataset file not found.")
except Exception as e:
    print("An error occurred:", e)

# Display the first few rows
print(df.head())

# Data structure overview
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())


#Task 2

# Summary statistics
print("\nStatistical Summary:")
print(df.describe())

# Group by species and compute mean
print("\nAverage features by species:")
grouped = df.groupby('species').mean()
print(grouped)

# Observations
print("\nInsights:")
print("- Setosa has smallest petal size; Virginica the largest.")
print("- Sepal width is largest for Setosa, smallest for Virginica.")


#Task3

# Seaborn style
sns.set(style="whitegrid")

# Line chart - simulate time-series using index
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.plot(df.index, df['petal length (cm)'], label='Petal Length')
plt.title('Simulated Time-Series of Sepal and Petal Length')
plt.xlabel('Index (Simulated Time)')
plt.ylabel('Length (cm)')
plt.legend()
plt.show()

# Bar Chart - Average petal length by species
plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='petal length (cm)', data=df, palette='pastel')
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

# Histogram - Sepal length distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['sepal length (cm)'], kde=True, bins=15, color='skyblue')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot - Sepal vs Petal Length
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()
