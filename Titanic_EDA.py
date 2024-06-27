import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('tested.csv')

#Data Visualisation
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.columns.tolist())
print(df.nunique())
print(df.isnull().sum())

#Data Cleaning
remove_columns = ['Cabin']
df.drop(remove_columns, inplace=True, axis=1)
print(df.head())

#Gender Distribution Bar Graph
gender_count = df['Sex'].value_counts()
plt.figure(figsize=(8, 6))
plt.bar(gender_count.index, gender_count, color='red')
plt.title('Count Plot of Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

#Kernel Density Plot
sns.set_style("darkgrid")
numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
plt.figure(figsize=(14, len(numerical_columns) * 3))
for idx, feature in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 2, idx)
    sns.histplot(df[feature], kde=True)
    plt.title(f"{feature} | Skewness: {round(df[feature].skew(), 2)}")
plt.tight_layout()
plt.show()

#Swarm Plot
plt.figure(figsize=(10, 8))
sns.swarmplot(x="Sex", y="Age", data=df, palette='viridis')
plt.title('Swarm Plot for Gender and Age')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.show()

#Pair plot
sns.set_palette("Pastel1")
plt.figure(figsize=(10, 6))
sns.pairplot(df)
plt.suptitle('Pair Plot for DataFrame')
plt.show()

#Box Plot
sns.boxplot(x='Survived', y='Age', data=df)
plt.show()


#Correlation Matrix
plt.figure(figsize=(15, 10))
dt = pd.read_csv('tested.csv')
remove_columns1 = ['Name','Sex','Ticket','Cabin','Embarked']
dt.drop(remove_columns1, inplace=True, axis=1)
sns.heatmap(dt.corr(), annot=True, fmt='.2f', cmap='Pastel2', linewidths=2)
plt.title('Correlation Heatmap')
plt.show()
