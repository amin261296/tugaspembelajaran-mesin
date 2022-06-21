# data manipulation
import pandas as pd
import numpy as np 

# visualiation
import seaborn as sns
import matplotlib.pyplot as plt

# model training
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# classifiers
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.tree import DecisionTreeClassifier # decision Tree
drug_data=pd.read_csv('/kaggle/input/drug-classification/drug200.csv')
drug_data.head()\
drug_data.shape
drug_data.info()
drug_data.describe()
drug_data.select_dtypes(include='object')
drug_data['Sex'].value_counts()
plt.figure(figsize=(5,4))
sns.countplot(drug_data['Sex'])
plt.title('counts of observations in Sex')