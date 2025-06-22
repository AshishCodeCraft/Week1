# Import necessary libraries
import pandas as pd # data manipulation
import numpy as np # numerical python - linear algebra

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('PB_All_2000_2021.csv', sep=';')
df.info()
df.shape
df.describe()
df.describe().T
# Missing values
df.isnull().sum()
# date is in object - date format
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
df.info()

df = df.sort_values(by=['id', 'date'])
df.head()

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

df.head()

df.columns