import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as seabornInstance

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv("./data/train_indessa.csv")

dataset = dataset.fillna(method='ffill')


#scrub
def transform_term():
    t = dataset['term']
    t.transform(lambda x: int(x.split()[0]))
    dataset['term'] = t

#transform_term()
dataset['term'] = dataset['term'].transform(lambda x: int(x.split()[0]))

x_train = dataset.loc[:, dataset.columns != 'loan_status'].values
y_train = dataset['loan_status'].values

regressor = LinearRegression()
regressor.fit(x_train, y_train)