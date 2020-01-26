import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math
import pickle

import seaborn as seabornInstance

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from transform import *


#dataset = pd.read_csv("./data/test_indessa.csv")
dataset = pd.read_csv("./data/10k_test.csv")

dataset = transform_dataset(dataset)

x_test = dataset.values

regressor = None

with open('model', 'rb') as f:
    regressor = pickle.load(f)

y_pred = regressor.predict(x_test)

print(y_pred)