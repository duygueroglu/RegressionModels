# -*- coding: utf-8 -*-

#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#reading .csv file data
data = pd.read_csv("insurance.csv")
print(data.columns)

##y-axis
expenses = data.expenses.values.reshape(-1,1)

##x-axis
ageBmis = data.iloc[:,[0,2]].values #all data, 0. and 1. columns

#create instance of linear regression
regression = LinearRegression()
regression.fit(ageBmis,expenses)

#sample prediction,expenses according to mass
print(regression.predict([[20,20],[20,21],[20,22],[20,23],[20,24]]))
#or
#print(regression.predict(np.array([[20,20]])))

