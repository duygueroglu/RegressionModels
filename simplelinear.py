# -*- coding: utf-8 -*-

#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#reading .csv file data
data = pd.read_csv("hw_25000.csv")

height = data.Height.values.reshape(-1,1)
weight = data.Weight.values.reshape(-1,1)

regression = LinearRegression()
regression.fit(height,weight) #fitting the lines according to weight and height

#sample predictions 
print(regression.predict([[60]])) 
print(regression.predict([[62]]))
print(regression.predict([[64]]))
print(regression.predict([[66]]))
print(regression.predict([[68]]))
print(regression.predict([[70]]))

print(data.columns)

plt.scatter(data.Height, data.Weight) #visualization
x = np.arange(min(data.Height),max(data.Height)).reshape(-1,1) #create a range 
plt.plot(x,regression.predict(x),color="red") #drawing predictions' line, y->x's predicted value
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Simple Linear Regression Model")
plt.show()

#calculation of algorithm success using r-square method
print(r2_score(weight, regression.predict(height)))