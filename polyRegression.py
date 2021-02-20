# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("positions.csv")
print(data.columns)

level = data.iloc[:,1].values.reshape(-1,1)
salary = data.iloc[:,2].values.reshape(-1,1)

regression = LinearRegression()
regression.fit(level,salary)

prediction = regression.predict([[8.3]])

regressionPoly = PolynomialFeatures(degree = 4)
levelPoly = regressionPoly.fit_transform(level)
regression2 = LinearRegression()
regression2.fit(levelPoly,salary)

prediction2 = regression2.predict(regressionPoly.fit_transform([[8.3]]))


plt.scatter(level,salary,color="red")
plt.plot(level,regression.predict(level),color="blue")
plt.plot(level,regression2.predict(levelPoly))
plt.show()
