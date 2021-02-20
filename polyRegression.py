# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("positions.csv")
print(data.columns)

level = data.iloc[:,1].values.reshape(-1,1)
salary = data.iloc[:,2].values.reshape(-1,1)

regression = LinearRegression()
regression.fit(level,salary)

prediction = regression.predict([[8.3]])

plt.scatter(level,salary,color="red")
plt.plot(level,regression.predict(level),color="blue")
plt.show()