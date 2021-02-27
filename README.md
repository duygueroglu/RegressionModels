# RegressionModels
This repo includes machine learning regression models and graphics.

## Table of Contents

+ [Simple Linear Regression Model](https://github.com/duygueroglu/RegressionModels/blob/main/simplelinear.ipynb)
+ [Multiple Linear Regression Model](https://github.com/duygueroglu/RegressionModels/blob/main/multiplelinear.py)
+ [Polynomial Regression Model](https://github.com/duygueroglu/RegressionModels/blob/main/polyRegression.ipynb)
+ [Decision Tree Regression Model](https://github.com/duygueroglu/RegressionModels/blob/main/decisionTree.ipynb)
+ [Random Forest Regression Model](https://github.com/duygueroglu/RegressionModels/blob/main/randomForest.ipynb)

## Descriptions and Graphics

### Simple Linear Regression Model

If a regression problem consists of an estimation variable (X) and the value to be predicted (Y), simple linear regression can be considered sufficiently sensitive(directly affect).

![](https://github.com/duygueroglu/RegressionModels/blob/main/simplelinear.png)

### Multiple Linear Regression Model

If there are more than one variable to estimate the value to be predicted, multiple linear regression can be considered succificiently sensitive.

### Polynomial Regression Model

Polynomial regression, derived from the linear regression library, is to express the input variable as the sum of polynomials (degrees) depending on the polynomial degree.

![](https://github.com/duygueroglu/RegressionModels/blob/main/polyregression2.PNG)

### Decision Tree Regression Model

Decision trees are structures that emerge by dividing the dataset into homogeneous / similar groups / splitting the column (classification or regression) over the concept called entropy. How far the trees will divide (the height of the tree) and from which value to find the root node are related to entropy / split concepts. 

![](https://github.com/duygueroglu/RegressionModels/blob/main/decisiontree.PNG)

### Random Forest Regression Model

It is an ensemble / community learning algorithm for regression that consists of multiple decision trees.
+ While a "Decision Tree" is being trained, it creates a single model using all the data allocated for the train. However, Random Forest trains each tree with the random data it chooses in the train data (not using all of the train data for each tree) depending on the number of trees selected. Therefore, every tree in the community is trained as different models.
