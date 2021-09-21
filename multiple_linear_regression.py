# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 04:48:41 2021

@author: hp
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values


# Taking care of the non numerical data. for that we need label encoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x = LabelEncoder()
X[:, 3] = labelencoder_x.fit_transform(X[:, 3])
# The above line has encoded the categorical data, Now we need to make columns for each
# category. That can be done by OneHotencoder

#onehotencoder = OneHotEncoder(categories_[3])
#X = onehotencoder.fit_transform(X).toarray()
ct = ColumnTransformer( [('one_hot_encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#Avoiding the dummy variable trap
X = X[:, 1:]
#Splitting the dataset into the Training set andn Train set

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state =0)


# Fitting multiple Linear Regressor to the Training set
from sklearn.linear_model import LinearRegression
linearregressor = LinearRegression()
linearregressor.fit(X_train, Y_train)

#Predicting the test set
y_predict = linearregressor.predict(X_test)

#Building the optimal model using backward elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
p=regressor_OLS.summary()
print(p)
X_opt = X[:, [0,1,3,4,5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
p=regressor_OLS.summary()
print(p)
X_opt = X[:, [0,3,4,5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
p=regressor_OLS.summary()
print(p)
X_opt = X[:, [3,4,5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
p=regressor_OLS.summary()
print(p)
X_opt = X[:, [3,4]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
p=regressor_OLS.summary()
print(p)