# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 18:49:16 2020

@author: kamlesh sencha
"""
#Implementing multiple linear regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import the dataset
dataset = pd.read_csv('50_Startups.csv')
x= dataset.iloc[:,:-1]
y= dataset.iloc[:, 4]

#convert the column into categorical columns
states = pd.get_dummies(x['State'],drop_first=True)

#drop the state column
x= x.drop('State', axis=1)

#concat the dummy variables
x=pd.concat([x,states],axis=1)

#splltting the dataset into the training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicting the test set results
y_pred= regressor.predict(x_test)
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)