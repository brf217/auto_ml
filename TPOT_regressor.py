#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 07:56:08 2018

@author: feebr01
"""

# Set up imports
from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd



# Load dataset
housing = load_boston()
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target,
       train_size=0.75, test_size=0.25)



# Call TOPT regressor and input parameters
tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, 
                     config_dict='TPOT light')


# Fit model
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_boston_pipeline.py')



# Model scoring dataframe sorted by best to worst scores
my_dict = list(tpot.evaluated_individuals_.items())

model_scores = pd.DataFrame()
for model in my_dict:
    model_name = model[0]
    model_info = model[1]
    cv_score = model[1].get('internal_cv_score')  # Pull out cv_score as a column (i.e., sortable)
    model_scores = model_scores.append({'model': model_name,
                                        'cv_score': cv_score,
                                        'model_info': model_info,},
                                       ignore_index=True)
model_scores = model_scores.sort_values('cv_score', ascending=False)
