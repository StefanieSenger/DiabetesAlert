#!/usr/bin/env python
# coding: utf-8
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import pickle
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# Parameters

kernel='rbf'
gamma='scale'
C=1
class_weight='balanced'


# Data Preparation

print('... preprocessing data ...')
data = pd.read_csv('data/diabetes.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


print('... substituting nan values ...')
def substituting_nan_values(X):
    columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
    for column in columns:
        X[column] = X[column].replace(0,X[column].mean())
    return X

substituting_nan_values(X_train)


print('... standardising data ...')
def scaling_data(X):
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    scaler = RobustScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=columns)
    return X, scaler

X_train, scaler = scaling_data(X_train)


# Training the model

svm_model = SVC(probability=True)

param_grid = [{
        "C": [0.5, 0.7, 1.0, 1.25],
        "kernel": ['linear', 'rbf', 'sigmoid'],
        "gamma": ['scale', 'auto'],
        "class_weight": ['balanced', None]
        }]

grid_search = GridSearchCV(svm_model, param_grid, scoring='f1', cv=5, verbose=1)
grid_search.fit(X_train, y_train)

svm_model = grid_search.best_estimator_

print(f'... training the final model ...')
def train(X_train, y_train):
    svm_model.fit(X_train, y_train)
    return svm_model

svm_model = train(X_train, y_train)


# Saving final model into a pickle file

output_file = 'scripts/svm_model.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump((svm_model,scaler), f_out)

print(f'... model saved to {output_file} ...')
