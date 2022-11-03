#!/usr/bin/env python
# coding: utf-8
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# Parameters

num_sample_of_X_test_for_prediction = 0


# Getting test data

data = pd.read_csv('data/diabetes.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


def substituting_nan_values(X, columns):
    for column in columns:
        X[column] = X[column].replace(0,X[column].mean())
    return X

columns_to_fillna = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
substituting_nan_values(X_test, columns_to_fillna)


def scaling_data(X, data):
    scaler = RobustScaler()
    X = pd.DataFrame(scaler.fit(X).transform(X), columns=data.iloc[:,:-1].columns)
    return X

scaling_data(X_test, data)


# Loading model

model_file = 'scripts/svm_model.bin'

with open(model_file, 'rb') as f_in:
    svm_model = pickle.load(f_in)


# Predict

def predict(X, num_sample_of_X_test_for_prediction):
    input_data = X.iloc[num_sample_of_X_test_for_prediction, :]
    y_pred = svm_model.predict(np.array(input_data).reshape(1,-1))
    return y_pred, input_data

y_pred, input_data = predict(X_test, num_sample_of_X_test_for_prediction)

print(f'For the {num_sample_of_X_test_for_prediction}th patient from the newly submitted data')
print(' ')
print('with the following diagnostics:')
print(input_data)
print(' ')
print(f'the prediction of having diabetes is: {y_pred}')
