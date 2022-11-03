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
from sklearn.svm import SVC

# Parameters

kernel='rbf'
gamma='scale'
C=1
class_weight='balanced'


# Data Preparation

print('... preparing data ...')
data = pd.read_csv('data/diabetes.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


print('... substituting nan values ...')
def substituting_nan_values(X, columns):
    for column in columns:
        X[column] = X[column].replace(0,X[column].mean())
    return X

columns_to_fillna = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
substituting_nan_values(X_train, columns_to_fillna)


print('... standardising data ...')
def scaling_data(X, data):
    scaler = RobustScaler()
    X = pd.DataFrame(scaler.fit(X).transform(X), columns=data.iloc[:,:-1].columns)
    return X

scaling_data(X_train, data)


# Training the final model

print(f'... training the final model with kernel={kernel}, gamma={gamma}, C={C}, class_weight={class_weight} ...')
def train(X_train, y_train):
    svm_model = SVC(kernel=kernel, gamma=gamma, C=C, class_weight=class_weight)
    svm_model.fit(X_train, y_train)
    return svm_model

svm_model = train(X_train, y_train)


# Saving final model into a pickle file

output_file = 'svm_model.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump(svm_model, f_out)

print(f'... model saved to {output_file} ...')
