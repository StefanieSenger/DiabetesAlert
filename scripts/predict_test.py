#!/usr/bin/env python
# coding: utf-8

import requests

# Testing predict.py, that is served via flask

url = 'http://localhost:9696/predict'

patient = {
   "Pregnancies": 2.0,
   "Glucose": 204.0,
   "BloodPressure": 65.0,
   "SkinThickness": 0.0,
   "Insulin": 43.0,
   "BMI": 19.0,
   "DiabetesPedigreeFunction": 0.49,
   "Age": 39.0
    }

response = requests.post(url, json=patient).json()
print(response)

if response['prediction'] == True:
    print('further diabetes diagnostics adviced')
else:
    print('further diabetes diagnostics not necessary')
