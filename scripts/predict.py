#!/usr/bin/env python
# coding: utf-8

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import pickle
import numpy as np

from flask import Flask
from flask import request
from flask import jsonify


# Loading model

#model_file = 'scripts/svm_model.bin'
model_file = 'svm_model.bin'

with open(model_file, 'rb') as f_in:
    (svm_model, scaler) = pickle.load(f_in)


# Predict

app = Flask('diabetes')

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()

    X = scaler.transform(np.array(list(patient.values())).reshape(1,-1))
    y_pred = svm_model.predict_proba(X)
    above_threshold = y_pred >= 0.5

    result = {'probability_of_diabetes': float(y_pred.T[1]), 'prediction': bool(above_threshold.T[1])}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
