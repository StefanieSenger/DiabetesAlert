## Project Description

This project is my midterm contribution for the DataTalks.Club [Machine Learning Zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code), that I am participating in in autumn 2022.

It is based on [this kaggle dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset) on diabetes factors, which can be downloaded directly from the page. The dataset is also in the data folder of this project.

The project has two main components: 1. predicting targets for unknown samples via machine learning and 2. model deployment.

    1. Machine Learning: data preprocessing and predicting whether a patient has diabetes based on \
    some features like the BMI, age or concentration of glucosis in their blood sample
    2. Model Deployment with Flask while managing dependencies with virtual environments and Docker

## Data and Problem to Solve

The data was collected by the US National Institute of Diabetes and Digestive and Kidney Diseases in India. Using the dataset for a classification task can help to detect diabetes patients, that haven't been diagnosed yet, based on an ensemble of diagnostic measurements, that each doctor can easily obtain.

## File Structure

```
├── build
│   └── lib
│       └── scripts
│           ├── __init__.py
│           ├── predict.py
│           ├── predict_test.py
│           └── train.py
├── data
│   └── diabetes.csv
├── Dockerfile
├── ml_zoomcamp_midterm_project.egg-info
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── requires.txt
│   ├── SOURCES.txt
│   └── top_level.txt
├── notebooks
│   ├── notebook.ipynb
│   └── predict_test.ipynb
├── README.md
├── requirements.txt
├── scripts
│   ├── __init__.py
│   ├── predict.py
│   ├── predict_test.py
│   ├── __pycache__
│   │   ├── __init__.cpython-39.pyc
│   │   ├── predict.cpython-39.pyc
│   │   └── train.cpython-39.pyc
│   ├── svm_model.bin
│   └── train.py
└── setup.py
```

## Model Selection

In the preprocessing step, we substituted all the NaN values (here showing as '0') with the median of all values and scaled all the numeric columns using scikit-learns RobustScaler.

For the machine learning task, three different model types were trained while varying their hyper-params (KNN Classifier, Logistic Regression and Support Vector Machine). Based on their F1-score the Support Vector Machine turned out to be the best model. It was further fine tuned with GridSearch. Due to the correlation between the features "age" and "number of pregnancies", the performance of the SVM model could be slightly increased by omitting the latter.

## Deployment

After training, the final model and a fitted scaler, that was used for preprocessing, were saved as a pickle file (svm_model.bin). They are deployed locally via Flask (predict.py). For environment management, a Docker container contains the final model with the scaler, the predict.py file and the requirements.txt. When we now request a test prediction (predict_test.py) from outside the Docker container, we will be returned a prediction of whether the patient is more likely to have diabetes or if they is healthy.

## Run the Project

Clone the repository.

From your terminal create a new virtual environment (pyenv) and install the requirements:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`pyenv virtualenv 3.9.13 ml-zoomcamp`<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`pyenv local ml-zoomcamp`<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`pip install -r requirements.txt`

Run the docker container:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`docker build -t midterm .`<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`docker run -it --rm -p 9696:9696 midterm`

It will install the dependencies from the requirements.txt within the container and make predict.py with the final model available.

If you now run the predict_test.py script from outside the container (terminal: `python scripts/predict_test.py`), you will recieve the prediction for respective patient. (They are at risk for diabetes. Further diabetes diagnosis adviced.)
