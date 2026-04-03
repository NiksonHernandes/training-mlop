import os
import random
import mlflow
import numpy as np
import random as python_random
import tensorflow
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.utils import to_categorical

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def reset_seeds() -> None:
  os.environ['PYTHONHASHSEED']=str(42)
  tf.random.set_seed(42)
  np.random.seed(42)
  random.seed(42)

url = 'raw.githubusercontent.com'
username = 'Nik.Hernandes14'
repository = 'mlops-ead'
file_name = 'fetal_health_reduced.csv'
#data = pd.read_csv(f'https://{url}/{username}/{repository}/master/{file_name}')
data = pd.read_csv(f'https://dagshub.com/{username}/{repository}/raw/main/{file_name}')

data.head()

X=data.drop(["fetal_health"], axis=1)
y=data["fetal_health"]

columns_names = list(X.columns)
scaler = preprocessing.StandardScaler()
X_df = scaler.fit_transform(X)
X_df = pd.DataFrame(X_df, columns=columns_names)

X_train, X_test, y_train, y_test = train_test_split(X_df,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42)

y_train = y_train -1
y_test = y_test - 1

reset_seeds()
model = Sequential()
model.add(InputLayer(input_shape=(X_train.shape[1], )))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

os.environ['MLFLOW_TRACKING_USERNAME'] = 'Nik.Hernandes14'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '98c50289cded2f647991488e12e8c3d0a5232ee1'
mlflow.set_tracking_uri('https://dagshub.com/Nik.Hernandes14/mlops-ead.mlflow')

mlflow.keras.autolog(log_models=True,
                     log_input_examples=True,
                     log_model_signatures=True)

with mlflow.start_run(run_name='experiment_mlops_ead') as run:
  model.fit(X_train,
            y_train,
            epochs=50,
            validation_split=0.2,
            verbose=3)

