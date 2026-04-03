import os
import mlflow
import random
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
from mlflow.tracking import MlflowClient

def reset_seeds() -> None:
  os.environ['PYTHONHASHSEED']=str(42)
  tf.random.set_seed(42)
  np.random.seed(42)
  random.seed(42)

def read_data():
    url = 'raw.githubusercontent.com'
    username = 'Nik.Hernandes14'
    repository = 'mlops-ead'
    file_name = 'fetal_health_reduced.csv'
    data = pd.read_csv(f'https://dagshub.com/{username}/{repository}/raw/main/{file_name}')    
    X = data.drop(["fetal_health"], axis=1) #Nome da coluna de target: fetal_health no data train
    y = data["fetal_health"]
    return X, y

    #https://raw.githubusercontent.com/renansantosmendes/lectures-cdas-2023/master/fetal_health_reduced.csv

def process_data(X, y):
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
    return X_train, X_test, y_train, y_test

def create_model(x):
    reset_seeds()
    model = Sequential()
    model.add(InputLayer(input_shape=(x.shape[1], )))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model

def config_mlflow():
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'Nik.Hernandes14'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = '98c50289cded2f647991488e12e8c3d0a5232ee1'
    mlflow.set_tracking_uri('https://dagshub.com/Nik.Hernandes14/mlops-ead.mlflow')
    mlflow.autolog(disable=True)

# def train_model(model, X_train, y_train, is_train=True):
#    with mlflow.start_run(run_name='experiment_mlops_ead_nik') as run:
#     model.fit(X_train,
#             y_train,
#             epochs=50,
#             validation_split=0.2,
#             verbose=3)
#     # name= no lugar de artifact_path= (nova sintaxe)
#     mlflow.keras.log_model(model, name="model")
#     print("Modelo salvo!")

def train_model(model, X_train, y_train):
    with mlflow.start_run(run_name='experiment_mlops_ead_nik') as run:
        run_id = run.info.run_id
        print(f"Run ID: {run_id}")

        mlflow.log_params({
            "epochs": 50, "optimizer": "adam",
            "loss_function": "sparse_categorical_crossentropy",
            "validation_split": 0.2, "layers": "10-10-3", "activation": "relu"
        })

        history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=2)

        for epoch, _ in enumerate(history.history['loss']):
            mlflow.log_metrics({k: history.history[k][epoch] for k in ['loss', 'val_loss', 'accuracy', 'val_accuracy']}, step=epoch)

        mlflow.tensorflow.log_model(model, artifact_path="model")
        print("Modelo salvo!")

    client = MlflowClient()
    version = client.create_model_version(name="fetal_health", source=f"runs:/{run_id}/model", run_id=run_id)
    print(f"Versão registrada: v{version.version}")

if __name__ == "__main__":
    X, y = read_data()
    X_train, X_test, y_train, y_test = process_data(X, y)
    model = create_model(X)
    config_mlflow()
    train_model(model, X_train, y_train)