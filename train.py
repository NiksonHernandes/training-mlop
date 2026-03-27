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

"""# Definindo funções adicionais"""

def reset_seeds() -> None:
  os.environ['PYTHONHASHSEED']=str(42)
  tf.random.set_seed(42)
  np.random.seed(42)
  random.seed(42)

"""# 2 - Fazendo a leitura do dataset e atribuindo às respectivas variáveis"""

url = 'raw.githubusercontent.com'
username = 'renansantosmendes'
repository = 'lectures-cdas-2023'
file_name = 'fetal_health_reduced.csv'
data = pd.read_csv(f'https://{url}/{username}/{repository}/master/{file_name}')

"""# 3 - Preparando o dado antes de iniciar o treino do modelo"""

X = data.drop(["fetal_health"], axis=1)
y = data["fetal_health"]

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

"""# 4 - Criando o modelo e adicionando as camadas"""

reset_seeds()
model = Sequential()
model.add(InputLayer(input_shape=(X_train.shape[1], )))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=3, activation='softmax'))

"""# 5 - Compilando o modelo"""

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

"""##**Configurando o mlflow**"""

os.environ['MLFLOW_TRACKING_USERNAME'] = 'renansantosmendes'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '6d730ef4a90b1caf28fbb01e5748f0874fda6077'
mlflow.set_tracking_uri('https://dagshub.com/renansantosmendes/puc_lectures_mlops.mlflow')

mlflow.keras.autolog(disable=True)

"""# 6 - Executando o treino do modelo"""

with mlflow.start_run(run_name='experiment_mlops_ead_nik') as run:
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")

    mlflow.log_params({
        "epochs": 50,
        "optimizer": "adam",
        "loss_function": "sparse_categorical_crossentropy",
        "validation_split": 0.2,
        "layers": "10-10-3",
        "activation": "relu"
    })

    history = model.fit(X_train, y_train,
                        epochs=50,
                        validation_split=0.2,
                        verbose=2)

    for epoch in range(len(history.history['loss'])):
        mlflow.log_metrics({
            "loss":         history.history['loss'][epoch],
            "val_loss":     history.history['val_loss'][epoch],
            "accuracy":     history.history['accuracy'][epoch],
            "val_accuracy": history.history['val_accuracy'][epoch],
        }, step=epoch)

    model.save("modelo.keras")
    mlflow.log_artifact("modelo.keras", artifact_path="model")
    print("Artefato salvo!")

client = MlflowClient()

version = client.create_model_version(
    name="fetal_health_nik",
    source=f"runs:/{run_id}/model/modelo.keras",
    run_id=run_id
)
print(f"Versão registrada: v{version.version}")

