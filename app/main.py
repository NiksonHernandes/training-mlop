import mlflow
import tempfile
import os
import json
import uvicorn
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI

# Comando para rodar no Anaconda: uvicorn main:app --reload

app = FastAPI(
    title="Fetal Health API",
    openapi_tags=[
        {
            "name": "Health",
            "description": "Get api health"
        },
        {
            "name": "Prediction",
            "description":"Model prediction"
        }
    ])


# url = 'raw.githubusercontent.com'
# username = 'Nik.Hernandes14'
# repository = 'mlops-ead'
# file_name = 'fetal_health_reduced.csv'
# data = pd.read_csv(f'https://dagshub.com/{username}/{repository}/raw/main/{file_name}')

def load_model():
    print('Loading model...')
    MLFLOW_TRACKING_URI = 'https://dagshub.com/Nik.Hernandes14/mlops-ead.mlflow'
    MLFLOW_TRACKING_USERNAME = 'Nik.Hernandes14'
    MLFLOW_TRACKING_PASSWORD = '98c50289cded2f647991488e12e8c3d0a5232ee1'
    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
    print('setting mlflow tracking uri...')
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print('MLflow tracking uri set successfully!')
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    print('MLflow client created successfully!')
    registered_model = client.get_registered_model('fetal_health_nik')
    print('Read registered model successfully!')

    run_id = registered_model.latest_versions[-1].run_id
    print(f'run_id: {run_id}')  
    logged_model = f'runs:/{run_id}/model'
    print(f'logged_model uri: {logged_model}')  
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    #loaded_model = mlflow.pyfunc.load_model('models:/fetal_health_nik/latest')

    print('Model loaded successfully!')
    return loaded_model

# def load_model():
#     print('Loading model...')
#     MLFLOW_TRACKING_URI = 'https://dagshub.com/Nik.Hernandes14/mlops-ead.mlflow'
#     MLFLOW_TRACKING_USERNAME = 'Nik.Hernandes14'
#     MLFLOW_TRACKING_PASSWORD = '98c50289cded2f647991488e12e8c3d0a5232ee1'

#     os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
#     os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
#     os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
#     mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

#     client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
#     registered_model = client.get_registered_model('fetal_health_nik')
    
#     latest_version = registered_model.latest_versions[-1]
#     run_id = latest_version.run_id
#     print(f'run_id: {run_id}')

#     # Baixar artefatos manualmente passando o tracking_uri explicitamente
#     artifact_path = mlflow.artifacts.download_artifacts(
#         run_id=run_id,
#         artifact_path='model',
#         tracking_uri=MLFLOW_TRACKING_URI  # ← força usar o servidor remoto
#     )
#     print(f'Artifact downloaded to: {artifact_path}')

#     loaded_model = mlflow.pyfunc.load_model(artifact_path)
#     print('Model loaded successfully!')
#     return loaded_model


@app.get(path='/', tags=["Health"])
def api_health():
    return {"status": "healthy"}

@app.post(path='/predict', tags=["Prediction"])
def predict():
    loaded_model = load_model()
    return {"prediction": 0}    
