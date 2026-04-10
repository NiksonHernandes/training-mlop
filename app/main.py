import mlflow
import tempfile
import os
import json
import uvicorn
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI

# Comando para rodar no Anaconda: uvicorn main:app --reload

# Variáveis para ler o dataset do DagsHub
# url = 'raw.githubusercontent.com'
# username = 'Nik.Hernandes14'
# repository = 'mlops-ead'
# file_name = 'fetal_health_reduced.csv'
# data = pd.read_csv(f'https://dagshub.com/{username}/{repository}/raw/main/{file_name}')

class FetalHealthData(BaseModel):
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    severe_decelerations: float

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


def load_model():
    print('Loading model...')
    MLFLOW_TRACKING_URI = 'https://dagshub.com/Nik.Hernandes14/mlops-ead.mlflow'
    MLFLOW_TRACKING_USERNAME = 'Nik.Hernandes14'
    MLFLOW_TRACKING_PASSWORD = '98c50289cded2f647991488e12e8c3d0a5232ee1'
    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
    print('setting mlflow tracking uri...')
    mlflow.set_tracking_uri('https://dagshub.com/Nik.Hernandes14/mlops-ead.mlflow')
    print('MLflow tracking uri set successfully!')
    client = mlflow.MlflowClient(tracking_uri='https://dagshub.com/Nik.Hernandes14/mlops-ead.mlflow')    
    print('MLflow client created successfully!')
    registered_model = client.get_registered_model('fetal_health')
    print('Read registered model successfully!')

    registered_model.latest_versions

    run_id = registered_model.latest_versions[-1].run_id
    print(f'run_id: {run_id}')  
    logged_model = registered_model.latest_versions[-1].source
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    print(f'logged_model uri: {logged_model}')  

    print('Model loaded successfully!')
    return loaded_model

@app.on_event("startup") # Anotação faz o método acontecer quando a aplicação é iniciada, ou seja, quando o servidor é ligado
def startup_event():
    global loaded_model # pode acessar a variável model que foi declarada fora do método, em todo o código
    loaded_model = load_model()

@app.get(path='/', tags=["Health"])
def api_health():
    return {"status": "healthy"}

@app.post(path='/predict', tags=["Prediction"])
def predict(request: FetalHealthData):    
    global loaded_model

    #converto o dado para um array numpy e reshape para o formato esperado pelo modelo
    received_data = np.array([
        request.accelerations,
        request.fetal_movement,
        request.uterine_contractions,
        request.severe_decelerations,
    ]).reshape(1, -1)

    print ('Received data for prediction:', received_data)
    prediction = loaded_model.predict(received_data)
    print('Prediction result:', prediction)

    return {"prediction": str(np.argmax(prediction[0]))}    
