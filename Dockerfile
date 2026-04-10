FROM python:3.11-slim

# Define o diretório de trabalho dentro do container
WORKDIR /code

# Copia o arquivo de dependências para o container (Máquina -> container)
COPY ./AmbientesConda/requirements.txt /code/requirements.txt

# Instala as dependências Python listadas no requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copia todo o código da aplicação para o container (Todo mundo dentro da pasta app -> container)
COPY ./app /code/app

# Define o comando padrão a executar quando o container iniciar (uvicorn main:app --reload)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

