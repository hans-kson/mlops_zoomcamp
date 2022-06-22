FROM python:3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY ["starter_param.py", "model.bin", "fhv_tripdata_2021-04.parquet", "./" ]

