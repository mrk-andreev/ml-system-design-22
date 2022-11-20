#bin/bash

set -e

export IMAGE=localhost:5000/mlflow:latest
export IMAGE_CHART_DIRECTORY=src/app_env/mlflow
docker build -t $IMAGE $IMAGE_CHART_DIRECTORY
docker push $IMAGE