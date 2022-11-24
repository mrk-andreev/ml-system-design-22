#bin/bash

set -e

export IMAGE=localhost:5000/predictor:latest
export IMAGE_CHART_DIRECTORY=src/app_predictor
docker build -t $IMAGE $IMAGE_CHART_DIRECTORY
docker push $IMAGE