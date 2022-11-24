#bin/bash

set -e

export IMAGE=localhost:5000/model:latest
export IMAGE_CHART_DIRECTORY=src/app_model
docker build -t $IMAGE $IMAGE_CHART_DIRECTORY
docker push $IMAGE