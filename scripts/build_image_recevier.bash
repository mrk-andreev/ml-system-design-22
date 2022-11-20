#bin/bash

set -e

export IMAGE=localhost:5000/receiver:latest
export IMAGE_CHART_DIRECTORY=src/app_receiver
docker build -t $IMAGE $IMAGE_CHART_DIRECTORY
docker push $IMAGE