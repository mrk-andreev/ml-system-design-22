#bin/bash

set -e

export IMAGE=localhost:5000/init_bucket:latest
export IMAGE_CHART_DIRECTORY=src/app_env/init_bucket
docker build -t $IMAGE $IMAGE_CHART_DIRECTORY
docker push $IMAGE