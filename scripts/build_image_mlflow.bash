#bin/bash
# example: bash scripts/build_image_mlflow.bash markandreev/mlflow:20221208

set -e

export IMAGE=$1
export IMAGE_CHART_DIRECTORY=src/app_env/mlflow
docker build -t $IMAGE $IMAGE_CHART_DIRECTORY
docker push $IMAGE
