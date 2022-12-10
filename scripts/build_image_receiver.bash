#bin/bash
# example: bash scripts/build_image_receiver.bash markandreev/receiver:20221209-3


set -e

export IMAGE=$1
export IMAGE_CHART_DIRECTORY=src/app_receiver
docker build -t $IMAGE $IMAGE_CHART_DIRECTORY
docker push $IMAGE
