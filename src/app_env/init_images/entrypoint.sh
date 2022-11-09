cd /opt/app_model
docker build . -t registry:5000/model:latest -t localhost:5000/model:latest
docker push registry:5000/model:latest
docker push localhost:5000/model:latest