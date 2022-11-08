import os
from datetime import datetime

from airflow.models import DAG
from airflow.operators.python import task
from airflow.providers.docker.operators.docker import DockerOperator
from mlflow import MlflowClient

MODEL_IMAGE = os.environ['ENDPOINT_IMAGE_REGISTRY'] + 'model:latest'
INNER_TRACKING_URI = os.environ['ENDPOINT_MLFLOW_TRACKING_URL_INNER_URL']
MODEL_NAME = 'yolov5'
S3_BUCKET_DATASET_PREFIX = 'dataset'
DEVICE = 'cpu'

with DAG(
        'yolov5_train_score',
        start_date=datetime(2021, 1, 1),
        schedule_interval=None,
        catchup=False,
        tags=['yolov5'],
) as dag:
    train = DockerOperator(
        docker_url='unix://var/run/docker.sock',
        command='train.py',
        image=MODEL_IMAGE,
        network_mode='host',
        task_id='train',
        dag=dag,
        shm_size=2147483648,
        environment={
            'S3_ENDPOINT': os.environ['SECRET_S3_ENDPOINT'],
            'S3_USER': os.environ['SECRET_S3_ACCESS_TOKEN'],
            'S3_PASSWORD': os.environ['SECRET_S3_SECRET_TOKEN'],
            'S3_BUCKET': os.environ['SECRET_S3_BUCKET'],
            'S3_BUCKET_DATASET_PREFIX': S3_BUCKET_DATASET_PREFIX,
            'DEVICE': DEVICE,
            'MLFLOW_TRACKING_URL': os.environ['ENDPOINT_MLFLOW_TRACKING_URL_EXTERNAL_URL'],
        }
    )

    score = DockerOperator(
        docker_url='unix://var/run/docker.sock',
        command='score.py',
        image=MODEL_IMAGE,
        network_mode='host',
        task_id='score',
        dag=dag,
        shm_size=2147483648,
        environment={
            'S3_ENDPOINT': os.environ['SECRET_S3_ENDPOINT'],
            'S3_USER': os.environ['SECRET_S3_ACCESS_TOKEN'],
            'S3_PASSWORD': os.environ['SECRET_S3_SECRET_TOKEN'],
            'S3_BUCKET': os.environ['SECRET_S3_BUCKET'],
            'S3_BUCKET_DATASET_PREFIX': S3_BUCKET_DATASET_PREFIX,
            'DEVICE': DEVICE,
            'MLFLOW_TRACKING_URL': os.environ['ENDPOINT_MLFLOW_TRACKING_URL_EXTERNAL_URL'],
            'MODEL_NAME': MODEL_NAME,
            'MODEL_STAGE': 'Staging'
        }
    )


    @task(task_id="transition_model_version_stage_to_staging")
    def transition_model_version_stage_to_staging(tracking_url):
        client = MlflowClient(tracking_url)
        model = client.get_registered_model(MODEL_NAME)
        last_version = model.latest_versions[-1]
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=last_version.version,
            stage="Staging"
        )


    @task(task_id="transition_model_version_stage_to_production")
    def transition_model_version_stage_to_production(tracking_url):
        client = MlflowClient(tracking_url)
        model = client.get_registered_model(MODEL_NAME)
        last_version = model.latest_versions[-1]
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=last_version.version,
            stage="Production"
        )


    (
            train >>
            transition_model_version_stage_to_staging(INNER_TRACKING_URI) >>
            score >>
            transition_model_version_stage_to_production(INNER_TRACKING_URI)
    )
