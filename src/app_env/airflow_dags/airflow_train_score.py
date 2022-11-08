from datetime import datetime

from airflow.models import DAG
from airflow.operators.python import task
from airflow.providers.docker.operators.docker import DockerOperator
from mlflow import MlflowClient

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
        image='localhost:5000/model:latest',
        network_mode='host',
        task_id='train',
        dag=dag,
        shm_size=2147483648,
        environment={
            'S3_ENDPOINT': 'http://localhost:9000',
            'S3_USER': 'miniouser',
            'S3_PASSWORD': 'miniopassword',
            'S3_BUCKET': 'mlsystemdesign22',
            'S3_BUCKET_DATASET_PREFIX': 'dataset',
            'DEVICE': 'cpu',
            'MLFLOW_TRACKING_URL': 'http://localhost:5500',
        }
    )

    score = DockerOperator(
        docker_url='unix://var/run/docker.sock',
        command='score.py',
        image='localhost:5000/model:latest',
        network_mode='host',
        task_id='score',
        dag=dag,
        shm_size=2147483648,
        environment={
            'S3_ENDPOINT': 'http://localhost:9000',
            'S3_USER': 'miniouser',
            'S3_PASSWORD': 'miniopassword',
            'S3_BUCKET': 'mlsystemdesign22',
            'S3_BUCKET_DATASET_PREFIX': 'dataset',
            'DEVICE': 'cpu',
            'MLFLOW_TRACKING_URL': 'http://localhost:5500',
            'MODEL_NAME': 'yolov5',
            'MODEL_STAGE': 'Staging'
        }
    )


    @task(task_id="transition_model_version_stage_to_staging")
    def transition_model_version_stage_to_staging():
        TRACKING_URI = 'http://mlflow:5500'
        MODEL_NAME = 'yolov5'

        client = MlflowClient(TRACKING_URI)
        model = client.get_registered_model(MODEL_NAME)
        last_version = model.latest_versions[-1]
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=last_version.version,
            stage="Staging"
        )


    @task(task_id="transition_model_version_stage_to_production")
    def transition_model_version_stage_to_production():
        TRACKING_URI = 'http://mlflow:5500'
        MODEL_NAME = 'yolov5'

        client = MlflowClient(TRACKING_URI)
        model = client.get_registered_model(MODEL_NAME)
        last_version = model.latest_versions[-1]
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=last_version.version,
            stage="Production"
        )


    train >> transition_model_version_stage_to_staging() >> score >> transition_model_version_stage_to_production()
