from datetime import datetime

from airflow.models import DAG
from airflow.providers.docker.operators.docker import DockerOperator

with DAG(
        'yolov5',
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

    train
