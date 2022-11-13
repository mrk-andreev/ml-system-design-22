import os
from datetime import datetime

from airflow.models import DAG
from airflow.operators.python import task
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from kubernetes.client.models import V1Volume
from kubernetes.client.models import V1VolumeMount
from mlflow import MlflowClient

KUBE_NAMESPACE = os.environ['KUBE_NAMESPACE']
KUBE_CONFIG_FILE = os.environ['KUBE_CONFIG_FILE']
KUBE_CLUSTER_CONTEXT = os.environ['KUBE_CLUSTER_CONTEXT']
IMAGE_REGISTRY = os.environ['ENDPOINT_IMAGE_REGISTRY']
INNER_TRACKING_URI = 'http://' + os.environ['NODE_IP'] + ':5500'
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
    volume = V1Volume(name='train-dir', empty_dir={
        'sizeLimit': '4Gi',
        'medium': 'Memory'
    })
    volume_mount = V1VolumeMount(name='train-dir', mount_path='/dev/shm')

    train = KubernetesPodOperator(
        name='train',
        task_id='train',
        namespace=KUBE_NAMESPACE,
        in_cluster=False,
        config_file=KUBE_CONFIG_FILE,
        cluster_context=KUBE_CLUSTER_CONTEXT,
        volumes=[volume],
        volume_mounts=[volume_mount],
        is_delete_operator_pod=True,
        get_logs=True,
        arguments=['train.py'],
        image=IMAGE_REGISTRY + 'model:latest',
        env_vars={
            'S3_ENDPOINT': os.environ['SECRET_S3_ENDPOINT'],
            'S3_USER': os.environ['SECRET_S3_ACCESS_TOKEN'],
            'S3_PASSWORD': os.environ['SECRET_S3_SECRET_TOKEN'],
            'S3_BUCKET': os.environ['SECRET_S3_BUCKET'],
            'S3_BUCKET_DATASET_PREFIX': S3_BUCKET_DATASET_PREFIX,
            'DEVICE': DEVICE,
            'MLFLOW_TRACKING_URL': os.environ['ENDPOINT_MLFLOW_TRACKING_URL_EXTERNAL_URL'],
        },
    )

    score = KubernetesPodOperator(
        name='score',
        task_id='score',
        namespace=KUBE_NAMESPACE,
        in_cluster=False,
        config_file=KUBE_CONFIG_FILE,
        cluster_context=KUBE_CLUSTER_CONTEXT,
        volumes=[volume],
        volume_mounts=[volume_mount],
        is_delete_operator_pod=True,
        get_logs=True,
        arguments=['score.py'],
        image=IMAGE_REGISTRY + 'model:latest',
        env_vars={
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
