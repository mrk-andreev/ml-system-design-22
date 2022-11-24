import os
from datetime import datetime

from airflow.models import DAG
from airflow.operators.bash import BashOperator

with DAG(
        'yolov5_deploy',
        start_date=datetime(2021, 1, 1),
        schedule_interval=None,
        catchup=False,
        tags=['yolov5'],
) as dag:
    s3_endpoint = os.environ['SECRET_S3_ENDPOINT']
    schema, endpoint = s3_endpoint.split('://')
    bucket = os.environ['SECRET_S3_BUCKET']
    chart_name = 'ml-system-design-22--predictor-1.0.0.tgz'

    download_chart = BashOperator(
        task_id='download',
        bash_command=' '.join([
            'mc',
            'cp',
            f'backend/{bucket}/ml-system-design-22--predictor-1.0.0.tgz',
            f'/tmp/{chart_name}'
        ]),
        env={
            'PATH': '/usr/bin/',
            'MC_HOST_backend': ''.join([
                schema,
                '://',
                os.environ['SECRET_S3_ACCESS_TOKEN'],
                ':',
                os.environ['SECRET_S3_SECRET_TOKEN'],
                '@',
                endpoint
            ])
        }
    )

    deploy = BashOperator(
        task_id='deploy_chart',
        bash_command=' '.join([
            'helm',
            '--kubeconfig=/root/.kube/config',
            '--set model.name=yolov5',
            '--set model.stage=Production',
            '--set model.stage=Production',
            f'--set integrations.mlflow.trackingUrl=http://${os.environ["NODE_IP"]}:5500',
            f'--set integrations.s3.endpointUrl={os.environ["SECRET_S3_ENDPOINT"]}',
            f'--set integrations.s3.accessKey={os.environ["SECRET_S3_ACCESS_TOKEN"]}',
            f'--set integrations.s3.secretKey={os.environ["SECRET_S3_SECRET_TOKEN"]}',
            f'--set integrations.s3.bucket={os.environ["SECRET_S3_BUCKET"]}',
            f'--set integrations.telegram.token={os.environ["TELEGRAM_TOKEN"]}',
            f'--set integrations.telegram.token={os.environ["TELEGRAM_TOKEN"]}',
            f'--set integrations.redis.host={os.environ["REDIS_HOST"]}',
            f'--set integrations.redis.port={os.environ["REDIS_PORT"]}',
            f'--set integrations.redis.queue={os.environ["REDIS_QUEUE"]}',
            f'--set integrations.redis.username={os.environ["REDIS_USERNAME"]}',
            f'--set integrations.redis.password={os.environ["REDIS_PASSWORD"]}',
            'upgrade',
            '--install',
            'predictor',
            f'/tmp/{chart_name}',
        ])
    )

    download_chart >> deploy
