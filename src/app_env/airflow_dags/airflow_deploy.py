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
    # TODO: download chart
    # TODO: add parameters
    deploy = BashOperator(
        bash_command=' '.join([
            'helm', 'upgrade', '--install', 'predictor', 'ml-system-design-22--predictor-1.0.0.tgz'
        ])
    )

    deploy
