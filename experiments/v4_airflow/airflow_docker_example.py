from datetime import datetime

from airflow.models import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator

with DAG(
        "docker",
        start_date=datetime(2021, 1, 1),
        catchup=False,
        tags=["example", "docker"],
) as dag:
    t1 = BashOperator(task_id='print_date', bash_command='date', dag=dag)
    t2 = BashOperator(task_id='sleep', bash_command='sleep 5', retries=3, dag=dag)
    t3 = DockerOperator(
        docker_url='unix://var/run/docker.sock',  # Set your docker URL
        command='/bin/sleep 30',
        image='centos:latest',
        network_mode='bridge',
        task_id='docker_op_tester',
        dag=dag,
    )
    t4 = BashOperator(task_id='print_hello', bash_command='echo "hello world!!!"', dag=dag)
    # t1 >> t2
    # t1 >> t3
    # t3 >> t4

    (
        # TEST BODY
            t1
            >> [t2, t3]
            >> t4
    )
