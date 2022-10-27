FROM python:3.10
ENV AIRFLOW_VERSION=2.4.2
ENV AIRFLOW_DOCKER_PROVIDER_VERSION=3.2.0
WORKDIR /opt/app

RUN PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)" && \
    CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt" && \
    pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}" && \
    pip install apache-airflow-providers-docker==${AIRFLOW_DOCKER_PROVIDER_VERSION}
COPY ml-requirements.txt /tmp/ml-requirements.txt
RUN pip install -r /tmp/ml-requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
RUN apt-get update -qq && apt-get install -yqq vim