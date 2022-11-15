FROM python:3.10
ENV AIRFLOW_VERSION=2.4.2
ENV AIRFLOW_DOCKER_PROVIDER_VERSION=3.2.0
WORKDIR /opt/app

RUN PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)" && \
    CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt" && \
    pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}" && \
    pip install apache-airflow-providers-docker==${AIRFLOW_DOCKER_PROVIDER_VERSION}
RUN pip install mlflow==1.30.0
RUN pip install apache-airflow-providers-cncf-kubernetes==4.4.0
RUN apt-get update -qq && apt-get install -yqq vim
RUN curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
RUN mkdir /root/airflow && mkdir /root/airflow/dags