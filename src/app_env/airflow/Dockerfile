FROM python:3.10
ENV AIRFLOW_VERSION=2.4.2
ENV AIRFLOW_DOCKER_PROVIDER_VERSION=3.2.0
WORKDIR /opt/app

RUN PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)" && \
    CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt" && \
    pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}" && \
    pip install apache-airflow-providers-docker==${AIRFLOW_DOCKER_PROVIDER_VERSION} && \
    pip install mlflow==1.30.0 && \
    pip install apache-airflow-providers-cncf-kubernetes==4.4.0 && \
    apt-get update -qq && \
    apt-get install -yqq vim curl && \
    curl -s https://dl.min.io/client/mc/release/linux-amd64/mc > /usr/bin/mc && \
    chmod +x /usr/bin/mc && \
    curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash && \
    mkdir /root/airflow && \
    mkdir /root/airflow/dags
