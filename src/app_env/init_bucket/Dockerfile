FROM ubuntu:22.10
WORKDIR /opt/app

RUN apt-get update -qq && \
    apt-get install -yqq unzip curl python3 python3-pip && \
    curl https://dl.min.io/client/mc/release/linux-amd64/mc \
      --create-dirs \
      -o /bin/mc && \
    chmod +x /bin/mc && \
    pip3 install gdown==4.5.3
COPY init_bucket.sh init_bucket.sh
COPY upload_dataset.sh upload_dataset.sh
COPY entrypoint.sh entrypoint.sh

ENTRYPOINT ["sh", "entrypoint.sh"]