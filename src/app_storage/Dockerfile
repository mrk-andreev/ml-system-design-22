FROM ubuntu:22.10

RUN apt-get update -qq && \
    apt-get install -yqq unzip curl python3 python3-pip && \
    curl https://dl.min.io/client/mc/release/linux-amd64/mc \
      --create-dirs \
      -o /bin/mc && \
    chmod +x /bin/mc && \
    pip3 install gdown==4.5.3
