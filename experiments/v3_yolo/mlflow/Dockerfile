FROM python:3.10
ENV PYTHONPATH /opt/app:
WORKDIR /opt/app

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
COPY entrypoint.bash /opt/entrypoint.bash

ENTRYPOINT ["bash", "/opt/entrypoint.bash"]