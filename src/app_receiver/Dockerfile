FROM python:3.10
WORKDIR /opt/app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r  /tmp/requirements.txt
COPY app_receiver.py app_receiver.py
ENTRYPOINT ["python"]
CMD ["app_receiver.py"]