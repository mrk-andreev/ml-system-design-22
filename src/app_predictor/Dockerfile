FROM python:3.10
WORKDIR /opt/app
ENV PYTHONPATH /opt/app:/opt/

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r  /tmp/requirements.txt
RUN apt-get update -qq && apt-get install -yy libgl1
COPY model model
COPY app_predictor.py app_predictor.py
COPY predict.py predict.py
COPY resnet.py resnet.py
COPY bisenet.py bisenet.py
ENTRYPOINT ["python"]
CMD ["app_predictor.py"]
