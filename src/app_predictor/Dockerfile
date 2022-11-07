FROM python:3.10
WORKDIR /opt/app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r  /tmp/requirements.txt
RUN apt-get update -qq && apt-get install -yy libgl1
RUN git clone https://github.com/ultralytics/yolov5.git /opt/yolov5
RUN touch /opt/yolov5/__init__.py
RUN pip install -r /opt/yolov5/requirements.txt
COPY app_predictor.py app_predictor.py
COPY predict.py predict.py
COPY model model
ENV PYTHONPATH /opt/app:/opt/:/opt/yolov5
ENTRYPOINT ["python"]
CMD ["app_predictor.py"]