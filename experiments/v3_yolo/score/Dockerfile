FROM python:3.10
ENV PYTHONPATH /opt/app:/opt/:/opt/yolov5
WORKDIR /opt/app

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN apt-get update -qq && apt-get install -yqq git libgl1
RUN git clone https://github.com/ultralytics/yolov5.git /opt/yolov5
RUN touch /opt/yolov5/__init__.py
RUN pip install -r /opt/yolov5/requirements.txt
COPY score.py /opt/app

ENTRYPOINT ["python3", "score.py"]