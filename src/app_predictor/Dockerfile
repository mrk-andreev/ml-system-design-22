FROM python:3.10
WORKDIR /opt/app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r  /tmp/requirements.txt
RUN apt-get update -qq && apt-get install -yy libgl1
COPY app_predictor.py app_predictor.py
COPY model model
ENTRYPOINT ["python"]
CMD ["app_predictor.py"]