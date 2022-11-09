import os

DATASET_DIR = os.path.join(os.path.dirname(__file__), '../dataset/')

S3_ENDPOINT = os.environ['S3_ENDPOINT']
S3_USER = os.environ['S3_USER']
S3_PASSWORD = os.environ['S3_PASSWORD']
S3_BUCKET = os.environ['S3_BUCKET']
S3_BUCKET_DATASET_PREFIX = os.environ['S3_BUCKET_DATASET_PREFIX']


def download_dataset():
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.system(f'/usr/bin/mc alias set myminio {S3_ENDPOINT} {S3_USER} {S3_PASSWORD}')
    os.system(f'/usr/bin/mc cp --recursive myminio/{S3_BUCKET}/{S3_BUCKET_DATASET_PREFIX}/ /opt/dataset/')
