mlflow ui \
  --host 0.0.0.0 \
  --port 5500 \
  --backend-store-uri postgresql://$DB_USER:$DB_PASS@$DB_HOST:$DB_PORT/$DB_NAME \
  --serve-artifacts --artifacts-destination s3://$S3_BUCKET/$S3_BUCKET_PREFIX
