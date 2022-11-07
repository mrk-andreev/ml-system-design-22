if [ ! -f /opt/minio-init-bucket/completed ]
then
  /bin/sh init_bucket.sh
  /bin/sh upload_dataset.sh
  touch /opt/minio-init-bucket/completed
fi
exit 0;