/usr/bin/mc alias set myminio $S3_ENDPOINT $S3_USER $S3_PASSWORD;
/usr/bin/mc mb myminio/$S3_BUCKET;