mkdir -p /tmp/dataset
cd /tmp/dataset
gdown https://drive.google.com/uc?id=15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M
unzip WIDER_train.zip
rm WIDER_train.zip
gdown https://drive.google.com/uc?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q
unzip WIDER_val.zip
rm WIDER_val.zip
gdown http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip
unzip wider_face_split.zip
rm wider_face_split.zip

/usr/bin/mc cp --recursive /tmp/dataset myminio/$S3_BUCKET/;