## Dataset (http://shuoyang1213.me/WIDERFACE/):

- **Train**: https://drive.google.com/file/d/15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M/view?usp=sharing
- **Validation**: https://drive.google.com/file/d/1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q/view?usp=sharing
- **Annotations**: http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip

## Run train

cd model
docker run -v /tmp/ml-system-design-22/experiments/v3_yolo/dataset:/opt/dataset $(docker build . -qq) train.py

# Run score

cd model
docker run -v /tmp/ml-system-design-22/experiments/v3_yolo/dataset:/opt/dataset -v
/tmp/ml-system-design-22/experiments/v3_yolo/sample_weight:/opt/sample_weight $(docker build . -qq) score.py
