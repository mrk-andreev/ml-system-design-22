import os
import tempfile

from experiments.v2_tracking.src.base import BaseModel
from experiments.v2_tracking.src.score import score_model

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')


class Cv2HaarCascadeFrontalFaceModel(BaseModel):
    def __init__(self):
        self._model = None
        self._workdir = tempfile.TemporaryDirectory()
        self._weight = os.path.join(self._workdir.name, 'data.xml')

    def __del__(self):
        self._workdir.cleanup()

    @property
    def name(self):
        return "cv2-haarcascade_frontalface_default"

    def load_context(self, context):
        import cv2

        with open(context.artifacts['weight'], 'rb') as f_in, open(self._weight, 'wb') as f_out:
            f_out.write(f_in.read())

        self._model = cv2.CascadeClassifier(self._weight)

    def predict_from_picture(self, img):
        import cv2

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._model.detectMultiScale(gray, 1.1, 4)
        return [[int(x) for x in list(face)] for face in faces]

    def predict(self, context, model_input):
        return self.predict_from_picture(model_input)


def main():
    cls = Cv2HaarCascadeFrontalFaceModel
    weight_path = os.path.join(DATA_DIR, 'haarcascade_frontalface_default.xml')
    score_model(cls, weight_path)


if __name__ == "__main__":
    main()
