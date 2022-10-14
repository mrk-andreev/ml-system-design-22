import abc

import mlflow


class BaseModel(mlflow.pyfunc.PythonModel, abc.ABC):
    @property
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def load_context(self, context):
        pass

    @abc.abstractmethod
    def predict_from_picture(self, img):
        pass

    def predict(self, context, model_input):
        return self.predict_from_picture(model_input)
