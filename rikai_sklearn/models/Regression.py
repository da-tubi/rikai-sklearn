from rikai_sklearn.models import SklearnModelType


class Regression(SklearnModelType):
    def schema(self) -> str:
        return "float"

    def predict(self, x, *args, **kwargs) -> float:
        assert self.model is not None
        return self.model.predict(x).tolist()

MODEL_TYPE = Regression()
