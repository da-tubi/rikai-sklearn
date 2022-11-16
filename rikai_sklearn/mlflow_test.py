from rikai_sklearn.mlflow import _get_model_type

def test_get_model_type():
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    assert _get_model_type(regressor) == "rikai_sklearn.models.regressor"

    from sklearn.linear_model import RidgeClassifier
    classifier = RidgeClassifier()
    assert _get_model_type(classifier) == "rikai_sklearn.models.classifier"
