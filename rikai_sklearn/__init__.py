import rikai
from typing import Any, Optional

def get_model_type(model):
    model_name = model.__module__[len("sklearn."):]
    model_name = model_name.split(".")[0]
    name = model.__class__.__name__
    return f"rikai_sklearn.{model_name}.{name}"


def log_model(
    model: Any,
    artifact_path: str = "model",
    schema: Optional[str] = None,
    registered_model_name: Optional[str] = None,
    customized_flavor: Optional[str] = None,
    labels: Optional[dict] = None,
    **kwargs,
):
    model_type = get_model_type(model)
    rikai.mlflow.sklearn.log_model(
        model,
        artifact_path,
        schema,
        registered_model_name,
        customized_flavor,
        model_type,
        labels,
        **kwargs
    )
