import rikai
from typing import Any, Optional


def _get_model_type(model):
    tpe = model._estimator_type
    if tpe not in ("regressor", "classifier"):
        raise RuntimeError(f"Estimator type ({tpe}) not supported")

    return f"rikai_sklearn.models.{tpe}"

def log_model(
    model: Any,
    registered_model_name: Optional[str] = None,
    schema: Optional[str] = None,
    customized_flavor: Optional[str] = None,
    labels: Optional[dict] = None,
    artifact_path: str = "model",
    **kwargs,
):
    model_type = _get_model_type(model)
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
