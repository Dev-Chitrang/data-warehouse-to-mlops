from typing import Any
from zenml import Model, step

@step
def load_model(model_name: str, artifact_name: str) -> Any:
    model = Model(name=model_name, version="production")
    model_obj = model.load_artifact(artifact_name)
    return model_obj