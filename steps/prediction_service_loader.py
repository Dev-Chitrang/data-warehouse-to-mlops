from zenml import step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer

@step
def prediction_service_loader(pipeline_name: str, step_name: str, model_name: str):
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
        model_name=model_name,
        running=True
    )
    if not services:
        raise RuntimeError(
            f"No active MLFlow service found for pipeline: {pipeline_name}, step: {step_name}, model: {model_name}"
        )
    return services[0]
