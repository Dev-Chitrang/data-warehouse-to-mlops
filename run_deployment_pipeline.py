import click
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)

MODEL_TYPES = ["forecasting", "classification", "recommendation"]

@click.command()
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop all running MLflow prediction services",
)
@click.option(
    "--run-inference",
    is_flag=True,
    default=False,
    help="Run inference pipeline for all deployed models",
)
def run_main(stop_service: bool, run_inference: bool):
    """Run or stop the full deployment and inference pipelines."""

    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    if stop_service:
        for model_name in MODEL_TYPES:
            services = model_deployer.find_model_server(
                pipeline_name="continuous_deployment_pipeline",
                pipeline_step_name="mlflow_model_deployer_step",
                model_name=f"{model_name}_model",
                running=True,
            )
            if services:
                print(f":stop_sign: Stopping {model_name} model server...")
                services[0].stop(timeout=10)
        return

    # Step 1: Run the deployment pipeline
    print(":rocket: [bold green]Running deployment pipeline...[/bold green]")
    continuous_deployment_pipeline()

    # Step 2: Run inference if requested
    if run_inference:
        for model_type in MODEL_TYPES:
            print(f":mag_right: Running inference for [bold blue]{model_type}[/bold blue] model...")
            inference_pipeline(model_type=model_type)()

    # Step 3: Print MLflow UI info
    print(
        "\n[bold green]To inspect experiment runs, open MLflow UI:[/bold green]\n"
        f"  [yellow]mlflow ui --backend-store-uri {get_tracking_uri()}[/yellow]\n"
    )

    # Step 4: Display prediction server URLs
    for model_type in MODEL_TYPES:
        services = model_deployer.find_model_server(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            model_name=f"{model_type}_model",
        )
        if services and services[0].is_running:
            print(
                f":white_check_mark: [bold]{model_type}[/bold] model is deployed at:\n"
                f"    [cyan]{services[0].prediction_url}[/cyan]\n"
                f"Run again with [yellow]--stop-service[/yellow] to shut it down."
            )


if __name__ == "__main__":
    run_main()
