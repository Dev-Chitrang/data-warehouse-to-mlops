# Data Warehouse to MLOps Pipeline

## Overview
This system processes retail data from a structured data warehouse into machine learning models. It supports forecasting, customer classification, and product recommendation in a single MLOps workflow.

## Architecture
- **Data Warehouse**: SQL Server implementation using a Medallion architecture (Bronze, Silver, Gold schemas).
- **Orchestration**: ZenML for defining and running pipeline steps.
- **Experiment Tracking**: MLflow for logging metrics, parameters, and model artifacts.
- **Inference Service**: FastAPI server that loads model artifacts for real-time predictions.
- **Infrastructure**: Containerized environment using Docker.

## Key Engineering Decisions
- **Medallion Data Layering**: Separates raw ingestion (Bronze), cleaning (Silver), and business aggregation (Gold) to maintain data lineage and facilitate debugging.
- **Factory Pattern for Ingestion**: Decouples data sources (SQL Server, Excel) from the pipeline logic.
- **Single Pipeline**: Executes three ML tasks (forecasting, classification, recommendation) in one run for data consistency.
- **Model Registry Decoupling**: Models are retrieved via MLflow artifact paths in the FastAPI service, separating training from inference.

## Features
- Data ingestion from SQL Server and Excel sources.
- EDA report generation via ydata-profiling.
- Multi-model pipeline execution:
    - Time-series forecasting using NeuralProphet.
    - Customer classification using Scikit-learn.
    - Product recommendation logic.
- Containerized inference API with endpoints for each model type.
- Model evaluation metrics stored in MLflow.

## Tech Stack
- **Backend**: Python 3.x, FastAPI
- **ML Frameworks**: ZenML, MLflow, Scikit-learn, NeuralProphet
- **Database**: SQL Server (T-SQL)
- **Data Processing**: Pandas, pyodbc
- **Reporting**: ydata-profiling
- **Infrastructure**: Docker, Docker Compose

## Workflow / API
### Training Workflow
1. `run_training_pipeline.py` triggers the ZenML pipeline.
2. Data is extracted from the SQL Server Gold layer.
3. Features are engineered and split for three separate model tasks.
4. Models are trained, evaluated, and logged to the MLflow artifact store.

### Inference API
The FastAPI server (`main.py`) provides the following endpoints:
- `POST /predict/classification`: Input features to receive customer category labels.
- `POST /predict/forecasting`: Input date sequences to receive future value predictions.
- `POST /predict/recommendation`: Input customer/product IDs to receive product recommendations.

## Challenges & Solutions
- **Multi-Model Orchestration**: Training different model types (NeuralProphet vs. Sklearn) in one pipeline required custom wrappers to standardize inputs and outputs for ZenML steps.
- **Windows Deployment Limits**: Since native ZenML/MLflow server deployment has limitations on Windows, a custom Docker-based deployment strategy was used to ensure portability and consistency.

## How to Run
1. **Configuration**: Set up database credentials in a `.env` file based on `requirements.txt` and project needs.
2. **Train Models**: Execute the training pipeline:
   ```bash
   python run_training_pipeline.py
   ```
3. **Start Inference API**: Run the FastAPI server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
   Or use Docker:
   ```bash
   docker-compose up --build
   ```
