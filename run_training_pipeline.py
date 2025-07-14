from pipelines.training_pipeline import ml_pipeline
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    ml_pipeline()