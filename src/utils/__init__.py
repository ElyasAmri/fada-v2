# FADA Utils Module
from .training import EarlyStopping, ModelCheckpoint
from .validation_metrics import ComprehensiveMetrics
from .image_processing import to_jpeg_bytes, to_base64, to_base64_data_url, load_and_convert
from .api_client import call_with_retry, retry_decorator, exponential_backoff, APIError
from .results_manager import ResultsManager
from .logging_config import setup_logging, get_logger

# MLflow utilities (optional - may not have mlflow installed)
try:
    from .mlflow_utils import (
        setup_mlflow_experiment,
        log_training_config,
        log_model_architecture,
        log_evaluation_results,
        log_confusion_matrix,
        log_training_curves,
        get_best_run,
        create_run_name,
        MLflowTrainerCallback,
        MLflowModelCheckpoint,
        log_gpu_metrics,
        log_inference_metrics,
    )
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False