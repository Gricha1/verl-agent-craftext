# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A unified tracking interface that supports logging data to different backend
"""

import dataclasses
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Union


class Tracking:
    """A unified tracking interface for logging experiment data to multiple backends.

    This class provides a centralized way to log experiment metrics, parameters, and artifacts
    to various tracking backends including WandB, MLflow, SwanLab, TensorBoard, and console.

    Attributes:
        supported_backend: List of supported tracking backends.
        logger: Dictionary of initialized logger instances for each backend.
    """

    supported_backend = ["wandb", "mlflow", "swanlab", "vemlp_wandb", "tensorboard", "console", "clearml", "comet"]

    def __init__(self, project_name, experiment_name, default_backend: Union[str, List[str]] = "console", config=None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == "tracking":
                import warnings

                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning, stacklevel=2)
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}

        if "tracking" in default_backend or "wandb" in default_backend:
            import wandb

            wandb.init(project=project_name, name=experiment_name, config=config)
            self.logger["wandb"] = wandb

        if "mlflow" in default_backend:
            import os

            import mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", None)
            if MLFLOW_TRACKING_URI:
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # Project_name is actually experiment_name in MLFlow
            # If experiment does not exist, will create a new experiment
            experiment = mlflow.set_experiment(project_name)
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger["mlflow"] = _MlflowLoggingAdapter()

        if "swanlab" in default_backend:
            import os

            import swanlab

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten

            if config is None:
                config = {}  # make sure config is not None, otherwise **config will raise error
            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config={"FRAMEWORK": "verl", **config},
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            self.logger["swanlab"] = swanlab

        if "vemlp_wandb" in default_backend:
            import os

            import volcengine_ml_platform
            from volcengine_ml_platform import wandb as vemlp_wandb

            volcengine_ml_platform.init(
                ak=os.environ["VOLC_ACCESS_KEY_ID"],
                sk=os.environ["VOLC_SECRET_ACCESS_KEY"],
                region=os.environ["MLP_TRACKING_REGION"],
            )

            vemlp_wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                sync_tensorboard=True,
            )
            self.logger["vemlp_wandb"] = vemlp_wandb

        if "tensorboard" in default_backend:
            self.logger["tensorboard"] = _TensorboardAdapter()

        if "console" in default_backend:
            from verl.utils.logger.aggregate_logger import LocalLogger

            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

        if "clearml" in default_backend:
            self.logger["clearml"] = ClearMLLogger(project_name, experiment_name, config)

        if "comet" in default_backend:
            self.logger["comet"] = CometMLLogger(project_name, experiment_name, config)

    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)

    def log_validation_video(self, gif_path: str, step: int, name: str = "validation_trajectory"):
        """Log a validation trajectory GIF to backends that support it (e.g. Comet ML)."""
        if "comet" in self.logger and hasattr(self.logger["comet"], "log_video"):
            self.logger["comet"].log_video(gif_path, step=step, name=name)

    def __del__(self):
        if "wandb" in self.logger:
            self.logger["wandb"].finish(exit_code=0)
        if "swanlab" in self.logger:
            self.logger["swanlab"].finish()
        if "vemlp_wandb" in self.logger:
            self.logger["vemlp_wandb"].finish(exit_code=0)
        if "tensorboard" in self.logger:
            self.logger["tensorboard"].finish()

        if "clearnml" in self.logger:
            self.logger["clearnml"].finish()
        if "comet" in self.logger:
            self.logger["comet"].finish()


class ClearMLLogger:
    def __init__(self, project_name: str, experiment_name: str, config):
        self.project_name = project_name
        self.experiment_name = experiment_name

        import clearml

        self._task: clearml.Task = clearml.Task.init(
            task_name=experiment_name,
            project_name=project_name,
            continue_last_task=True,
            output_uri=False,
        )

        self._task.connect_configuration(config, name="Hyperparameters")

    def _get_logger(self):
        return self._task.get_logger()

    def log(self, data, step):
        import numpy as np
        import pandas as pd

        # logs = self._rewrite_logs(data)
        logger = self._get_logger()
        for k, v in data.items():
            title, series = k.split("/", 1)

            if isinstance(v, (int, float, np.floating, np.integer)):
                logger.report_scalar(
                    title=title,
                    series=series,
                    value=v,
                    iteration=step,
                )
            elif isinstance(v, pd.DataFrame):
                logger.report_table(
                    title=title,
                    series=series,
                    table_plot=v,
                    iteration=step,
                )
            else:
                logger.warning(f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}". This invocation of ClearML logger\'s function is incorrect so this attribute was dropped. ')

    def finish(self):
        self._task.mark_completed()


class CometMLLogger:
    def __init__(self, project_name: str, experiment_name: str, config):
        import os

        from comet_ml import Experiment

        # Get API key from environment or use default
        api_key = os.environ.get("COMET_API_KEY", None)
        workspace = os.environ.get("COMET_WORKSPACE", None)
        
        # Prefer RUN_NAME environment variable if available, otherwise use experiment_name parameter
        # This ensures the experiment name matches the RUN_NAME from the training script
        run_name = os.environ.get("RUN_NAME", None)
        final_experiment_name = run_name if run_name else experiment_name
        
        # Initialize Comet experiment
        # Set display_summary_level=0 to reduce console output
        self.experiment = Experiment(
            api_key=api_key,
            workspace=workspace,
            project_name=project_name,
            experiment_name=final_experiment_name,
            auto_param_logging=False,  # We'll log params manually
            auto_metric_logging=False,  # We'll log metrics manually
            display_summary_level=0,
        )
        
        # Explicitly set the experiment name to ensure it's used
        # This is a safeguard in case the constructor parameter doesn't work as expected
        self.experiment.set_name(final_experiment_name)
        
        # Store experiment reference for validation logging
        # Comet ML automatically sets this as the global experiment
        self._experiment_name = final_experiment_name
        
        # Log hyperparameters if config is provided
        if config is not None:
            self._log_config(config)

    def _log_config(self, config):
        """Log configuration parameters to Comet ML"""
        params = _flatten_dict(
            _transform_params_to_json_serializable(config, convert_list_to_dict=True),
            sep="/"
        )
        self.experiment.log_parameters(params)

    def log(self, data, step):
        """Log metrics to Comet ML"""
        for key, value in data.items():
            # Comet ML expects numeric values for metrics
            if isinstance(value, (int, float)):
                self.experiment.log_metric(key, value, step=step)
            # Handle other types if needed
            elif hasattr(value, 'item'):  # For torch tensors, numpy scalars, etc.
                try:
                    self.experiment.log_metric(key, value.item(), step=step)
                except (AttributeError, ValueError):
                    pass  # Skip if conversion fails

    def log_video(self, gif_path: str, step: int, name: str = "validation_trajectory"):
        """Log a GIF video (e.g. validation trajectory) to Comet ML."""
        import os
        if os.path.isfile(gif_path):
            # Comet log_image accepts name= and step=; image_metadata is not supported
            self.experiment.log_image(gif_path, name=name, step=step)

    def finish(self):
        """End the Comet ML experiment"""
        self.experiment.end()


class _TensorboardAdapter:
    def __init__(self):
        import os
        import time

        from torch.utils.tensorboard import SummaryWriter

        run_name = os.environ.get("RUN_NAME")
        if run_name is None:
            tensorboard_dir = f"runs/run_{time.strftime('%Y%m%d-%H%M%S')}"
        else:
            tensorboard_dir = "runs/" + run_name

        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def finish(self):
        self.writer.close()


class _MlflowLoggingAdapter:
    def log(self, data, step):
        import mlflow

        results = {k.replace("@", "_at_"): v for k, v in data.items()}
        mlflow.log_metrics(metrics=results, step=step)


def _compute_mlflow_params_from_objects(params) -> Dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: Dict[str, Any], *, sep: str) -> Dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans


@dataclasses.dataclass
class ValidationGenerationsLogger:
    def log(self, loggers, samples, step):
        if "wandb" in loggers:
            self.log_generations_to_wandb(samples, step)
        if "swanlab" in loggers:
            self.log_generations_to_swanlab(samples, step)
        if "mlflow" in loggers:
            self.log_generations_to_mlflow(samples, step)

        if "clearml" in loggers:
            self.log_generation_to_clearml(samples, step)
        if "comet" in loggers:
            self.log_generations_to_comet(samples, step)

    def log_generations_to_wandb(self, samples, step):
        """Log samples to wandb as a table"""
        import wandb

        # Create column names for all samples
        columns = ["step"] + sum([[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))], [])

        if not hasattr(self, "validation_table"):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(step)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=step)
        self.validation_table = new_table

    def log_generations_to_swanlab(self, samples, step):
        """Log samples to swanlab as text"""
        import swanlab

        swanlab_text_list = []
        for i, sample in enumerate(samples):
            row_text = f"""
            input: {sample[0]}
            
            ---
            
            output: {sample[1]}
            
            ---
            
            score: {sample[2]}
            """
            swanlab_text_list.append(swanlab.Text(row_text, caption=f"sample {i + 1}"))

        # Log to swanlab
        swanlab.log({"val/generations": swanlab_text_list}, step=step)

    def log_generations_to_mlflow(self, samples, step):
        """Log validation generation to mlflow as artifacts"""
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        import json
        import tempfile

        import mlflow

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                row_data = []
                for sample in samples:
                    data = {"input": sample[0], "output": sample[1], "score": sample[2]}
                    row_data.append(data)
                with open(validation_gen_step_file, "w") as file:
                    json.dump(row_data, file)
                mlflow.log_artifact(validation_gen_step_file)
        except Exception as e:
            print(f"WARNING: save validation generation file to mlflow failed with error {e}")

    def log_generation_to_clearml(self, samples, step):
        """Log validation generation to clearml as table"""

        import clearml
        import pandas as pd

        task: clearml.Task | None = clearml.Task.current_task()
        if task is None:
            return

        table = [
            {
                "step": step,
                "input": sample[0],
                "output": sample[1],
                "score": sample[2],
            }
            for sample in samples
        ]

        logger = task.get_logger()
        logger.report_table(
            series="Validation generations",
            title="Validation",
            table_plot=pd.DataFrame.from_records(table),
            iteration=step,
        )

    def log_generations_to_comet(self, samples, step):
        """Log validation generation to Comet ML as a table"""
        import json
        import tempfile
        from pathlib import Path

        try:
            from comet_ml import Experiment

            # Try to get the current experiment
            # Comet ML sets the experiment as global when initialized
            experiment = Experiment.get_global_experiment()
            if experiment is None:
                # If no global experiment, try to find it by name
                # This is a fallback and may not always work
                return

            # Create a table-like structure for Comet ML
            # Comet ML supports logging HTML tables or JSON artifacts
            with tempfile.TemporaryDirectory() as tmp_dir:
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                row_data = []
                for i, sample in enumerate(samples):
                    data = {
                        "sample_id": i + 1,
                        "step": step,
                        "input": sample[0],
                        "output": sample[1],
                        "score": sample[2],
                    }
                    row_data.append(data)
                
                with open(validation_gen_step_file, "w") as file:
                    json.dump(row_data, file, indent=2)
                
                # Log as asset (artifact)
                experiment.log_asset(
                    validation_gen_step_file,
                    file_name=f"validation_generations_step_{step}.json",
                    step=step,
                )
                
                # Also log as text for easy viewing in Comet UI
                summary_text = "\n\n".join(
                    [
                        f"Sample {i+1}:\nInput: {sample[0]}\nOutput: {sample[1]}\nScore: {sample[2]}"
                        for i, sample in enumerate(samples)
                    ]
                )
                experiment.log_text(summary_text, step=step, metadata={"type": "validation_generations"})
        except Exception as e:
            print(f"WARNING: save validation generation file to Comet ML failed with error {e}")
