#!/usr/bin/env python
"""Fine-tune a YOLO detector and log everything to MLflow. Designed to run as a DVC stage.

Reads hyper-parameters from params.yaml (`train:` section) so that `dvc repro` and `dvc exp run`
track them. Requires the `train` extra:  pip install -e ".[train]"

Usage:
  python scripts/train.py --params params.yaml
  MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/train.py   # remote tracking server
"""
import argparse
import json
import shutil
from pathlib import Path
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
import yaml
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--experiment", default="visionpipe-detector")
    ap.add_argument("--out-model", default="models/best.pt")
    ap.add_argument("--metrics-out", default="metrics/train_metrics.json")
    args = ap.parse_args()

    params = yaml.safe_load(open(args.params))
    tp, data_yaml = dict(params["train"]), params["data"]["yaml"]
    base = tp.pop("model")

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run() as run:
        mlflow.log_params({f"train.{k}": v for k, v in tp.items()} | {"train.base_model": base, "data.yaml": data_yaml})
        model = YOLO(base)
        results = model.train(data=data_yaml, project="runs", name="train", exist_ok=True, **tp)

        metrics = {k.replace("(", "_").replace(")", "").replace("/", "_"): float(v) for k, v in results.results_dict.items()}
        mlflow.log_metrics(metrics)

        best = Path(model.trainer.best)
        Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, args.out_model)
        mlflow.log_artifact(args.out_model, artifact_path="model")
        for extra in ("results.png", "confusion_matrix.png", "PR_curve.png"):
            p = best.parent.parent / extra
            if p.exists():
                mlflow.log_artifact(str(p), artifact_path="plots")

        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_out).write_text(json.dumps({**metrics, "mlflow_run_id": run.info.run_id}, indent=2))
        print(f"best weights -> {args.out_model}; MLflow run {run.info.run_id}")


if __name__ == "__main__":
    main()
