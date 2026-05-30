"""
modelling.py
============
Skrip training untuk MLflow Project (tahap Workflow CI / SMSML).

Berbeda dengan `modelling_tuning.py` yang melakukan tracking online ke DagsHub,
skrip ini dirancang agar dapat dijalankan otomatis oleh GitHub Actions melalui
perintah `mlflow run`. Tracking dilakukan secara LOKAL (folder `mlruns/`) sehingga
artefak model tersimpan di dalam repositori dan dapat:
  - di-upload sebagai artefak workflow (GitHub Actions / Google Drive), dan
  - dipakai oleh `mlflow models build-docker` untuk membangun Docker Image.

Entry point menerima parameter hyperparameter sehingga mudah dikonfigurasi dari
file `MLProject` maupun command line:

    mlflow run . --env-manager=local \
        -P n_estimators=400 -P max_depth=10

Setelah run selesai, run_id dicetak ke stdout dan ditulis ke `run_id.txt`
supaya step berikutnya pada CI (build-docker) dapat mengambil model.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend non-GUI agar aman di CI runner

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset_preprocessing"


# ----------------------------- Helpers --------------------------------------
def load_dataset():
    """Muat dataset hasil preprocessing yang sudah siap dilatih."""
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test


def save_confusion_matrix(y_true, y_pred, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    for (i, j), val in __import__("numpy").ndenumerate(cm):
        plt.text(j, i, int(val), ha="center", va="center")
    plt.xticks([0, 1], ["Not Survived", "Survived"])
    plt.yticks([0, 1], ["Not Survived", "Survived"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RandomForest (MLflow Project).")
    parser.add_argument("--n_estimators", type=int, default=400)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    return parser.parse_args()


# ----------------------------- Main pipeline --------------------------------
def main() -> None:
    args = parse_args()

    # Saat dijalankan via `mlflow run`, MLflow sudah membuat run aktif dan
    # menetapkan experiment lewat env var (MLFLOW_RUN_ID / MLFLOW_EXPERIMENT_ID).
    # Dalam kondisi itu kita TIDAK boleh memanggil set_experiment / set_tracking_uri
    # lagi agar tidak bentrok. Jika dijalankan langsung (`python modelling.py`),
    # barulah kita atur tracking lokal sendiri.
    existing_run_id = os.getenv("MLFLOW_RUN_ID")
    if existing_run_id is None:
        if not os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri((BASE_DIR / "mlruns").as_uri())
        mlflow.set_experiment("Titanic_CI")

    # autolog menghasilkan struktur model/ standar (MLmodel, conda.yaml,
    # model.pkl, python_env.yaml, requirements.txt) + estimator.html.
    mlflow.sklearn.autolog(log_models=True, log_input_examples=False)

    X_train, X_test, y_train, y_test = load_dataset()

    # max_depth=-1 (atau <=0) diinterpretasikan sebagai None (tanpa batas).
    max_depth = args.max_depth if args.max_depth and args.max_depth > 0 else None

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=42,
        n_jobs=-1,
    )

    # Jika sudah ada run aktif dari `mlflow run`, lampirkan ke run tersebut
    # (tanpa run_name baru). Selain itu buat run baru sendiri.
    run_ctx = (
        mlflow.start_run()
        if existing_run_id
        else mlflow.start_run(run_name="rf_ci")
    )
    with run_ctx as run:
        run_id = run.info.run_id
        print(f"[ci] run_id = {run_id}")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)
        for k, v in metrics.items():
            print(f"  {k:10s} = {v:.4f}")

        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag("stage", "workflow_ci")

        # Artefak tambahan: confusion matrix + classification report.
        art_dir = BASE_DIR / "artifacts_tmp"
        art_dir.mkdir(exist_ok=True)

        cm_path = art_dir / "confusion_matrix.png"
        save_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        report = classification_report(
            y_test, y_pred,
            target_names=["Not Survived", "Survived"],
            digits=4,
        )
        report_path = art_dir / "classification_report.txt"
        report_path.write_text(report, encoding="utf-8")
        mlflow.log_artifact(str(report_path), artifact_path="reports")

        meta_path = art_dir / "metric_info.json"
        meta_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(meta_path))

        # Tulis run_id agar step CI berikutnya (build-docker) bisa memakainya.
        (BASE_DIR / "run_id.txt").write_text(run_id, encoding="utf-8")
        print(f"[ci] run_id ditulis ke {BASE_DIR / 'run_id.txt'}")
        print(f"[ci] model URI       = runs:/{run_id}/model")
        print("[ci] DONE.")


if __name__ == "__main__":
    main()
