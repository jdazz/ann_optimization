import hashlib
import json
import os
import re
import shutil
import zipfile
from datetime import datetime
from typing import Dict, Optional, Tuple


def _sanitize_name(name: str) -> str:
    """Return a filesystem-safe name: lowercase, alnum/underscore only."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "dataset"


def make_config_hash(config: Dict, length: int = 10) -> str:
    """Deterministic short hash of a config dict."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha1(config_str.encode("utf-8")).hexdigest()[:length]


def derive_dataset_name(dataset_train, fallback: str = "dataset") -> str:
    """
    Try to derive a readable dataset name:
    - dataset_train.name if present
    - basename of an attached path
    - fallback to provided default.
    """
    candidate = None

    if hasattr(dataset_train, "name") and dataset_train.name:
        candidate = str(dataset_train.name)
    elif hasattr(dataset_train, "source") and isinstance(dataset_train.source, str):
        candidate = os.path.basename(dataset_train.source)

    if candidate:
        candidate = os.path.splitext(candidate)[0]
    else:
        candidate = fallback

    return _sanitize_name(candidate)


def derive_run_label(
    train_filename: Optional[str],
    test_filename: Optional[str] = None,
    fallback: str = "dataset",
) -> str:
    """
    Build a filesystem-safe label from the uploaded train/test filenames.
    Example: train.csv + test.csv -> "train_test".
    If the test file is missing, only the train filename is used.
    """
    parts = []

    for name in (train_filename, test_filename):
        if not name:
            continue
        base = os.path.splitext(str(name))[0]
        sanitized = _sanitize_name(base)
        if sanitized:
            parts.append(sanitized)

    if not parts:
        return _sanitize_name(fallback)

    return "_".join(parts)


def get_run_paths(base_dir: str, dataset_name: str, config_hash: str, status: str) -> str:
    """Build run dir path with status suffix."""
    run_id = f"{dataset_name}__{config_hash}"
    return os.path.join(base_dir, f"{run_id}__{status}")


def start_run_folder(
    base_dir: str,
    dataset_name: str,
    config_hash: str,
) -> Tuple[str, str]:
    """
    Prepare a fresh IN_PROGRESS folder. If an old IN_PROGRESS/DONE exists for the
    same id, remove it to avoid mixing artifacts.
    Returns (run_dir, run_id).
    """
    os.makedirs(base_dir, exist_ok=True)
    in_progress = get_run_paths(base_dir, dataset_name, config_hash, "IN_PROGRESS")
    done = get_run_paths(base_dir, dataset_name, config_hash, "DONE")

    # Clear previous attempts
    for p in (in_progress, done):
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)

    os.makedirs(in_progress, exist_ok=True)
    return in_progress, f"{dataset_name}__{config_hash}"


def finalize_run_folder(
    run_dir: str,
    status: str = "DONE",
) -> str:
    """Rename the run dir to reflect final status."""
    base = os.path.dirname(run_dir)
    name = os.path.basename(run_dir)
    if "__" not in name:
        return run_dir
    prefix = "__".join(name.split("__")[:2])  # dataset__hash
    target = os.path.join(base, f"{prefix}__{status}")
    # Remove target if it exists to avoid rename collisions
    if os.path.isdir(target):
        shutil.rmtree(target, ignore_errors=True)
    try:
        if os.path.isdir(run_dir):
            os.replace(run_dir, target)
            return target
    except Exception:
        pass
    return run_dir


def write_summary(
    run_dir: str,
    dataset_name: str,
    config_hash: str,
    status: str,
    best_params: Optional[Dict] = None,
    best_loss: Optional[float] = None,
    test_metrics: Optional[Dict] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
    notes: Optional[str] = None,
):
    """Write a simple summary.txt into the run dir."""
    os.makedirs(run_dir, exist_ok=True)
    lines = []
    lines.append(f"Dataset: {dataset_name}")
    lines.append(f"Config Hash: {config_hash}")
    lines.append(f"Status: {status}")
    if started_at:
        lines.append(f"Started: {started_at}")
    if finished_at:
        lines.append(f"Finished: {finished_at}")
    if best_loss is not None:
        lines.append(f"Best CV Loss: {best_loss}")
    if isinstance(test_metrics, dict):
        nmae = test_metrics.get("NMAE")
        r2 = test_metrics.get("R2")
        acc = test_metrics.get("Accuracy")
        lines.append("Test Metrics (best model on unseen data):")
        if nmae is not None:
            lines.append(f"  NMAE: {nmae}")
        if r2 is not None:
            lines.append(f"  R2: {r2}")
        if acc is not None:
            lines.append(f"  Accuracy: {acc}")
    if notes:
        lines.append(f"Notes: {notes}")

    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, "a") as f:
        if f.tell() != 0:
            f.write("\n")
        f.write("\n".join(lines))

    return summary_path


def find_existing_run(base_dir: str, dataset_name: str, config_hash: str):
    """Return (status, path) if a matching run folder exists, else (None, None)."""
    in_progress = get_run_paths(base_dir, dataset_name, config_hash, "IN_PROGRESS")
    done = get_run_paths(base_dir, dataset_name, config_hash, "DONE")

    if os.path.isdir(in_progress):
        return "IN_PROGRESS", in_progress
    if os.path.isdir(done):
        return "DONE", done
    return None, None


def append_run_log(run_dir: str, message: str):
    """Append a log line to summary.txt for the active run."""
    if not run_dir:
        return
    # Filter out verbose metric payloads from the summary log
    msg = str(message)
    if any(
        tag in msg
        for tag in (
            
            "y_pred",
            "y_true",
            "[best_params]"
        )
    ):
        return
    try:
        # Do not recreate missing folders (avoids resurrecting IN_PROGRESS after abort)
        if not os.path.isdir(run_dir):
            return
        path = os.path.join(run_dir, "summary.txt")
        with open(path, "a") as f:
            f.write(msg.rstrip("\n") + "\n")
    except Exception:
        # Swallow logging errors to avoid breaking UI updates
        pass


def write_value_file(run_dir: str, filename: str, value):
    """Persist a small value (json or text) under run_dir."""
    if not run_dir:
        return
    try:
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, filename)
        if isinstance(value, dict):
            # Only keep whitelist keys for best_metrics
            if filename == "best_metrics.json":
                value = {
                    k: value.get(k)
                    for k in (
                        "NMAE",
                        "R2",
                        "Accuracy",
                        "best_cv_loss",
                    )
                    if k in value
                }
            with open(path, "w") as f:
                json.dump(value, f)
        elif isinstance(value, list):
            with open(path, "w") as f:
                json.dump(value, f)
        else:
            with open(path, "w") as f:
                f.write(str(value))
    except Exception:
        pass


def read_value_file(run_dir: str, filename: str):
    """Read a small value (json or text) from run_dir."""
    if not run_dir:
        return None
    path = os.path.join(run_dir, filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            content = f.read()
        try:
            data = json.loads(content)
            # Normalize best_metrics to allowed keys only
            if filename == "best_metrics.json" and isinstance(data, dict):
                allowed_keys = {
                    "NMAE",
                    "R2",
                    "Accuracy",
                    "best_cv_loss",
                }
                data = {k: data.get(k) for k in allowed_keys if k in data}
            return data
        except Exception:
            try:
                return float(content)
            except Exception:
                return content
    except Exception:
        return None


def update_best_metrics(run_dir: str, updates: Dict):
    """
    Merge metric updates into best_metrics.json to avoid scattering separate files.
    Only whitelisted keys are persisted.
    """
    if not run_dir:
        return
    allowed_keys = {
        "NMAE",
        "R2",
        "Accuracy",
        "best_cv_loss",
    }
    existing = read_value_file(run_dir, "best_metrics.json")
    if not isinstance(existing, dict):
        existing = {}
    for k, v in updates.items():
        if k in allowed_keys:
            existing[k] = v
    write_value_file(run_dir, "best_metrics.json", existing)


def zip_run_dir(run_dir: str) -> Optional[str]:
    """Create a zip archive of the given run directory. Returns zip path."""
    if not run_dir or not os.path.isdir(run_dir):
        return None

    # Remove transient progress file before archiving
    progress_file = os.path.join(run_dir, "current_trial_number.txt")
    if os.path.exists(progress_file):
        try:
            os.remove(progress_file)
        except Exception:
            pass

    base_name = run_dir.rstrip(os.sep)
    zip_path = f"{base_name}.zip"
    # Remove existing archive to avoid stale contents
    if os.path.exists(zip_path):
        try:
            os.remove(zip_path)
        except Exception:
            pass
    try:
        shutil.make_archive(base_name, "zip", root_dir=run_dir)
        return zip_path
    except Exception:
        return None


def find_any_in_progress(base_dir: str):
    """
    Return the first IN_PROGRESS run folder found (status, path, run_id) or (None, None, None).
    Useful to detect a still-running job after a reload.
    """
    if not os.path.isdir(base_dir):
        return None, None, None

    for name in os.listdir(base_dir):
        if name.endswith("__IN_PROGRESS"):
            path = os.path.join(base_dir, name)
            if os.path.isdir(path):
                run_id = name.rsplit("__", 1)[0]
                return "IN_PROGRESS", path, run_id
    return None, None, None
