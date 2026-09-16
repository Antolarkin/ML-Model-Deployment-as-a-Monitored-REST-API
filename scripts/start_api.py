import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings

MODEL_TEMPLATE_DIR = Path("/app/ml/model_template")


def copy_missing_model_files() -> None:
    model_files = {
        settings.MODEL_PATH: "model.joblib",
        settings.TARGET_NAMES_PATH: "target_names.joblib",
        settings.MODEL_METADATA_PATH: "model_metadata.json",
    }

    for target_path, template_name in model_files.items():
        if target_path.exists():
            continue

        template_path = MODEL_TEMPLATE_DIR / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Model template not found at {template_path}")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(template_path, target_path)


def main() -> None:
    copy_missing_model_files()
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
    )


if __name__ == "__main__":
    main()
