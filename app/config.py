from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_PATH: Path = Path("ml/saved_model/model.joblib")
    TARGET_NAMES_PATH: Path = Path("ml/saved_model/target_names.joblib")
    MODEL_METADATA_PATH: Path = Path("ml/saved_model/model_metadata.json")
    API_TITLE: str = "ML Model API"
    LOG_LEVEL: str = "INFO"
    MAX_BATCH_SIZE: int = 100

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
