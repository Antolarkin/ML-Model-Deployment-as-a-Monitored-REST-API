from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_PATH: Path = Path("ml/saved_model/model.joblib")
    TARGET_NAMES_PATH: Path = Path("ml/saved_model/target_names.joblib")
    MODEL_METADATA_PATH: Path = Path("ml/saved_model/model_metadata.json")
    API_TITLE: str = "ML Model API"
    LOG_LEVEL: str = "INFO"
    MAX_BATCH_SIZE: int = 100
    API_KEY: str = "dev-secret-key"
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
