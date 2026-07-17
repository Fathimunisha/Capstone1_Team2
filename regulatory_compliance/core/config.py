from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from .env
    """

    APP_NAME: str
    APP_VERSION: str
    HOST: str
    PORT: int
    OPENAI_API_KEY: str
    PG_CONNECTION_STRING: str
    UPLOAD_FOLDER: str
    LOG_LEVEL: str
    TOP_K_RESULTS: int
    MODEL_NAME: str
    EMBEDDING_MODEL: str
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
