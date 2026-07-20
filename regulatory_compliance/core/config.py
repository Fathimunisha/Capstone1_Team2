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
    UPLOAD_FOLDER: str
    LOG_LEVEL: str
    TOP_K_RESULTS: int
    MODEL_NAME: str
    OPENAI_MODEL: str
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 100
    DATABASE_HOST: str
    DATABASE_PORT: int
    DATABASE_NAME: str
    DATABASE_USER: str
    DATABASE_PASSWORD: str
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"dbname={self.DATABASE_NAME} "
            f"user={self.DATABASE_USER} "
            f"password={self.DATABASE_PASSWORD} "
            f"host={self.DATABASE_HOST} "
            f"port={self.DATABASE_PORT}"
        )


settings = Settings()
