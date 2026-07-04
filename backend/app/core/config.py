from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLINICAL_AI_", env_file=".env")

    queue_timeout_seconds: float = 30.0


settings = Settings()
