from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLINICAL_AI_", env_file=".env")

    queue_timeout_seconds: float = 30.0
    worker_count: int = 4

    # Hugging Face model repo id for the classifier, e.g.
    # "lxyuan/vit-xray-pneumonia-classification". Verify it exists on
    # huggingface.co before use. Empty string = use the placeholder TorchModel.
    model_id: str = "microsoft/resnet-50"


settings = Settings()
