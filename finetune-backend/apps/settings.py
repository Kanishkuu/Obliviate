from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")
    api_log_level: str = Field("info", alias="API_LOG_LEVEL")
    allowed_origins: str = Field("*", alias="ALLOWED_ORIGINS")

    artifacts_dir: str = Field("/app/artifacts", alias="ARTIFACTS_DIR")
    uploads_dir: str = Field("/app/uploads", alias="UPLOADS_DIR")

    redis_url: str = Field("redis://localhost:6379/0", alias="REDIS_URL")
    rq_queue: str = Field("finetune", alias="RQ_QUEUE")

    mixed_precision: str = Field("bf16", alias="MIXED_PRECISION")

    class Config:
        env_file = ".env"

settings = Settings()
