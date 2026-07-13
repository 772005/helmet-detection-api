from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    roboflow_api_key: str
    roboflow_workspace: str = "harsh-chakravarti"
    roboflow_workflow_id: str = "helmet-detection-safety-monitoring-1783929349435"
    roboflow_api_url: str = "https://serverless.roboflow.com"

    # Resilience settings for the Roboflow workflow call. The inference-sdk
    # client has no built-in per-call HTTP timeout, so `roboflow_client`
    # enforces one itself using a thread with a hard deadline, combined with
    # a small retry-with-backoff loop around that.
    roboflow_request_timeout_seconds: float = 15.0
    roboflow_max_retries: int = 2
    roboflow_retry_backoff_seconds: float = 1.0

    # This workflow triggers a live safety check (and a Vision Events alert
    # on violations), so by default we do not want cached results.
    roboflow_use_cache: bool = False

    max_upload_mb: int = 5
    allowed_image_types: str = "image/jpeg,image/png,image/webp"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def allowed_types(self) -> set[str]:
        return {x.strip() for x in self.allowed_image_types.split(",") if x.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
