from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MODEL_")  # pyright: ignore[reportUnannotatedClassAttribute]
    name: str = "gemma4"
    url: str = "http://localhost:11434"


settings = ModelConfig()
