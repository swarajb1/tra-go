import logging
from enum import StrEnum, auto
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.config import Config

ENV_FILEPATH: Path = Path(find_dotenv())

load_dotenv(ENV_FILEPATH)
config: Config = Config(ENV_FILEPATH)


class EnvFlavour(StrEnum):
    dev = auto()
    stg = auto()
    prod = auto()


class GlobalConfig(BaseSettings):
    """Global configurations."""

    PROJECT_NAME: str = Field(default="tra_go", description="Project name")

    VERSION: str = Field(default="1.0", description="Application version")

    FLAVOUR: EnvFlavour = Field(default=EnvFlavour.dev, description="Environment flavour")

    # Trading credentials - required fields
    ZERODHA_ID: str = Field(description="Zerodha user ID")
    PASSWORD: str = Field(description="Trading password")
    API_KEY: str = Field(description="API key for trading platform")
    ACCESS_TOKEN: str = Field(description="Access token for API")

    # Model training parameters
    NUMBER_OF_EPOCHS: int = Field(default=1000, gt=0, description="Number of training epochs")
    BATCH_SIZE: int = Field(default=512, gt=0, description="Training batch size")
    LEARNING_RATE: float = Field(default=0.0001, gt=0, le=1, description="Learning rate for model training")
    TEST_SIZE: float = Field(default=0.2, gt=0, le=0.5, description="Test set size ratio")

    # Neural network architecture
    NUMBER_OF_NEURONS: int = Field(default=128, gt=0, description="Number of neurons per layer")
    NUMBER_OF_LAYERS: int = Field(default=3, gt=0, description="Number of hidden layers")
    INITIAL_DROPOUT_PERCENT: float = Field(default=0, ge=0, le=100, description="Initial dropout percentage")

    # Trading parameters
    RISK_TO_REWARD_RATIO: float = Field(description="Risk to reward ratio for trades")
    SAFETY_FACTOR: int = Field(default=1, ge=1, description="Safety factor for trading")

    # Application settings
    DEBUG: bool = Field(default=False, description="Debug mode flag")

    @computed_field
    @property
    def LOGGING_LEVEL(self) -> int:
        """Compute logging level based on debug flag."""
        return logging.DEBUG if self.DEBUG else logging.INFO

    @field_validator("RISK_TO_REWARD_RATIO")
    @classmethod
    def validate_risk_reward_ratio(cls, v: float) -> float:
        """Validate risk to reward ratio."""
        if v <= 0:
            raise ValueError("Risk to reward ratio must be positive")
        return v

    model_config = SettingsConfigDict(
        env_file=ENV_FILEPATH,
        env_file_encoding="utf-8",
        case_sensitive=True,
        validate_assignment=True,
    )


class DevConfig(GlobalConfig):
    """Development configurations."""

    DEBUG: bool = True


class StageConfig(GlobalConfig):
    """Staging configurations."""


class ProdConfig(GlobalConfig):
    """Production configurations."""


class FactoryConfig:
    """Returns a config instance depending on the env FLAVOUR variable."""

    def __init__(self, flavour: EnvFlavour):
        self.FLAVOUR = flavour

    def __call__(self) -> GlobalConfig:
        config: type[GlobalConfig] = GlobalConfig

        if self.FLAVOUR == EnvFlavour.dev:
            config = DevConfig

        elif self.FLAVOUR == EnvFlavour.stg:
            config = StageConfig

        elif self.FLAVOUR == EnvFlavour.prod:
            config = ProdConfig

        return config.model_validate({})


settings: GlobalConfig = FactoryConfig(config("FLAVOUR", default=EnvFlavour.dev, cast=EnvFlavour))()

__all__ = ["settings"]
