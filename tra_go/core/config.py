import logging
from enum import StrEnum, auto
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic import Field, computed_field, field_validator, model_validator
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

    # Dropout settings
    INITIAL_DROPOUT: float = Field(default=0, ge=0, le=1, description="Initial dropout")
    RECURRENT_DROPOUT: float = Field(default=0, ge=0, le=1, description="Recurrent dropout")

    # Trading parameters
    RISK_TO_REWARD_RATIO: float = Field(description="Risk to reward ratio for trades")
    SAFETY_FACTOR: int = Field(default=1, ge=1, description="Safety factor for trading")

    # Application settings
    DEBUG: bool = Field(default=False, description="Debug mode flag")
    USE_OPTIMIZED_DATA_LOADER: bool = Field(
        default=True,
        description="Use optimized tf.data pipeline for data loading",
    )

    TF_INTRA_OP_PARALLELISM_THREADS: int = Field(
        default=8,
        gt=-1,
        description="Number of intra-op parallelism threads for TensorFlow",
    )
    TF_INTER_OP_PARALLELISM_THREADS: int = Field(
        default=8,
        gt=-1,
        description="Number of inter-op parallelism threads for TensorFlow",
    )

    # Training enhancement settings
    EARLY_STOPPING_ENABLED: bool = Field(default=False, description="Enable early stopping during training")
    EARLY_STOPPING_PATIENCE: int = Field(default=100, gt=0, description="Patience for early stopping")
    EARLY_STOPPING_MIN_DELTA: float = Field(default=0.0001, ge=0, description="Minimum delta for early stopping")
    EARLY_STOPPING_RESTORE_BEST_WEIGHTS: bool = Field(
        default=True,
        description="Restore best weights on early stopping",
    )

    LR_DECAY_ENABLED: bool = Field(default=False, description="Enable learning rate decay on plateau")
    LR_DECAY_FACTOR: float = Field(default=0.5, gt=0, le=1, description="Factor by which to reduce learning rate")
    LR_DECAY_PATIENCE: int = Field(default=50, gt=0, description="Patience for learning rate decay")
    LR_DECAY_MIN_LR: float = Field(default=1e-7, gt=0, description="Minimum learning rate for decay")

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

    @model_validator(mode="after")
    def _validate_env_values(self) -> "GlobalConfig":
        """Model-level validation for environment variables (runs after field validation).

        This mirrors the assertions that used to live in `core.assertions` so
        that configuration is validated as soon as the settings object is
        created.
        """
        if self.SAFETY_FACTOR < 1:
            raise ValueError(f"Safety Factor should be greater than or equal to 1, got {self.SAFETY_FACTOR}")

        if not (0 < self.TEST_SIZE <= 0.5):
            raise ValueError(f"Test Size should be between 0 and 0.5, got {self.TEST_SIZE}")

        if not (0 < self.LEARNING_RATE <= 1):
            raise ValueError(f"Learning rate should be between 0 and 1, got {self.LEARNING_RATE}")

        if self.BATCH_SIZE <= 0:
            raise ValueError(f"Batch size should be positive, got {self.BATCH_SIZE}")

        if self.NUMBER_OF_EPOCHS <= 0:
            raise ValueError(f"Number of epochs should be positive, got {self.NUMBER_OF_EPOCHS}")

        if self.NUMBER_OF_NEURONS <= 0:
            raise ValueError(f"Number of neurons should be positive, got {self.NUMBER_OF_NEURONS}")

        if self.NUMBER_OF_LAYERS <= 0:
            raise ValueError(f"Number of layers should be positive, got {self.NUMBER_OF_LAYERS}")

        if not (0 <= self.INITIAL_DROPOUT <= 1):
            raise ValueError(f"Dropout should be between 0 and 1, got {self.INITIAL_DROPOUT}")

        return self

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
