import logging
from enum import StrEnum, auto
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic import Field
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

    PROJECT_NAME: str = Field(default="tra_go")

    VERSION: str = "1.0"

    FLAVOUR: EnvFlavour = Field(default=EnvFlavour.dev)

    ZERODHA_ID: str
    PASSWORD: str

    API_KEY: str
    ACCESS_TOKEN: str

    NUMBER_OF_EPOCHS: int = Field(default=1000)
    BATCH_SIZE: int = Field(default=512)
    LEARNING_RATE: float = Field(default=0.0001)
    TEST_SIZE: float = Field(default=0.2)

    NUMBER_OF_NEURONS: int = Field(default=128)
    NUMBER_OF_LAYERS: int = Field(default=3)
    INITIAL_DROPOUT_PERCENT: float = Field(default=0)

    RISK_TO_REWARD_RATIO: float

    SAFETY_FACTOR: int = Field(default=1)

    DEBUG: bool = Field(default=False)
    LOGGING_LEVEL: int = logging.DEBUG if DEBUG else logging.INFO

    model_config = SettingsConfigDict()


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
