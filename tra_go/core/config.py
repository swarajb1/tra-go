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

    FLAVOUR: EnvFlavour = Field(default=EnvFlavour.dev)

    PROJECT_NAME: str = Field(default="tra_go")

    API_PREFIX: str = "/api"
    VERSION: str = "1.0"

    DOCS_URL: str = "/docs"

    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    WORKERS_COUNT: int = Field(default=10)

    DB_USER: str
    DB_PASS: str
    DB_HOST: str
    RO_DB_HOST: str
    DB_PORT: int
    DB_NAME: str
    DB_SCHEMA: str

    DEBUG: bool = Field(default=False)
    LOGGING_LEVEL: int = logging.DEBUG if DEBUG else logging.INFO

    AUTH_SECRET: str

    PROFILING_ENABLED: bool = False

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


# ------------------------------


# from core.config import settings

# {
#                 "main:app",
#             "host":settings.HOST,
#             "port":settings.PORT,
#             "workers":settings.WORKERS_COUNT,
#             "reload":settings.RELOAD,
#             "reload_includes":["policy.polar"],
#             "log_level":settings.LOGGING_LEVEL,

# }
