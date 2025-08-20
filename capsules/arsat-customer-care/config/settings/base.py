import logging
import pathlib
from ast import literal_eval

import decouple
from pydantic import BaseConfig
from pydantic_settings import BaseSettings

ROOT_DIR: pathlib.Path = pathlib.Path(
    __file__
).parent.parent.parent.resolve()


class BackendBaseSettings(BaseSettings):
    # Maximo settings
    MAXIMO_BASE_URL: str = decouple.config("MAXIMO_BASE_URL", default=None, cast=str)
    MAXIMO_REQUEST_TIMEOUT: float = decouple.config(
        "MAXIMO_REQUEST_TIMEOUT", default=10.0, cast=float
    )
    MAXIMO_HTTP_VERIFY_SSL: bool = decouple.config(
        "MAXIMO_HTTP_VERIFY_SSL", default=True, cast=bool
    )
    MAXIMO_USER_ID: str = decouple.config("MAXIMO_USER_ID", default=None, cast=str)
    MAXIMO_PASSWD: str = decouple.config("MAXIMO_PASSWD", default=None, cast=str)
    MAXIMO_OPEN_TICKET_STATUSES: list[str] = decouple.config(
        "MAXIMO_OPEN_TICKET_STATUSES", default="", cast=decouple.Csv()
    )

    class Config(BaseConfig):
        case_sensitive: bool = True
        env_file: str = f"{str(ROOT_DIR)}/.env"
        validate_assignment: bool = True
