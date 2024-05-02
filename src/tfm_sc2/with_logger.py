import logging
from pathlib import Path
from typing import Optional, Union


class WithLogger:
    _LOGGING_INITIALIZED = False
    def __init__(self, logger: Optional[logging.Logger] = None, log_name: str = None, log_level: Optional[int] = logging.DEBUG, **kwargs):
        super().__init__(**kwargs)

        if not WithLogger._LOGGING_INITIALIZED:
            WithLogger.init_logging()

        self._log_name = log_name or self.__class__.__name__
        self._logger = logger or logging.getLogger(self._log_name)
        self._logger.setLevel(log_level)

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @staticmethod
    def init_logging(stream_level: int = logging.INFO, file_name: Optional[Union[str, Path]] = None, file_level: int = logging.DEBUG, file_mode: str = 'w', file_encoding: str = 'utf-8'):
        handlers = []
        sh = logging.StreamHandler()
        log_format = (
            '[%(asctime)s] '
            '[%(levelname)s] [%(name)s] '
            '[%(filename)s:%(lineno)d] %(message)s'
        )
        date_format = "%Y-%m-%d %H:%M:%S"
        log_formatter = logging.Formatter(log_format, datefmt=date_format)
        sh.setFormatter(log_formatter)
        sh.setLevel(stream_level)

        handlers.append(sh)
        log_level = stream_level

        if file_name is not None:
            fh = logging.FileHandler(file_name, mode=file_mode, encoding=file_encoding)
            fh.setLevel(file_level)
            fh.setFormatter(log_formatter)
            log_level = min(file_level, stream_level)
            handlers.append(fh)

        logging.basicConfig(
            format=log_format,
            datefmt=date_format,
            level=log_level,
            handlers=handlers)

        logger = logging.getLogger()
        logger.handlers = handlers
        WithLogger._LOGGING_INITIALIZED = True