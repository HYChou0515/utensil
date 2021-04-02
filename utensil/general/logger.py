import logging
import os
import socket
from dataclasses import dataclass
from logging import handlers
from typing import Union, Iterable

from utensil import constant


def parse_log_level(level):
    if isinstance(level, str):
        if level.upper() == 'NOTSET':
            return logging.NOTSET
        elif level.upper() == 'DEBUG':
            return logging.DEBUG
        elif level.upper() == 'INFO':
            return logging.INFO
        elif level.upper() == 'WARNING':
            return logging.WARNING
        elif level.upper() == 'WARN':
            return logging.WARN
        elif level.upper() == 'ERROR':
            return logging.ERROR
        elif level.upper() == 'FATAL':
            return logging.FATAL
        elif level.upper() == 'CRITICAL':
            return logging.CRITICAL
    try:
        return int(level)
    except ValueError as e:
        raise ValueError(f"invalid log level: '{level}'") from e


@dataclass
class LoggerConfig:
    level: Union[str, int] = constant.LOG.get('Level')
    handlers: Iterable[logging.Handler] = None,
    format: str = '{asctime:s}.{msecs:06.0f} ' + constant.HOST_INFO.get('HostName') + ' ' + socket.gethostname() + \
                  ' {processName:s}({process:d}) {threadName:s}({thread:d}) {levelname:s} ' \
                  '({name:s}.{funcName:s}) {message:s} '
    style: str = '{'
    datefmt: str = '%Y-%m-%d %I:%M:%S'

    def __post_init__(self):

        _logger_handlers = []
        if constant.LOG.get('Stream', 'NOTSET') != 'NOTSET':
            handler = logging.StreamHandler()
            handler.setLevel(parse_log_level(constant.LOG.get('Stream', 'NOTSET')))
            _logger_handlers.append(handler)
        if constant.LOG.get('Syslog', 'NOTSET') != 'NOTSET':
            handler = handlers.SysLogHandler()
            handler.setLevel(parse_log_level(constant.LOG.get('Syslog', 'NOTSET')))
            _logger_handlers.append(handler)
        if constant.LOG.get('File', 'NOTSET') != 'NOTSET':
            log_file_name = constant.LOG.get('FilePrefix', 'log')
            if not os.path.isdir(os.path.dirname(log_file_name)):
                os.makedirs(os.path.dirname(log_file_name))
            handler = handlers.WatchedFileHandler(log_file_name)
            handler.setLevel(parse_log_level(constant.LOG.get('File', 'NOTSET')))
            _logger_handlers.append(handler)

        self.level = parse_log_level(self.level)
        if self.handlers == (None,):
            self.handlers = _logger_handlers


def get_logger(name, logger_config=None):
    if logger_config is None:
        logger_config = LoggerConfig()
    logger = logging.getLogger(name)
    logger.setLevel(logger_config.level)
    formatter = logging.Formatter(
        fmt=logger_config.format, datefmt=logger_config.datefmt, style=logger_config.style
    )
    for handler in logger_config.handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class _DummyLogger:

    def setLevel(self, level):
        pass

    def debug(self, msg, *args, **kwargs):
        pass

    def info(self, msg, *args, **kwargs):
        pass

    def warning(self, msg, *args, **kwargs):
        pass

    def warn(self, msg, *args, **kwargs):
        pass

    def error(self, msg, *args, **kwargs):
        pass

    def exception(self, msg, *args, exc_info=True, **kwargs):
        pass

    def critical(self, msg, *args, **kwargs):
        pass

    fatal = critical

    def log(self, level, msg, *args, **kwargs):
        pass

    def findCaller(self, stack_info=False, stacklevel=1):
        pass

    def makeRecord(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None):
        pass

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        pass

    def handle(self, record):
        pass

    def addHandler(self, hdlr):
        pass

    def removeHandler(self, hdlr):
        pass

    def hasHandlers(self):
        pass

    def callHandlers(self, record):
        pass

    def getEffectiveLevel(self):
        pass

    def isEnabledFor(self, level):
        pass

    def getChild(self, suffix):
        pass


DUMMY_LOGGER = _DummyLogger()
