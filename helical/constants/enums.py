from enum import Enum

class LoggingType(str, Enum):
    FILE='FILE'
    CONSOLE='CONSOLE'
    FILE_AND_CONSOLE='FILE_AND_CONSOLE'
    NOTSET='NOTSET'

class LoggingLevel(str, Enum):
    CRITICAL = "CRITICAL"
    FATAL = "FATAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    WARN = "WARN"
    INFO = "INFO"
    DEBUG = "DEBUG"
    NOTSET = "NOTSET"