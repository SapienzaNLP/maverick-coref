import logging
import sys
import threading
from typing import Optional

from rich.console import Console

_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None

_default_log_level = logging.WARNING

# fancy logger
_console = Console()


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:

    global _default_handler

    with _lock:
        if _default_handler:
            # This library has already configured the library root logger.
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        _default_handler.flush = sys.stderr.flush

        # Apply our default configuration to the library root logger.
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_default_log_level)
        library_root_logger.propagate = False


def _reset_library_root_logger() -> None:

    global _default_handler

    with _lock:
        if not _default_handler:
            return

        library_root_logger = _get_library_root_logger()
        library_root_logger.removeHandler(_default_handler)
        library_root_logger.setLevel(logging.NOTSET)
        _default_handler = None


def set_log_level(level: int, logger: logging.Logger = None) -> None:
    """
    Set the log level.
    Args:
        level (:obj:`int`):
            Logging level.
        logger (:obj:`logging.Logger`):
            Logger to set the log level.
    """
    if not logger:
        _configure_library_root_logger()
        logger = _get_library_root_logger()
    logger.setLevel(level)


def get_logger(
    name: Optional[str] = None,
    level: Optional[int] = None,
    formatter: Optional[str] = None,
) -> logging.Logger:
    """
    Return a logger with the specified name.
    """

    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()

    if level is not None:
        set_log_level(level)

    if formatter is None:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
    _default_handler.setFormatter(formatter)

    return logging.getLogger(name)


def get_console_logger():
    return _console
