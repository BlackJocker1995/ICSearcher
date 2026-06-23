"""Unified loguru configuration for the whole project.

Importing :func:`setup_logging` and calling it once (typically at the top of
each pipeline entry point) ensures every module emits log records through the
same loguru sink. The legacy stdlib ``logging`` calls that remain in some
modules during the staged refactor are bridged into loguru so nothing is
silenced.
"""
import logging
import sys

from loguru import logger


class _InterceptHandler(logging.Handler):
    """Forward stdlib ``logging`` records into loguru.

    This is the official loguru recipe[1]. It lets us keep the remaining
    ``logging.info/debug/warning`` calls working while the migration to
    ``logger.*`` proceeds module by module.

    [1]: https://loguru.readthedocs.io/en/stable/resources/recipes.html
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Match the loguru level names. Loguru stores levels by their textual
        # name, so map the stdlib numeric level to the closest loguru level.
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find the caller frame so loguru reports the real source location.
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(debug: bool = False, log_file=None) -> None:
    """Configure the global loguru sink and bridge stdlib logging.

    Args:
        debug: When True the stderr sink runs at DEBUG, otherwise INFO.
        log_file: Optional path to also mirror records into a rotating file.
    """
    logger.remove()
    level = "DEBUG" if debug else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )
    if log_file:
        logger.add(
            log_file,
            level=level,
            rotation="10 MB",
            retention="5",
            encoding="utf-8",
        )

    # Bridge stdlib logging into loguru so the still-unmigrated modules are
    # not silenced (this was the bug introduced by commit caa146d).
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
