"""Общая настройка логирования проекта."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Включает единый компактный формат сообщений в консоли."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=False,
    )
