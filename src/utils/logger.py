"""
Structured logging module.

Provides JSON-formatted logging with:
- Configurable log levels
- Log rotation
- Sensitive data filtering
- Context-aware logging
"""

import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from structlog.types import EventDict, Processor

# Sensitive fields to filter from logs
SENSITIVE_FIELDS = {
    "api_key",
    "password",
    "token",
    "secret",
    "authorization",
    "ngrok_auth_token",
}


def filter_sensitive_data(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """
    Filter sensitive data from log entries.
    
    Replaces sensitive field values with [REDACTED].
    """
    for key in list(event_dict.keys()):
        if any(sensitive in key.lower() for sensitive in SENSITIVE_FIELDS):
            event_dict[key] = "[REDACTED]"
        elif isinstance(event_dict[key], dict):
            # Recursively filter nested dicts
            for nested_key in list(event_dict[key].keys()):
                if any(sensitive in nested_key.lower() for sensitive in SENSITIVE_FIELDS):
                    event_dict[key][nested_key] = "[REDACTED]"
    return event_dict


def add_timestamp(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add ISO format timestamp to log entries."""
    event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return event_dict


def add_log_level(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add log level to event dict."""
    event_dict["level"] = method_name.upper()
    return event_dict


class ContextLogger:
    """
    Context-aware logger wrapper.
    
    Allows binding context that persists across log calls.
    """

    def __init__(self, logger: structlog.BoundLogger):
        self._logger = logger
        self._context: Dict[str, Any] = {}

    def bind(self, **kwargs: Any) -> "ContextLogger":
        """Bind additional context to the logger."""
        self._context.update(kwargs)
        self._logger = self._logger.bind(**kwargs)
        return self

    def unbind(self, *keys: str) -> "ContextLogger":
        """Remove context keys from the logger."""
        for key in keys:
            self._context.pop(key, None)
        self._logger = self._logger.unbind(*keys)
        return self

    @contextmanager
    def context(self, **kwargs: Any):
        """Temporary context binding."""
        original_context = self._context.copy()
        try:
            self.bind(**kwargs)
            yield self
        finally:
            # Restore original context
            for key in kwargs:
                self.unbind(key)
            self._context = original_context

    def debug(self, event: str, **kwargs: Any) -> None:
        self._logger.debug(event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self._logger.info(event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._logger.warning(event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._logger.error(event, **kwargs)

    def exception(self, event: str, **kwargs: Any) -> None:
        self._logger.exception(event, **kwargs)

    def critical(self, event: str, **kwargs: Any) -> None:
        self._logger.critical(event, **kwargs)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    is_development: bool = True,
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None for console only)
        json_format: Use JSON format for logs
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        is_development: Enable development-friendly formatting
    """
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure processors
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        add_log_level,
        add_timestamp,
        filter_sensitive_data,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if is_development and not json_format:
        # Development: colorful console output
        shared_processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        # Production: JSON format
        shared_processors.append(
            structlog.processors.format_exc_info,
        )
        shared_processors.append(
            structlog.processors.JSONRenderer()
        )

    # Configure structlog
    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    handlers: list[logging.Handler] = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    handlers.append(console_handler)

    # File handler with rotation
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        format="%(message)s",
        force=True,
    )

    # Set level for third-party loggers
    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        logging.getLogger(logger_name).setLevel(numeric_level)

    # Reduce noise from some libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str = "frc_rag") -> ContextLogger:
    """
    Get a context-aware logger instance.
    
    Args:
        name: Logger name (module name recommended)
        
    Returns:
        ContextLogger instance
    """
    return ContextLogger(structlog.get_logger(name))


# Convenience function for quick logging
_default_logger: Optional[ContextLogger] = None


def logger() -> ContextLogger:
    """Get the default application logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger("frc_rag")
    return _default_logger
