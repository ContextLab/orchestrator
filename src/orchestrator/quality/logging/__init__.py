"""
Comprehensive logging and monitoring framework for orchestrator quality control.

This package provides advanced logging capabilities specifically designed for
quality control systems, including structured logging, external monitoring 
integration, and comprehensive quality event tracking.

Key Components:
- logger.py: Core structured logging framework with quality-aware features
- handlers.py: Specialized log handlers and formatters for quality control
- monitoring.py: External monitoring system integration (Prometheus, Grafana, etc.)

Features:
- Structured JSON logging with quality metrics
- Performance-optimized logging with buffering and async processing
- Integration with external monitoring systems (Prometheus, ELK, webhooks)
- Quality event tracking and alerting
- Configurable log levels and filtering
- Audit trail capabilities
- Context-aware logging with execution metadata

Usage Example:
    from orchestrator.quality.logging import get_logger, LogLevel, LogCategory
    
    logger = get_logger("my_component", LogLevel.INFO)
    
    with logger.context(execution_id="exec-123", pipeline_id="pipe-456"):
        logger.info("Starting validation process", category=LogCategory.VALIDATION)
        
        with logger.operation_timer("validation_step"):
            # Perform validation work
            pass
        
        logger.log_validation_result({
            "severity": "PASS",
            "quality_score": 0.95,
            "violations": []
        })

Integration with Validation System:
    The logging framework integrates seamlessly with the existing validation
    system to provide comprehensive quality control insights and monitoring.
"""

from .logger import (
    # Core logging classes
    StructuredLogger,
    LogLevel,
    LogCategory,
    LogContext,
    QualityEvent,
    
    # Logger registry functions
    get_logger,
    flush_all_loggers,
    get_all_quality_metrics
)

from .handlers import (
    # Formatters
    QualityJSONFormatter,
    
    # Specialized handlers
    QualityRotatingFileHandler,
    AsyncQualityHandler,
    PrometheusMetricsHandler,
    QualityEventStreamHandler,
    
    # Setup function
    create_quality_logging_setup
)

from .monitoring import (
    # Monitoring classes
    QualityMonitor,
    MonitoringAlert,
    AlertRule,
    
    # Backend implementations
    MonitoringBackend,
    PrometheusBackend,
    WebhookBackend,
    ElasticsearchBackend,
    
    # Setup function
    create_monitoring_setup
)

# Version information
__version__ = "1.0.0"
__author__ = "Orchestrator Quality Team"

# Public API
__all__ = [
    # Core logging
    "StructuredLogger",
    "LogLevel", 
    "LogCategory",
    "LogContext",
    "QualityEvent",
    "get_logger",
    "flush_all_loggers",
    "get_all_quality_metrics",
    
    # Handlers and formatters
    "QualityJSONFormatter",
    "QualityRotatingFileHandler",
    "AsyncQualityHandler", 
    "PrometheusMetricsHandler",
    "QualityEventStreamHandler",
    "create_quality_logging_setup",
    
    # Monitoring
    "QualityMonitor",
    "MonitoringAlert",
    "AlertRule",
    "MonitoringBackend",
    "PrometheusBackend",
    "WebhookBackend", 
    "ElasticsearchBackend",
    "create_monitoring_setup",
    
    # Package metadata
    "__version__"
]


# Package-level configuration and initialization
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


def configure_quality_logging(
    config_path: Optional[str] = None,
    log_level: LogLevel = LogLevel.INFO,
    enable_console: bool = True,
    enable_structured: bool = True,
    log_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Configure quality logging system with sensible defaults.
    
    Args:
        config_path: Path to YAML configuration file
        log_level: Default logging level
        enable_console: Enable console output
        enable_structured: Enable structured JSON logging
        log_dir: Directory for log files
    
    Returns:
        Dictionary containing configured loggers and handlers
    """
    # Load configuration if provided
    config = {}
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logging.warning(f"Failed to load logging config from {config_path}: {e}")
    
    # Set up log directory
    if not log_dir:
        log_dir = config.get('logging', {}).get('log_dir', 'logs/quality')
    
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create handlers based on configuration
    handlers = create_quality_logging_setup(
        log_dir=log_path,
        log_level=log_level,
        enable_prometheus=config.get('monitoring', {}).get('backends', {}).get('prometheus', {}).get('enabled', False),
        enable_async=config.get('outputs', {}).get('async', {}).get('enabled', True)
    )
    
    # Configure root quality logger
    quality_logger = get_logger(
        "quality_control",
        level=log_level,
        enable_structured=enable_structured,
        enable_performance_logging=True,
        enable_quality_logging=True
    )
    
    # Add handlers to standard logging system for integration
    root_logger = logging.getLogger("orchestrator.quality")
    root_logger.setLevel(log_level.value)
    
    if enable_console and 'console' in handlers:
        root_logger.addHandler(handlers['console'])
    
    if enable_structured and 'structured' in handlers:
        root_logger.addHandler(handlers['structured'])
    
    return {
        'quality_logger': quality_logger,
        'handlers': handlers,
        'config': config,
        'log_dir': str(log_path)
    }


# Auto-configure logging if environment variable is set
_auto_config_path = os.getenv('ORCHESTRATOR_LOGGING_CONFIG')
if _auto_config_path and Path(_auto_config_path).exists():
    try:
        configure_quality_logging(config_path=_auto_config_path)
    except Exception as e:
        logging.warning(f"Failed to auto-configure quality logging: {e}")


# Convenience functions for common use cases
def setup_development_logging(log_level: LogLevel = LogLevel.DEBUG) -> StructuredLogger:
    """Set up logging optimized for development."""
    configure_quality_logging(
        log_level=log_level,
        enable_console=True,
        enable_structured=True,
        log_dir="logs/dev"
    )
    return get_logger("development", level=log_level)


def setup_production_logging(config_path: str) -> StructuredLogger:
    """Set up logging optimized for production."""
    configure_quality_logging(
        config_path=config_path,
        log_level=LogLevel.INFO,
        enable_console=False,
        enable_structured=True
    )
    return get_logger("production", level=LogLevel.INFO)


def setup_testing_logging(log_level: LogLevel = LogLevel.WARNING) -> StructuredLogger:
    """Set up logging optimized for testing."""
    configure_quality_logging(
        log_level=log_level,
        enable_console=False,
        enable_structured=False,
        log_dir="logs/test"
    )
    return get_logger("testing", level=log_level)


# Integration hooks for validation system
def create_validation_logger(validation_component: str) -> StructuredLogger:
    """Create logger specifically for validation components."""
    return get_logger(
        f"validation.{validation_component}",
        level=LogLevel.INFO,
        enable_quality_logging=True,
        enable_performance_logging=True
    )


def create_monitoring_logger() -> StructuredLogger:
    """Create logger specifically for monitoring components."""
    return get_logger(
        "monitoring",
        level=LogLevel.DEBUG,
        enable_quality_logging=True,
        enable_performance_logging=True
    )