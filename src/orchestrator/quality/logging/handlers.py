"""
Specialized log handlers and formatters for quality control system.

This module provides advanced logging handlers that integrate with external
monitoring systems, support different output formats, and provide specialized
functionality for quality control and performance monitoring.

Key Components:
- JSON formatters for structured logging
- File rotation handlers with quality-aware naming
- External system handlers (Prometheus, Grafana, etc.)
- Performance-optimized handlers for high-volume logging
- Quality event handlers for validation and monitoring integration
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import asdict
import threading
import gzip
import shutil
from concurrent.futures import ThreadPoolExecutor
import queue

from .logger import LogLevel, LogCategory, LogContext, QualityEvent


class QualityJSONFormatter(logging.Formatter):
    """
    Advanced JSON formatter with quality control enhancements.
    
    Formats log records as structured JSON with quality metrics,
    performance data, and external system compatibility.
    """
    
    def __init__(
        self,
        include_quality_metrics: bool = True,
        include_performance_metrics: bool = True,
        include_stack_trace: bool = True,
        timestamp_format: str = "iso",
        exclude_fields: Optional[List[str]] = None
    ):
        super().__init__()
        self.include_quality_metrics = include_quality_metrics
        self.include_performance_metrics = include_performance_metrics
        self.include_stack_trace = include_stack_trace
        self.timestamp_format = timestamp_format
        self.exclude_fields = exclude_fields or []

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        try:
            # Start with basic log record data
            log_data = {
                'timestamp': self._format_timestamp(record.created),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'thread': record.thread,
                'thread_name': record.threadName,
                'process': record.process,
            }
            
            # Add exception information if present
            if record.exc_info and self.include_stack_trace:
                log_data['exception'] = {
                    'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                    'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                    'traceback': self.formatException(record.exc_info) if record.exc_info else None
                }
            
            # Add structured data from record if available
            if hasattr(record, 'structured_data'):
                log_data.update(record.structured_data)
            
            # Add quality metrics if available and enabled
            if self.include_quality_metrics and hasattr(record, 'quality_metrics'):
                log_data['quality_metrics'] = record.quality_metrics
            
            # Add performance metrics if available and enabled
            if self.include_performance_metrics and hasattr(record, 'performance_metrics'):
                log_data['performance_metrics'] = record.performance_metrics
            
            # Remove excluded fields
            for field in self.exclude_fields:
                log_data.pop(field, None)
            
            return json.dumps(log_data, default=self._json_serializer, separators=(',', ':'))
            
        except Exception as e:
            # Fallback to basic formatting to prevent log loss
            return f'{{"timestamp": "{self._format_timestamp(record.created)}", "level": "ERROR", "message": "Failed to format log record: {e}", "original_message": "{record.getMessage()}"}}')

    def _format_timestamp(self, timestamp: float) -> str:
        """Format timestamp according to configured format."""
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        
        if self.timestamp_format == "iso":
            return dt.isoformat()
        elif self.timestamp_format == "unix":
            return str(timestamp)
        elif self.timestamp_format == "rfc3339":
            return dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        else:
            return dt.strftime(self.timestamp_format)

    def _json_serializer(self, obj) -> Any:
        """Custom JSON serializer for complex objects."""
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        elif isinstance(obj, Exception):
            return str(obj)
        else:
            return str(obj)


class QualityRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    Enhanced rotating file handler with quality-aware rotation.
    
    Provides intelligent log rotation based on quality events,
    time periods, and file sizes with compression support.
    """
    
    def __init__(
        self,
        filename: str,
        maxBytes: int = 100 * 1024 * 1024,  # 100MB default
        backupCount: int = 10,
        compress_rotated: bool = True,
        rotate_on_quality_events: bool = True,
        quality_event_threshold: int = 1000,
        time_based_rotation: Optional[str] = None  # daily, weekly, monthly
    ):
        super().__init__(filename, maxBytes=maxBytes, backupCount=backupCount)
        self.compress_rotated = compress_rotated
        self.rotate_on_quality_events = rotate_on_quality_events
        self.quality_event_threshold = quality_event_threshold
        self.time_based_rotation = time_based_rotation
        
        self._quality_event_count = 0
        self._last_rotation_time = time.time()
        
    def shouldRollover(self, record: logging.LogRecord) -> bool:
        """Enhanced rollover logic with quality event consideration."""
        # Standard size-based rollover
        if super().shouldRollover(record):
            return True
            
        # Quality event-based rollover
        if self.rotate_on_quality_events and hasattr(record, 'quality_event'):
            self._quality_event_count += 1
            if self._quality_event_count >= self.quality_event_threshold:
                self._quality_event_count = 0
                return True
        
        # Time-based rollover
        if self.time_based_rotation:
            current_time = time.time()
            rotation_interval = self._get_rotation_interval()
            if current_time - self._last_rotation_time >= rotation_interval:
                return True
                
        return False
    
    def doRollover(self):
        """Enhanced rollover with compression support."""
        super().doRollover()
        self._last_rotation_time = time.time()
        
        # Compress rotated files if enabled
        if self.compress_rotated and self.backupCount > 0:
            self._compress_backup_files()
    
    def _get_rotation_interval(self) -> float:
        """Get rotation interval in seconds based on configuration."""
        intervals = {
            'daily': 24 * 60 * 60,
            'weekly': 7 * 24 * 60 * 60,
            'monthly': 30 * 24 * 60 * 60
        }
        return intervals.get(self.time_based_rotation, 24 * 60 * 60)
    
    def _compress_backup_files(self):
        """Compress backup log files to save disk space."""
        base_filename = self.baseFilename
        for i in range(self.backupCount, 0, -1):
            backup_name = f"{base_filename}.{i}"
            compressed_name = f"{backup_name}.gz"
            
            if Path(backup_name).exists() and not Path(compressed_name).exists():
                try:
                    with open(backup_name, 'rb') as f_in:
                        with gzip.open(compressed_name, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    Path(backup_name).unlink()
                except Exception as e:
                    # Log compression failure but don't crash
                    print(f"Warning: Failed to compress {backup_name}: {e}")


class AsyncQualityHandler(logging.Handler):
    """
    Asynchronous logging handler for high-performance quality logging.
    
    Provides non-blocking logging with background processing for
    high-volume quality events and performance metrics.
    """
    
    def __init__(
        self,
        target_handler: logging.Handler,
        queue_size: int = 10000,
        num_workers: int = 2,
        batch_size: int = 100,
        flush_interval: float = 1.0
    ):
        super().__init__()
        self.target_handler = target_handler
        self.queue_size = queue_size
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Create queue and worker threads
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._shutdown_event = threading.Event()
        self._workers: List[threading.Thread] = []
        
        # Start worker threads
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self._workers.append(worker)
        
        # Start batch processor
        self._batch_processor = threading.Thread(target=self._batch_processor_loop, daemon=True)
        self._batch_processor.start()

    def emit(self, record: logging.LogRecord):
        """Add record to async processing queue."""
        try:
            if not self._shutdown_event.is_set():
                self._queue.put_nowait(record)
        except queue.Full:
            # Handle queue overflow by dropping oldest records
            try:
                self._queue.get_nowait()  # Remove oldest
                self._queue.put_nowait(record)  # Add new
            except queue.Empty:
                pass  # Queue became empty, just continue

    def _worker_loop(self):
        """Worker thread loop for processing log records."""
        batch = []
        last_flush = time.time()
        
        while not self._shutdown_event.is_set():
            try:
                # Get record with timeout
                record = self._queue.get(timeout=0.1)
                batch.append(record)
                
                # Process batch when full or on timeout
                if (len(batch) >= self.batch_size or 
                    time.time() - last_flush >= self.flush_interval):
                    self._process_batch(batch)
                    batch.clear()
                    last_flush = time.time()
                    
            except queue.Empty:
                # Process remaining batch on timeout
                if batch and time.time() - last_flush >= self.flush_interval:
                    self._process_batch(batch)
                    batch.clear()
                    last_flush = time.time()
            except Exception as e:
                # Log processing error but continue
                print(f"Error in async log worker: {e}")
        
        # Process remaining batch on shutdown
        if batch:
            self._process_batch(batch)

    def _batch_processor_loop(self):
        """Background processor for batched operations."""
        while not self._shutdown_event.is_set():
            try:
                # Perform periodic maintenance
                if hasattr(self.target_handler, 'flush'):
                    self.target_handler.flush()
                
                time.sleep(self.flush_interval)
            except Exception as e:
                print(f"Error in batch processor: {e}")

    def _process_batch(self, batch: List[logging.LogRecord]):
        """Process a batch of log records."""
        for record in batch:
            try:
                self.target_handler.emit(record)
            except Exception as e:
                # Handle individual record errors
                print(f"Error processing log record: {e}")

    def close(self):
        """Shutdown async handler gracefully."""
        self._shutdown_event.set()
        
        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=5.0)
        
        if self._batch_processor.is_alive():
            self._batch_processor.join(timeout=5.0)
        
        # Process remaining queued records
        remaining_records = []
        while True:
            try:
                remaining_records.append(self._queue.get_nowait())
            except queue.Empty:
                break
        
        if remaining_records:
            self._process_batch(remaining_records)
        
        # Close target handler
        if hasattr(self.target_handler, 'close'):
            self.target_handler.close()
        
        super().close()


class PrometheusMetricsHandler(logging.Handler):
    """
    Logging handler that exports quality metrics to Prometheus.
    
    Converts quality events and performance metrics to Prometheus
    metrics format for external monitoring system integration.
    """
    
    def __init__(
        self,
        metrics_port: int = 8090,
        metrics_path: str = '/metrics',
        enable_histogram_metrics: bool = True,
        enable_counter_metrics: bool = True,
        enable_gauge_metrics: bool = True
    ):
        super().__init__()
        self.metrics_port = metrics_port
        self.metrics_path = metrics_path
        self.enable_histogram_metrics = enable_histogram_metrics
        self.enable_counter_metrics = enable_counter_metrics
        self.enable_gauge_metrics = enable_gauge_metrics
        
        # Initialize metrics collectors
        self._metrics = {
            'log_entries_total': 0,
            'quality_events_total': 0,
            'validation_failures_total': 0,
            'execution_duration_seconds': [],
            'quality_scores': [],
            'error_counts': {}
        }
        self._metrics_lock = threading.Lock()

    def emit(self, record: logging.LogRecord):
        """Process log record and update Prometheus metrics."""
        with self._metrics_lock:
            self._metrics['log_entries_total'] += 1
            
            # Count errors by level
            level = record.levelname
            if level not in self._metrics['error_counts']:
                self._metrics['error_counts'][level] = 0
            self._metrics['error_counts'][level] += 1
            
            # Process quality events
            if hasattr(record, 'quality_event'):
                self._metrics['quality_events_total'] += 1
                
                quality_event = record.quality_event
                if quality_event.quality_score is not None:
                    self._metrics['quality_scores'].append(quality_event.quality_score)
            
            # Process validation results
            if hasattr(record, 'validation_result'):
                validation_result = record.validation_result
                if validation_result.get('severity') in ['FAIL', 'CRITICAL']:
                    self._metrics['validation_failures_total'] += 1
            
            # Process performance metrics
            if hasattr(record, 'performance_metrics'):
                perf_metrics = record.performance_metrics
                if 'duration_ms' in perf_metrics:
                    duration_seconds = perf_metrics['duration_ms'] / 1000.0
                    self._metrics['execution_duration_seconds'].append(duration_seconds)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics in Prometheus format."""
        with self._metrics_lock:
            metrics_output = []
            
            # Counter metrics
            if self.enable_counter_metrics:
                metrics_output.extend([
                    f'# HELP orchestrator_log_entries_total Total number of log entries',
                    f'# TYPE orchestrator_log_entries_total counter',
                    f'orchestrator_log_entries_total {self._metrics["log_entries_total"]}',
                    f'# HELP orchestrator_quality_events_total Total number of quality events',
                    f'# TYPE orchestrator_quality_events_total counter', 
                    f'orchestrator_quality_events_total {self._metrics["quality_events_total"]}',
                    f'# HELP orchestrator_validation_failures_total Total validation failures',
                    f'# TYPE orchestrator_validation_failures_total counter',
                    f'orchestrator_validation_failures_total {self._metrics["validation_failures_total"]}',
                ])
                
                # Error counts by level
                for level, count in self._metrics['error_counts'].items():
                    metrics_output.extend([
                        f'orchestrator_log_entries_by_level{{level="{level}"}} {count}'
                    ])
            
            # Gauge metrics
            if self.enable_gauge_metrics and self._metrics['quality_scores']:
                avg_quality = sum(self._metrics['quality_scores']) / len(self._metrics['quality_scores'])
                metrics_output.extend([
                    f'# HELP orchestrator_quality_score_avg Average quality score',
                    f'# TYPE orchestrator_quality_score_avg gauge',
                    f'orchestrator_quality_score_avg {avg_quality:.4f}'
                ])
            
            # Histogram metrics
            if self.enable_histogram_metrics and self._metrics['execution_duration_seconds']:
                durations = sorted(self._metrics['execution_duration_seconds'])
                count = len(durations)
                
                # Calculate percentiles
                p50 = durations[int(0.5 * count)] if count > 0 else 0
                p95 = durations[int(0.95 * count)] if count > 0 else 0
                p99 = durations[int(0.99 * count)] if count > 0 else 0
                
                metrics_output.extend([
                    f'# HELP orchestrator_execution_duration_seconds Execution duration histogram',
                    f'# TYPE orchestrator_execution_duration_seconds histogram',
                    f'orchestrator_execution_duration_seconds_bucket{{le="0.1"}} {len([d for d in durations if d <= 0.1])}',
                    f'orchestrator_execution_duration_seconds_bucket{{le="1.0"}} {len([d for d in durations if d <= 1.0])}',
                    f'orchestrator_execution_duration_seconds_bucket{{le="10.0"}} {len([d for d in durations if d <= 10.0])}',
                    f'orchestrator_execution_duration_seconds_bucket{{le="+Inf"}} {count}',
                    f'orchestrator_execution_duration_seconds_sum {sum(durations):.4f}',
                    f'orchestrator_execution_duration_seconds_count {count}',
                    f'# Quality score percentiles',
                    f'orchestrator_execution_duration_seconds_p50 {p50:.4f}',
                    f'orchestrator_execution_duration_seconds_p95 {p95:.4f}',
                    f'orchestrator_execution_duration_seconds_p99 {p99:.4f}',
                ])
            
            return '\n'.join(metrics_output)


class QualityEventStreamHandler(logging.StreamHandler):
    """
    Specialized stream handler for quality events.
    
    Provides real-time quality event streaming with filtering
    and formatting optimized for quality control monitoring.
    """
    
    def __init__(
        self,
        stream=None,
        quality_level_filter: Optional[str] = None,
        event_type_filter: Optional[List[str]] = None,
        include_recommendations: bool = True,
        colorize_output: bool = True
    ):
        super().__init__(stream)
        self.quality_level_filter = quality_level_filter
        self.event_type_filter = event_type_filter or []
        self.include_recommendations = include_recommendations
        self.colorize_output = colorize_output
        
        # Color codes for different quality levels
        self._colors = {
            'PASS': '\033[92m',      # Green
            'WARNING': '\033[93m',   # Yellow
            'FAIL': '\033[91m',      # Red
            'CRITICAL': '\033[95m',  # Magenta
            'RESET': '\033[0m'       # Reset
        }

    def emit(self, record: logging.LogRecord):
        """Emit quality-focused log record."""
        # Filter by quality level
        if (self.quality_level_filter and 
            hasattr(record, 'quality_level') and 
            record.quality_level != self.quality_level_filter):
            return
        
        # Filter by event type
        if (self.event_type_filter and 
            hasattr(record, 'event_type') and 
            record.event_type not in self.event_type_filter):
            return
        
        # Format quality-specific message
        formatted_record = self._format_quality_record(record)
        super().emit(formatted_record)

    def _format_quality_record(self, record: logging.LogRecord) -> logging.LogRecord:
        """Format record with quality-specific information."""
        # Create new record to avoid modifying original
        new_record = logging.LogRecord(
            record.name, record.levelno, record.pathname, record.lineno,
            record.msg, record.args, record.exc_info, record.funcName, record.stack_info
        )
        
        # Add quality-specific formatting
        message_parts = [f"[{record.levelname}]"]
        
        if hasattr(record, 'quality_level'):
            color = self._colors.get(record.quality_level, '') if self.colorize_output else ''
            reset = self._colors.get('RESET', '') if self.colorize_output else ''
            message_parts.append(f"{color}Quality: {record.quality_level}{reset}")
        
        if hasattr(record, 'validation_score'):
            message_parts.append(f"Score: {record.validation_score:.2f}")
        
        if hasattr(record, 'event_type'):
            message_parts.append(f"Event: {record.event_type}")
        
        message_parts.append(record.getMessage())
        
        # Add recommendations if available and enabled
        if (self.include_recommendations and 
            hasattr(record, 'quality_event') and 
            record.quality_event.recommendations):
            message_parts.append("Recommendations:")
            for rec in record.quality_event.recommendations:
                message_parts.append(f"  - {rec}")
        
        new_record.msg = " | ".join(message_parts)
        new_record.args = ()
        
        return new_record


def create_quality_logging_setup(
    log_dir: Path,
    log_level: LogLevel = LogLevel.INFO,
    enable_prometheus: bool = True,
    enable_async: bool = True,
    prometheus_port: int = 8090
) -> Dict[str, logging.Handler]:
    """
    Create comprehensive quality logging setup with multiple handlers.
    
    Returns dictionary of configured handlers for different logging needs.
    """
    handlers = {}
    
    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON file handler for structured logging
    json_formatter = QualityJSONFormatter(
        include_quality_metrics=True,
        include_performance_metrics=True
    )
    
    json_file_handler = QualityRotatingFileHandler(
        str(log_dir / "quality.json"),
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=10,
        compress_rotated=True,
        rotate_on_quality_events=True
    )
    json_file_handler.setFormatter(json_formatter)
    json_file_handler.setLevel(log_level.value)
    
    if enable_async:
        handlers['structured'] = AsyncQualityHandler(json_file_handler)
    else:
        handlers['structured'] = json_file_handler
    
    # Console handler for real-time monitoring
    console_handler = QualityEventStreamHandler(
        colorize_output=True,
        include_recommendations=True
    )
    console_handler.setLevel(LogLevel.INFO.value)
    handlers['console'] = console_handler
    
    # Prometheus metrics handler
    if enable_prometheus:
        prometheus_handler = PrometheusMetricsHandler(
            metrics_port=prometheus_port,
            enable_histogram_metrics=True,
            enable_counter_metrics=True,
            enable_gauge_metrics=True
        )
        prometheus_handler.setLevel(LogLevel.DEBUG.value)
        handlers['prometheus'] = prometheus_handler
    
    return handlers