"""Performance monitoring and metrics collection for pipeline testing."""

import json
import logging
import os
import resource
import sqlite3
import time
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Tuple, Union
import threading
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class ResourceMetrics:
    """Resource usage metrics for a specific time period."""
    
    # CPU metrics
    cpu_percent: float = 0.0
    cpu_count: int = 0
    
    # Memory metrics
    memory_used_mb: float = 0.0
    memory_peak_mb: float = 0.0
    memory_percent: float = 0.0
    
    # Disk I/O metrics
    disk_read_bytes: int = 0
    disk_write_bytes: int = 0
    
    # Network I/O metrics (if available)
    network_sent_bytes: int = 0
    network_recv_bytes: int = 0
    
    # Process-specific metrics
    process_memory_rss: int = 0
    process_memory_vms: int = 0
    process_cpu_times: float = 0.0
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionMetrics:
    """Performance metrics for a pipeline execution."""
    
    # Basic execution metrics
    pipeline_name: str
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time_seconds: float = 0.0
    success: bool = False
    
    # API and model usage metrics
    api_calls_count: int = 0
    total_tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls_breakdown: Dict[str, int] = field(default_factory=dict)
    estimated_cost_usd: float = 0.0
    
    # Resource usage metrics
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    average_cpu_percent: float = 0.0
    total_disk_io_bytes: int = 0
    total_network_io_bytes: int = 0
    
    # Output metrics
    output_files_count: int = 0
    output_total_size_bytes: int = 0
    output_file_types: Dict[str, int] = field(default_factory=dict)
    
    # Quality metrics (from other streams)
    quality_score: Optional[float] = None
    template_resolution_issues: int = 0
    
    # Performance indicators
    throughput_files_per_second: float = 0.0
    throughput_tokens_per_second: float = 0.0
    cost_per_output_file: float = 0.0
    
    # Resource usage history (sampled during execution)
    resource_samples: List[ResourceMetrics] = field(default_factory=list)
    
    def calculate_derived_metrics(self):
        """Calculate derived performance metrics."""
        if self.end_time and self.start_time:
            self.execution_time_seconds = (self.end_time - self.start_time).total_seconds()
        
        # Calculate throughput metrics
        if self.execution_time_seconds > 0:
            if self.output_files_count > 0:
                self.throughput_files_per_second = self.output_files_count / self.execution_time_seconds
            if self.total_tokens_used > 0:
                self.throughput_tokens_per_second = self.total_tokens_used / self.execution_time_seconds
            if self.output_files_count > 0 and self.estimated_cost_usd > 0:
                self.cost_per_output_file = self.estimated_cost_usd / self.output_files_count
        
        # Calculate resource usage averages
        if self.resource_samples:
            memory_samples = [sample.memory_used_mb for sample in self.resource_samples]
            cpu_samples = [sample.cpu_percent for sample in self.resource_samples]
            
            self.peak_memory_mb = max(memory_samples) if memory_samples else 0.0
            self.average_memory_mb = mean(memory_samples) if memory_samples else 0.0
            self.peak_cpu_percent = max(cpu_samples) if cpu_samples else 0.0
            self.average_cpu_percent = mean(cpu_samples) if cpu_samples else 0.0


@dataclass
class PerformanceBaseline:
    """Performance baseline for a specific pipeline."""
    
    pipeline_name: str
    baseline_date: datetime
    sample_count: int
    
    # Execution time metrics
    avg_execution_time: float
    median_execution_time: float
    p95_execution_time: float
    execution_time_std: float
    
    # Cost metrics
    avg_cost: float
    median_cost: float
    p95_cost: float
    cost_std: float
    
    # Resource usage metrics
    avg_memory_mb: float
    avg_cpu_percent: float
    avg_tokens_per_second: float
    
    # Success rate metrics
    success_rate: float
    avg_quality_score: Optional[float] = None
    
    # Confidence metrics
    baseline_confidence: float = 1.0  # 0.0 to 1.0, based on sample size and consistency
    last_updated: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def create_from_executions(cls, pipeline_name: str, executions: List[ExecutionMetrics]) -> 'PerformanceBaseline':
        """Create baseline from a collection of execution metrics."""
        if not executions:
            raise ValueError("Cannot create baseline from empty execution list")
        
        successful_executions = [e for e in executions if e.success]
        if not successful_executions:
            raise ValueError("Cannot create baseline from executions with no successes")
        
        # Calculate execution time statistics
        exec_times = [e.execution_time_seconds for e in successful_executions]
        costs = [e.estimated_cost_usd for e in successful_executions]
        memory_usage = [e.peak_memory_mb for e in successful_executions if e.peak_memory_mb > 0]
        cpu_usage = [e.peak_cpu_percent for e in successful_executions if e.peak_cpu_percent > 0]
        token_rates = [e.throughput_tokens_per_second for e in successful_executions if e.throughput_tokens_per_second > 0]
        quality_scores = [e.quality_score for e in successful_executions if e.quality_score is not None]
        
        return cls(
            pipeline_name=pipeline_name,
            baseline_date=datetime.now(),
            sample_count=len(successful_executions),
            avg_execution_time=mean(exec_times),
            median_execution_time=median(exec_times),
            p95_execution_time=cls._calculate_percentile(exec_times, 95),
            execution_time_std=stdev(exec_times) if len(exec_times) > 1 else 0.0,
            avg_cost=mean(costs) if costs else 0.0,
            median_cost=median(costs) if costs else 0.0,
            p95_cost=cls._calculate_percentile(costs, 95) if costs else 0.0,
            cost_std=stdev(costs) if len(costs) > 1 else 0.0,
            avg_memory_mb=mean(memory_usage) if memory_usage else 0.0,
            avg_cpu_percent=mean(cpu_usage) if cpu_usage else 0.0,
            avg_tokens_per_second=mean(token_rates) if token_rates else 0.0,
            success_rate=len(successful_executions) / len(executions),
            avg_quality_score=mean(quality_scores) if quality_scores else None,
            baseline_confidence=cls._calculate_confidence(len(successful_executions), exec_times)
        )
    
    @staticmethod
    def _calculate_percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile of a dataset."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    @staticmethod
    def _calculate_confidence(sample_size: int, data: List[float]) -> float:
        """Calculate confidence level based on sample size and data consistency."""
        if sample_size < 3:
            return 0.3
        elif sample_size < 5:
            return 0.5
        elif sample_size < 10:
            return 0.7
        
        # Factor in data consistency (lower standard deviation = higher confidence)
        if len(data) > 1:
            coefficient_of_variation = stdev(data) / mean(data) if mean(data) > 0 else 1.0
            consistency_factor = max(0.1, 1.0 - coefficient_of_variation)
            return min(1.0, 0.7 + (0.3 * consistency_factor))
        
        return 0.8


class PerformanceMonitor:
    """
    Advanced performance monitoring system for pipeline testing.
    
    Features:
    - Real-time resource usage tracking during execution
    - Comprehensive performance metrics collection
    - Historical performance data storage
    - Performance baseline establishment and management
    - Regression detection with configurable thresholds
    - Performance trend analysis and reporting
    """
    
    def __init__(self, 
                 storage_path: Optional[Path] = None,
                 sampling_interval: float = 1.0,
                 enable_detailed_tracking: bool = True):
        """
        Initialize performance monitor.
        
        Args:
            storage_path: Path to store performance data (default: performance_data.db)
            sampling_interval: Interval for resource sampling in seconds
            enable_detailed_tracking: Enable detailed resource tracking
        """
        self.storage_path = storage_path or Path("performance_data.db")
        self.sampling_interval = sampling_interval
        self.enable_detailed_tracking = enable_detailed_tracking
        
        # Current monitoring state
        self._monitoring_active = False
        self._current_execution: Optional[ExecutionMetrics] = None
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Performance baselines cache
        self._baselines: Dict[str, PerformanceBaseline] = {}
        self._baselines_loaded = False
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Initialized PerformanceMonitor (storage: {self.storage_path})")
    
    def _init_database(self):
        """Initialize SQLite database for performance data storage."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.cursor()
                
                # Execution metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS execution_metrics (
                        id TEXT PRIMARY KEY,
                        pipeline_name TEXT NOT NULL,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                        execution_time_seconds REAL,
                        success INTEGER,
                        api_calls_count INTEGER,
                        total_tokens_used INTEGER,
                        estimated_cost_usd REAL,
                        peak_memory_mb REAL,
                        average_memory_mb REAL,
                        peak_cpu_percent REAL,
                        average_cpu_percent REAL,
                        output_files_count INTEGER,
                        output_total_size_bytes INTEGER,
                        quality_score REAL,
                        throughput_tokens_per_second REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Performance baselines table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_baselines (
                        pipeline_name TEXT PRIMARY KEY,
                        baseline_date TEXT NOT NULL,
                        sample_count INTEGER,
                        avg_execution_time REAL,
                        median_execution_time REAL,
                        p95_execution_time REAL,
                        execution_time_std REAL,
                        avg_cost REAL,
                        median_cost REAL,
                        avg_memory_mb REAL,
                        success_rate REAL,
                        avg_quality_score REAL,
                        baseline_confidence REAL,
                        last_updated TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Resource samples table (for detailed tracking)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS resource_samples (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        execution_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        cpu_percent REAL,
                        memory_used_mb REAL,
                        memory_percent REAL,
                        disk_read_bytes INTEGER,
                        disk_write_bytes INTEGER,
                        FOREIGN KEY (execution_id) REFERENCES execution_metrics (id)
                    )
                """)
                
                conn.commit()
                logger.info("Performance database initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize performance database: {e}")
            raise
    
    def start_execution_monitoring(self, pipeline_name: str, execution_id: Optional[str] = None) -> str:
        """
        Start monitoring performance for a pipeline execution.
        
        Args:
            pipeline_name: Name of the pipeline being executed
            execution_id: Unique execution ID (generated if not provided)
            
        Returns:
            str: Execution ID for this monitoring session
        """
        if self._monitoring_active:
            logger.warning("Performance monitoring already active, stopping previous session")
            self.stop_execution_monitoring()
        
        execution_id = execution_id or f"{pipeline_name}_{int(time.time() * 1000)}"
        
        self._current_execution = ExecutionMetrics(
            pipeline_name=pipeline_name,
            execution_id=execution_id,
            start_time=datetime.now()
        )
        
        # Start resource monitoring thread if detailed tracking is enabled
        if self.enable_detailed_tracking:
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(
                target=self._resource_monitoring_loop,
                daemon=True
            )
            self._monitoring_thread.start()
        
        self._monitoring_active = True
        
        logger.info(f"Started performance monitoring for {pipeline_name} (ID: {execution_id})")
        return execution_id
    
    def stop_execution_monitoring(self, success: bool = True, 
                                output_metrics: Optional[Dict[str, Any]] = None) -> ExecutionMetrics:
        """
        Stop monitoring and finalize performance metrics.
        
        Args:
            success: Whether the execution was successful
            output_metrics: Additional metrics from pipeline execution
            
        Returns:
            ExecutionMetrics: Complete performance metrics for this execution
        """
        if not self._monitoring_active or not self._current_execution:
            raise ValueError("No active monitoring session to stop")
        
        # Stop resource monitoring
        if self._monitoring_thread:
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5.0)
        
        # Finalize execution metrics
        self._current_execution.end_time = datetime.now()
        self._current_execution.success = success
        
        # Add output metrics if provided
        if output_metrics:
            self._update_execution_with_output_metrics(output_metrics)
        
        # Calculate derived metrics
        self._current_execution.calculate_derived_metrics()
        
        # Store metrics in database
        self._store_execution_metrics(self._current_execution)
        
        # Reset monitoring state
        execution_result = self._current_execution
        self._current_execution = None
        self._monitoring_active = False
        self._monitoring_thread = None
        
        logger.info(f"Stopped performance monitoring for {execution_result.pipeline_name} "
                   f"(duration: {execution_result.execution_time_seconds:.1f}s)")
        
        return execution_result
    
    def _resource_monitoring_loop(self):
        """Resource monitoring loop running in separate thread."""
        if not _PSUTIL_AVAILABLE:
            logger.warning("psutil not available, resource monitoring disabled")
            return
            
        try:
            process = psutil.Process()
            last_disk_io = process.io_counters() if hasattr(process, 'io_counters') else None
        except Exception as e:
            logger.warning(f"Failed to initialize process monitoring: {e}")
            return
        
        while not self._stop_monitoring.is_set():
            try:
                # Get system resource metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()
                
                # Get process-specific metrics
                process_memory = process.memory_info()
                process_cpu = process.cpu_percent()
                
                # Get disk I/O metrics (delta since last measurement)
                disk_read_bytes = 0
                disk_write_bytes = 0
                if hasattr(process, 'io_counters'):
                    current_io = process.io_counters()
                    if last_disk_io:
                        disk_read_bytes = current_io.read_bytes - last_disk_io.read_bytes
                        disk_write_bytes = current_io.write_bytes - last_disk_io.write_bytes
                    last_disk_io = current_io
                
                # Create resource sample
                sample = ResourceMetrics(
                    cpu_percent=cpu_percent,
                    cpu_count=psutil.cpu_count(),
                    memory_used_mb=memory_info.used / (1024 * 1024),
                    memory_percent=memory_info.percent,
                    disk_read_bytes=disk_read_bytes,
                    disk_write_bytes=disk_write_bytes,
                    process_memory_rss=process_memory.rss,
                    process_memory_vms=process_memory.vms,
                    process_cpu_times=process_cpu,
                    timestamp=datetime.now()
                )
                
                # Add sample to current execution
                if self._current_execution:
                    self._current_execution.resource_samples.append(sample)
                
                # Store sample in database if execution is active
                if self._current_execution:
                    self._store_resource_sample(self._current_execution.execution_id, sample)
                
            except Exception as e:
                logger.warning(f"Error during resource monitoring: {e}")
            
            # Wait for next sampling interval
            self._stop_monitoring.wait(self.sampling_interval)
    
    def _update_execution_with_output_metrics(self, output_metrics: Dict[str, Any]):
        """Update execution metrics with output information."""
        if not self._current_execution:
            return
        
        # Extract API and cost metrics
        if 'api_calls' in output_metrics:
            self._current_execution.api_calls_count = output_metrics['api_calls']
        
        if 'tokens_used' in output_metrics:
            self._current_execution.total_tokens_used = output_metrics['tokens_used']
        
        if 'input_tokens' in output_metrics:
            self._current_execution.input_tokens = output_metrics['input_tokens']
        
        if 'output_tokens' in output_metrics:
            self._current_execution.output_tokens = output_metrics['output_tokens']
        
        if 'estimated_cost' in output_metrics:
            self._current_execution.estimated_cost_usd = output_metrics['estimated_cost']
        
        if 'model_calls' in output_metrics and isinstance(output_metrics['model_calls'], dict):
            self._current_execution.model_calls_breakdown = output_metrics['model_calls']
        
        # Extract output file metrics
        if 'output_files' in output_metrics:
            output_files = output_metrics['output_files']
            if isinstance(output_files, list):
                self._current_execution.output_files_count = len(output_files)
                
                # Calculate total size and file type breakdown
                total_size = 0
                file_types = {}
                for file_path in output_files:
                    try:
                        if isinstance(file_path, (str, Path)):
                            path = Path(file_path)
                            if path.exists():
                                size = path.stat().st_size
                                total_size += size
                                
                                ext = path.suffix.lower() or 'no_extension'
                                file_types[ext] = file_types.get(ext, 0) + 1
                    except Exception:
                        continue
                
                self._current_execution.output_total_size_bytes = total_size
                self._current_execution.output_file_types = file_types
        
        # Extract quality metrics
        if 'quality_score' in output_metrics:
            self._current_execution.quality_score = output_metrics['quality_score']
        
        if 'template_issues' in output_metrics:
            self._current_execution.template_resolution_issues = output_metrics['template_issues']
    
    def _store_execution_metrics(self, execution: ExecutionMetrics):
        """Store execution metrics in database."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO execution_metrics
                    (id, pipeline_name, start_time, end_time, execution_time_seconds,
                     success, api_calls_count, total_tokens_used, estimated_cost_usd,
                     peak_memory_mb, average_memory_mb, peak_cpu_percent, average_cpu_percent,
                     output_files_count, output_total_size_bytes, quality_score,
                     throughput_tokens_per_second)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    execution.execution_id,
                    execution.pipeline_name,
                    execution.start_time.isoformat(),
                    execution.end_time.isoformat() if execution.end_time else None,
                    execution.execution_time_seconds,
                    int(execution.success),
                    execution.api_calls_count,
                    execution.total_tokens_used,
                    execution.estimated_cost_usd,
                    execution.peak_memory_mb,
                    execution.average_memory_mb,
                    execution.peak_cpu_percent,
                    execution.average_cpu_percent,
                    execution.output_files_count,
                    execution.output_total_size_bytes,
                    execution.quality_score,
                    execution.throughput_tokens_per_second
                ))
                
                conn.commit()
        
        except Exception as e:
            logger.error(f"Failed to store execution metrics: {e}")
    
    def _store_resource_sample(self, execution_id: str, sample: ResourceMetrics):
        """Store resource sample in database."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO resource_samples
                    (execution_id, timestamp, cpu_percent, memory_used_mb, memory_percent,
                     disk_read_bytes, disk_write_bytes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    execution_id,
                    sample.timestamp.isoformat(),
                    sample.cpu_percent,
                    sample.memory_used_mb,
                    sample.memory_percent,
                    sample.disk_read_bytes,
                    sample.disk_write_bytes
                ))
                
                conn.commit()
        
        except Exception as e:
            logger.error(f"Failed to store resource sample: {e}")
    
    def get_execution_history(self, 
                            pipeline_name: Optional[str] = None,
                            days_back: int = 30,
                            include_failed: bool = True) -> List[ExecutionMetrics]:
        """
        Get execution history from database.
        
        Args:
            pipeline_name: Filter by specific pipeline (None for all)
            days_back: Number of days to look back
            include_failed: Include failed executions
            
        Returns:
            List[ExecutionMetrics]: Historical execution data
        """
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.cursor()
                
                # Build query
                query = """
                    SELECT * FROM execution_metrics 
                    WHERE start_time >= datetime('now', '-{} days')
                """.format(days_back)
                
                params = []
                
                if pipeline_name:
                    query += " AND pipeline_name = ?"
                    params.append(pipeline_name)
                
                if not include_failed:
                    query += " AND success = 1"
                
                query += " ORDER BY start_time DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert rows to ExecutionMetrics objects
                executions = []
                for row in rows:
                    execution = ExecutionMetrics(
                        pipeline_name=row[1],
                        execution_id=row[0],
                        start_time=datetime.fromisoformat(row[2]),
                        end_time=datetime.fromisoformat(row[3]) if row[3] else None,
                        execution_time_seconds=row[4] or 0.0,
                        success=bool(row[5]),
                        api_calls_count=row[6] or 0,
                        total_tokens_used=row[7] or 0,
                        estimated_cost_usd=row[8] or 0.0,
                        peak_memory_mb=row[9] or 0.0,
                        average_memory_mb=row[10] or 0.0,
                        peak_cpu_percent=row[11] or 0.0,
                        average_cpu_percent=row[12] or 0.0,
                        output_files_count=row[13] or 0,
                        output_total_size_bytes=row[14] or 0,
                        quality_score=row[15],
                        throughput_tokens_per_second=row[16] or 0.0
                    )
                    executions.append(execution)
                
                return executions
        
        except Exception as e:
            logger.error(f"Failed to get execution history: {e}")
            return []
    
    def establish_baseline(self, 
                         pipeline_name: str, 
                         min_samples: int = 5,
                         days_back: int = 30) -> Optional[PerformanceBaseline]:
        """
        Establish performance baseline for a pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            min_samples: Minimum number of successful executions needed
            days_back: Days to look back for baseline data
            
        Returns:
            PerformanceBaseline: Established baseline or None if insufficient data
        """
        # Get historical executions
        executions = self.get_execution_history(
            pipeline_name=pipeline_name,
            days_back=days_back,
            include_failed=False  # Only successful executions for baseline
        )
        
        if len(executions) < min_samples:
            logger.warning(f"Insufficient data for baseline: {len(executions)} < {min_samples} samples")
            return None
        
        try:
            baseline = PerformanceBaseline.create_from_executions(pipeline_name, executions)
            
            # Store baseline in database
            self._store_baseline(baseline)
            
            # Update cache
            self._baselines[pipeline_name] = baseline
            
            logger.info(f"Established baseline for {pipeline_name} "
                       f"(samples: {baseline.sample_count}, confidence: {baseline.baseline_confidence:.2f})")
            
            return baseline
        
        except Exception as e:
            logger.error(f"Failed to establish baseline for {pipeline_name}: {e}")
            return None
    
    def _store_baseline(self, baseline: PerformanceBaseline):
        """Store baseline in database."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO performance_baselines
                    (pipeline_name, baseline_date, sample_count, avg_execution_time,
                     median_execution_time, p95_execution_time, execution_time_std,
                     avg_cost, median_cost, avg_memory_mb, success_rate,
                     avg_quality_score, baseline_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    baseline.pipeline_name,
                    baseline.baseline_date.isoformat(),
                    baseline.sample_count,
                    baseline.avg_execution_time,
                    baseline.median_execution_time,
                    baseline.p95_execution_time,
                    baseline.execution_time_std,
                    baseline.avg_cost,
                    baseline.median_cost,
                    baseline.avg_memory_mb,
                    baseline.success_rate,
                    baseline.avg_quality_score,
                    baseline.baseline_confidence
                ))
                
                conn.commit()
        
        except Exception as e:
            logger.error(f"Failed to store baseline: {e}")
    
    def get_baseline(self, pipeline_name: str) -> Optional[PerformanceBaseline]:
        """
        Get performance baseline for a pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            
        Returns:
            PerformanceBaseline: Baseline data or None if not available
        """
        # Check cache first
        if pipeline_name in self._baselines:
            return self._baselines[pipeline_name]
        
        # Load from database
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM performance_baselines WHERE pipeline_name = ?
                """, (pipeline_name,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                baseline = PerformanceBaseline(
                    pipeline_name=row[0],
                    baseline_date=datetime.fromisoformat(row[1]),
                    sample_count=row[2],
                    avg_execution_time=row[3],
                    median_execution_time=row[4],
                    p95_execution_time=row[5],
                    execution_time_std=row[6],
                    avg_cost=row[7],
                    median_cost=row[8],
                    avg_memory_mb=row[9],
                    success_rate=row[10],
                    avg_quality_score=row[11],
                    baseline_confidence=row[12]
                )
                
                # Cache for future use
                self._baselines[pipeline_name] = baseline
                
                return baseline
        
        except Exception as e:
            logger.error(f"Failed to get baseline for {pipeline_name}: {e}")
            return None
    
    def update_baseline(self, 
                       pipeline_name: str,
                       force_update: bool = False) -> Optional[PerformanceBaseline]:
        """
        Update baseline with recent execution data.
        
        Args:
            pipeline_name: Name of the pipeline
            force_update: Force update even if recent baseline exists
            
        Returns:
            PerformanceBaseline: Updated baseline or None
        """
        existing_baseline = self.get_baseline(pipeline_name)
        
        # Check if update is needed
        if existing_baseline and not force_update:
            days_since_update = (datetime.now() - existing_baseline.last_updated).days
            if days_since_update < 7:  # Don't update more than once per week
                logger.info(f"Baseline for {pipeline_name} is recent, skipping update")
                return existing_baseline
        
        # Establish new baseline
        return self.establish_baseline(pipeline_name)
    
    def get_performance_summary(self, 
                               pipeline_name: Optional[str] = None,
                               days_back: int = 7) -> Dict[str, Any]:
        """
        Get performance summary for pipeline(s).
        
        Args:
            pipeline_name: Specific pipeline or None for all
            days_back: Days to include in summary
            
        Returns:
            Dict: Performance summary data
        """
        executions = self.get_execution_history(
            pipeline_name=pipeline_name,
            days_back=days_back
        )
        
        if not executions:
            return {"error": "No execution data available"}
        
        # Group by pipeline if analyzing multiple pipelines
        if pipeline_name is None:
            pipeline_groups = {}
            for execution in executions:
                name = execution.pipeline_name
                if name not in pipeline_groups:
                    pipeline_groups[name] = []
                pipeline_groups[name].append(execution)
        else:
            pipeline_groups = {pipeline_name: executions}
        
        summary = {}
        
        for name, pipeline_executions in pipeline_groups.items():
            successful = [e for e in pipeline_executions if e.success]
            
            pipeline_summary = {
                "total_executions": len(pipeline_executions),
                "successful_executions": len(successful),
                "success_rate": len(successful) / len(pipeline_executions) if pipeline_executions else 0,
                "total_cost": sum(e.estimated_cost_usd for e in pipeline_executions),
                "average_execution_time": mean([e.execution_time_seconds for e in successful]) if successful else 0,
                "total_tokens_used": sum(e.total_tokens_used for e in pipeline_executions),
                "average_quality_score": mean([e.quality_score for e in successful if e.quality_score is not None]) if successful else None
            }
            
            # Add baseline comparison if available
            baseline = self.get_baseline(name)
            if baseline:
                recent_avg_time = pipeline_summary["average_execution_time"]
                if recent_avg_time > 0:
                    time_change_percent = ((recent_avg_time - baseline.avg_execution_time) / baseline.avg_execution_time) * 100
                    pipeline_summary["baseline_comparison"] = {
                        "time_change_percent": time_change_percent,
                        "baseline_confidence": baseline.baseline_confidence,
                        "baseline_age_days": (datetime.now() - baseline.baseline_date).days
                    }
            
            summary[name] = pipeline_summary
        
        return summary