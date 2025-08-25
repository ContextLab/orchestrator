#!/usr/bin/env python3
"""
Performance Monitoring System for Issue #262.

Advanced performance monitoring and analysis system that tracks pipeline 
execution metrics, detects regressions, establishes baselines, and provides
comprehensive performance analytics for the dashboard system.
"""

import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
import psutil
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import threading
import queue

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance measurement."""
    pipeline_name: str
    metric_type: str  # execution_time, memory_usage, cpu_usage, io_operations
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelinePerformanceProfile:
    """Comprehensive performance profile for a pipeline."""
    pipeline_name: str
    baseline_execution_time: float
    baseline_memory_usage: float
    baseline_cpu_usage: float
    current_metrics: Dict[str, List[PerformanceMetric]]
    performance_trends: Dict[str, str]  # improving, stable, declining
    regression_alerts: List[str]
    optimization_opportunities: List[str]
    resource_efficiency_score: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison."""
    pipeline_name: str
    metric_type: str
    baseline_value: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    last_updated: datetime
    stability_score: float  # 0-1, higher means more stable


class PerformanceMonitor:
    """Advanced performance monitoring system."""
    
    def __init__(self, root_path: str = ".", monitoring_interval: float = 1.0):
        self.root_path = Path(root_path).resolve()
        self.monitoring_interval = monitoring_interval
        
        # Performance data storage
        self.performance_dir = self.root_path / "temp" / "performance"
        self.performance_dir.mkdir(parents=True, exist_ok=True)
        
        self.database_path = self.performance_dir / "performance_metrics.db"
        self._setup_database()
        
        # Real-time monitoring
        self.active_monitors = {}  # pipeline_name -> monitoring thread
        self.metric_queue = queue.Queue()
        self.monitoring_active = False
        
        # Performance baselines
        self.baselines = {}  # pipeline_name -> {metric_type -> PerformanceBaseline}
        self._load_baselines()
        
        # Configuration
        self.regression_thresholds = {
            'execution_time': 1.5,      # 50% increase triggers alert
            'memory_usage': 2.0,        # 100% increase triggers alert
            'cpu_usage': 1.8,           # 80% increase triggers alert
            'io_operations': 2.5        # 150% increase triggers alert
        }
        
        self.baseline_confidence = 0.95  # 95% confidence for baselines
        self.min_samples_for_baseline = 10
        
        logger.info(f"Performance Monitor initialized for: {self.root_path}")

    def _setup_database(self):
        """Setup SQLite database for performance metrics storage."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline_name TEXT,
                metric_type TEXT,
                value REAL,
                timestamp DATETIME,
                metadata TEXT
            )
        ''')
        
        # Performance baselines table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_baselines (
                pipeline_name TEXT,
                metric_type TEXT,
                baseline_value REAL,
                confidence_lower REAL,
                confidence_upper REAL,
                sample_size INTEGER,
                last_updated DATETIME,
                stability_score REAL,
                PRIMARY KEY (pipeline_name, metric_type)
            )
        ''')
        
        # Performance profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline_name TEXT,
                profile_data TEXT,
                timestamp DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()

    def _load_baselines(self):
        """Load performance baselines from database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT pipeline_name, metric_type, baseline_value, confidence_lower, 
                       confidence_upper, sample_size, last_updated, stability_score
                FROM performance_baselines
            ''')
            
            for row in cursor.fetchall():
                pipeline_name = row[0]
                metric_type = row[1]
                
                baseline = PerformanceBaseline(
                    pipeline_name=pipeline_name,
                    metric_type=metric_type,
                    baseline_value=row[2],
                    confidence_interval=(row[3], row[4]),
                    sample_size=row[5],
                    last_updated=datetime.fromisoformat(row[6]),
                    stability_score=row[7]
                )
                
                if pipeline_name not in self.baselines:
                    self.baselines[pipeline_name] = {}
                self.baselines[pipeline_name][metric_type] = baseline
            
            conn.close()
            logger.info(f"Loaded {len(self.baselines)} performance baselines")
            
        except Exception as e:
            logger.error(f"Failed to load baselines: {e}")

    def start_monitoring(self, pipeline_name: str, process_id: Optional[int] = None) -> str:
        """Start performance monitoring for a pipeline."""
        if pipeline_name in self.active_monitors:
            logger.warning(f"Monitoring already active for pipeline: {pipeline_name}")
            return f"monitoring_already_active_{pipeline_name}"
        
        monitor_id = f"monitor_{pipeline_name}_{int(time.time())}"
        
        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=self._monitor_pipeline_performance,
            args=(pipeline_name, process_id, monitor_id),
            daemon=True
        )
        monitor_thread.start()
        
        self.active_monitors[pipeline_name] = {
            'thread': monitor_thread,
            'monitor_id': monitor_id,
            'start_time': datetime.now(),
            'process_id': process_id
        }
        
        logger.info(f"Started performance monitoring for pipeline: {pipeline_name}")
        return monitor_id

    def stop_monitoring(self, pipeline_name: str) -> Optional[PipelinePerformanceProfile]:
        """Stop monitoring and generate performance profile."""
        if pipeline_name not in self.active_monitors:
            logger.warning(f"No active monitoring for pipeline: {pipeline_name}")
            return None
        
        monitor_info = self.active_monitors.pop(pipeline_name)
        
        # Allow thread to finish current cycle
        time.sleep(self.monitoring_interval * 2)
        
        # Generate performance profile
        profile = self._generate_performance_profile(pipeline_name, monitor_info)
        
        # Update baselines
        self._update_baselines(pipeline_name)
        
        logger.info(f"Stopped performance monitoring for pipeline: {pipeline_name}")
        return profile

    def _monitor_pipeline_performance(self, pipeline_name: str, process_id: Optional[int], monitor_id: str):
        """Monitor performance metrics for a pipeline in real-time."""
        logger.debug(f"Starting performance monitoring thread for {pipeline_name}")
        
        start_time = time.time()
        last_cpu_times = None
        
        while pipeline_name in self.active_monitors:
            try:
                current_time = time.time()
                timestamp = datetime.now()
                
                # System-wide metrics
                system_metrics = self._collect_system_metrics()
                
                # Process-specific metrics if available
                process_metrics = {}
                if process_id:
                    try:
                        process = psutil.Process(process_id)
                        if process.is_running():
                            process_metrics = self._collect_process_metrics(process)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        logger.debug(f"Process {process_id} no longer accessible")
                        process_id = None
                
                # Store metrics
                all_metrics = {**system_metrics, **process_metrics}
                for metric_type, value in all_metrics.items():
                    metric = PerformanceMetric(
                        pipeline_name=pipeline_name,
                        metric_type=metric_type,
                        value=value,
                        timestamp=timestamp,
                        metadata={'monitor_id': monitor_id, 'elapsed_time': current_time - start_time}
                    )
                    self._store_metric(metric)
                
                # Calculate execution time if monitoring is complete
                if pipeline_name not in self.active_monitors:  # Monitoring stopped
                    execution_time = current_time - start_time
                    execution_metric = PerformanceMetric(
                        pipeline_name=pipeline_name,
                        metric_type='execution_time',
                        value=execution_time,
                        timestamp=timestamp,
                        metadata={'monitor_id': monitor_id, 'total_execution': True}
                    )
                    self._store_metric(execution_metric)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring for {pipeline_name}: {e}")
                break
        
        logger.debug(f"Performance monitoring thread ended for {pipeline_name}")

    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-wide performance metrics."""
        metrics = {}
        
        try:
            # CPU usage
            metrics['system_cpu_percent'] = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics['system_memory_percent'] = memory.percent
            metrics['system_memory_available_gb'] = memory.available / (1024**3)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                metrics['system_disk_read_mb'] = disk_io.read_bytes / (1024**2)
                metrics['system_disk_write_mb'] = disk_io.write_bytes / (1024**2)
            
            # Network I/O
            network_io = psutil.net_io_counters()
            if network_io:
                metrics['system_network_sent_mb'] = network_io.bytes_sent / (1024**2)
                metrics['system_network_recv_mb'] = network_io.bytes_recv / (1024**2)
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
        
        return metrics

    def _collect_process_metrics(self, process: psutil.Process) -> Dict[str, float]:
        """Collect process-specific performance metrics."""
        metrics = {}
        
        try:
            # CPU usage
            metrics['process_cpu_percent'] = process.cpu_percent()
            
            # Memory usage
            memory_info = process.memory_info()
            metrics['process_memory_rss_mb'] = memory_info.rss / (1024**2)
            metrics['process_memory_vms_mb'] = memory_info.vms / (1024**2)
            
            # I/O operations
            io_counters = process.io_counters()
            metrics['process_io_read_mb'] = io_counters.read_bytes / (1024**2)
            metrics['process_io_write_mb'] = io_counters.write_bytes / (1024**2)
            
            # Number of threads
            metrics['process_num_threads'] = process.num_threads()
            
            # Number of file descriptors (Unix only)
            try:
                metrics['process_num_fds'] = process.num_fds()
            except AttributeError:
                pass  # Not available on Windows
            
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"Failed to collect process metrics: {e}")
        
        return metrics

    def _store_metric(self, metric: PerformanceMetric):
        """Store performance metric in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_metrics (pipeline_name, metric_type, value, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                metric.pipeline_name,
                metric.metric_type,
                metric.value,
                metric.timestamp.isoformat(),
                json.dumps(metric.metadata)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store metric: {e}")

    def _generate_performance_profile(self, pipeline_name: str, monitor_info: Dict[str, Any]) -> PipelinePerformanceProfile:
        """Generate comprehensive performance profile for a pipeline."""
        logger.info(f"Generating performance profile for {pipeline_name}")
        
        # Load metrics from monitoring session
        metrics = self._load_metrics_for_session(pipeline_name, monitor_info['monitor_id'])
        
        # Group metrics by type
        grouped_metrics = defaultdict(list)
        for metric in metrics:
            grouped_metrics[metric.metric_type].append(metric)
        
        # Calculate baseline metrics
        baseline_execution_time = self._calculate_baseline_metric(pipeline_name, 'execution_time')
        baseline_memory_usage = self._calculate_baseline_metric(pipeline_name, 'process_memory_rss_mb')
        baseline_cpu_usage = self._calculate_baseline_metric(pipeline_name, 'process_cpu_percent')
        
        # Analyze performance trends
        performance_trends = {}
        for metric_type, metric_list in grouped_metrics.items():
            if len(metric_list) > 5:  # Need sufficient data for trend analysis
                performance_trends[metric_type] = self._analyze_performance_trend(metric_list)
        
        # Detect regression alerts
        regression_alerts = self._detect_performance_regressions(pipeline_name, grouped_metrics)
        
        # Identify optimization opportunities
        optimization_opportunities = self._identify_optimization_opportunities(pipeline_name, grouped_metrics)
        
        # Calculate resource efficiency score
        efficiency_score = self._calculate_efficiency_score(grouped_metrics)
        
        profile = PipelinePerformanceProfile(
            pipeline_name=pipeline_name,
            baseline_execution_time=baseline_execution_time,
            baseline_memory_usage=baseline_memory_usage,
            baseline_cpu_usage=baseline_cpu_usage,
            current_metrics=dict(grouped_metrics),
            performance_trends=performance_trends,
            regression_alerts=regression_alerts,
            optimization_opportunities=optimization_opportunities,
            resource_efficiency_score=efficiency_score
        )
        
        # Store profile in database
        self._store_performance_profile(profile)
        
        return profile

    def _load_metrics_for_session(self, pipeline_name: str, monitor_id: str) -> List[PerformanceMetric]:
        """Load performance metrics for a specific monitoring session."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT metric_type, value, timestamp, metadata
                FROM performance_metrics 
                WHERE pipeline_name = ? AND metadata LIKE ?
                ORDER BY timestamp ASC
            ''', (pipeline_name, f'%{monitor_id}%'))
            
            metrics = []
            for row in cursor.fetchall():
                try:
                    metadata = json.loads(row[3]) if row[3] else {}
                except:
                    metadata = {}
                
                metrics.append(PerformanceMetric(
                    pipeline_name=pipeline_name,
                    metric_type=row[0],
                    value=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    metadata=metadata
                ))
            
            conn.close()
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to load session metrics: {e}")
            return []

    def _calculate_baseline_metric(self, pipeline_name: str, metric_type: str) -> float:
        """Calculate baseline value for a specific metric."""
        if pipeline_name in self.baselines and metric_type in self.baselines[pipeline_name]:
            return self.baselines[pipeline_name][metric_type].baseline_value
        
        # If no baseline exists, try to calculate from recent data
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Get recent values (last 30 days)
            cutoff_date = datetime.now() - timedelta(days=30)
            cursor.execute('''
                SELECT value FROM performance_metrics 
                WHERE pipeline_name = ? AND metric_type = ? AND timestamp >= ?
            ''', (pipeline_name, metric_type, cutoff_date.isoformat()))
            
            values = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if len(values) >= 3:
                return np.percentile(values, 75)  # Use 75th percentile as baseline
            elif values:
                return np.mean(values)
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Failed to calculate baseline for {metric_type}: {e}")
            return 0.0

    def _analyze_performance_trend(self, metrics: List[PerformanceMetric]) -> str:
        """Analyze performance trend for a series of metrics."""
        if len(metrics) < 5:
            return "stable"
        
        # Sort by timestamp
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
        values = [m.value for m in sorted_metrics]
        
        # Calculate linear regression slope
        n = len(values)
        x = list(range(n))
        x_mean = np.mean(x)
        y_mean = np.mean(values)
        
        slope_numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        slope_denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if slope_denominator == 0:
            return "stable"
        
        slope = slope_numerator / slope_denominator
        
        # Determine trend based on slope and metric type
        threshold = abs(y_mean) * 0.1 / n if y_mean != 0 else 0.1  # 10% change threshold
        
        if abs(slope) < threshold:
            return "stable"
        elif slope > 0:
            # For execution time and resource usage, positive slope is declining performance
            if any(keyword in metrics[0].metric_type.lower() for keyword in ['time', 'memory', 'cpu']):
                return "declining"
            else:
                return "improving"
        else:
            if any(keyword in metrics[0].metric_type.lower() for keyword in ['time', 'memory', 'cpu']):
                return "improving"
            else:
                return "declining"

    def _detect_performance_regressions(self, pipeline_name: str, grouped_metrics: Dict[str, List[PerformanceMetric]]) -> List[str]:
        """Detect performance regressions compared to baselines."""
        regressions = []
        
        for metric_type, metrics in grouped_metrics.items():
            if not metrics:
                continue
            
            # Get baseline for comparison
            baseline_value = self._calculate_baseline_metric(pipeline_name, metric_type)
            if baseline_value == 0:
                continue
            
            # Calculate current average
            current_values = [m.value for m in metrics if m.value > 0]
            if not current_values:
                continue
            
            current_avg = np.mean(current_values)
            threshold = self.regression_thresholds.get(metric_type, 1.5)
            
            # Check for regression (current significantly worse than baseline)
            if current_avg > baseline_value * threshold:
                regression_pct = ((current_avg - baseline_value) / baseline_value) * 100
                regressions.append(
                    f"{metric_type}: {regression_pct:.1f}% increase from baseline "
                    f"({current_avg:.2f} vs {baseline_value:.2f})"
                )
        
        return regressions

    def _identify_optimization_opportunities(self, pipeline_name: str, grouped_metrics: Dict[str, List[PerformanceMetric]]) -> List[str]:
        """Identify opportunities for performance optimization."""
        opportunities = []
        
        # Analyze memory usage patterns
        memory_metrics = grouped_metrics.get('process_memory_rss_mb', [])
        if memory_metrics:
            memory_values = [m.value for m in memory_metrics if m.value > 0]
            if memory_values:
                max_memory = max(memory_values)
                avg_memory = np.mean(memory_values)
                
                if max_memory > 1000:  # > 1GB
                    opportunities.append(f"High memory usage detected (max: {max_memory:.1f}MB)")
                
                if max_memory / avg_memory > 3:  # High memory variance
                    opportunities.append("Memory usage is highly variable - consider memory management optimization")
        
        # Analyze CPU usage patterns
        cpu_metrics = grouped_metrics.get('process_cpu_percent', [])
        if cpu_metrics:
            cpu_values = [m.value for m in cpu_metrics if m.value > 0]
            if cpu_values:
                max_cpu = max(cpu_values)
                avg_cpu = np.mean(cpu_values)
                
                if avg_cpu > 80:
                    opportunities.append(f"High CPU usage detected (avg: {avg_cpu:.1f}%)")
                
                if max_cpu > 95:
                    opportunities.append("CPU utilization near maximum - consider parallel processing or optimization")
        
        # Analyze I/O patterns
        io_read_metrics = grouped_metrics.get('process_io_read_mb', [])
        io_write_metrics = grouped_metrics.get('process_io_write_mb', [])
        
        if io_read_metrics and io_write_metrics:
            total_io = sum(m.value for m in io_read_metrics + io_write_metrics if m.value > 0)
            if total_io > 100:  # > 100MB I/O
                opportunities.append(f"High I/O activity detected ({total_io:.1f}MB) - consider caching or batching")
        
        # Analyze execution time if available
        exec_metrics = grouped_metrics.get('execution_time', [])
        if exec_metrics:
            exec_times = [m.value for m in exec_metrics if m.value > 0]
            if exec_times:
                avg_exec_time = np.mean(exec_times)
                if avg_exec_time > 300:  # > 5 minutes
                    opportunities.append(f"Long execution time ({avg_exec_time:.1f}s) - consider performance optimizations")
        
        return opportunities

    def _calculate_efficiency_score(self, grouped_metrics: Dict[str, List[PerformanceMetric]]) -> float:
        """Calculate overall resource efficiency score (0-100)."""
        score_components = []
        
        # Memory efficiency (lower usage = higher score)
        memory_metrics = grouped_metrics.get('process_memory_rss_mb', [])
        if memory_metrics:
            avg_memory = np.mean([m.value for m in memory_metrics if m.value > 0])
            # Score based on memory usage (lower is better)
            memory_score = max(0, 100 - (avg_memory / 10))  # 1000MB = 0 score
            score_components.append(memory_score)
        
        # CPU efficiency (moderate usage = higher score)
        cpu_metrics = grouped_metrics.get('process_cpu_percent', [])
        if cpu_metrics:
            avg_cpu = np.mean([m.value for m in cpu_metrics if m.value > 0])
            # Optimal CPU usage around 50-70%
            if avg_cpu <= 70:
                cpu_score = 100 - (abs(60 - avg_cpu) * 2)
            else:
                cpu_score = max(0, 100 - (avg_cpu - 70) * 3)
            score_components.append(cpu_score)
        
        # I/O efficiency (lower is generally better)
        io_metrics = (grouped_metrics.get('process_io_read_mb', []) + 
                     grouped_metrics.get('process_io_write_mb', []))
        if io_metrics:
            total_io = sum(m.value for m in io_metrics if m.value > 0)
            io_score = max(0, 100 - (total_io / 5))  # 500MB = 0 score
            score_components.append(io_score)
        
        # Stability score (lower variance = higher score)
        all_metrics = []
        for metrics_list in grouped_metrics.values():
            all_metrics.extend([m.value for m in metrics_list if m.value > 0])
        
        if all_metrics:
            cv = np.std(all_metrics) / np.mean(all_metrics) if np.mean(all_metrics) > 0 else 0
            stability_score = max(0, 100 - (cv * 100))
            score_components.append(stability_score)
        
        return np.mean(score_components) if score_components else 50.0

    def _update_baselines(self, pipeline_name: str):
        """Update performance baselines for a pipeline."""
        logger.debug(f"Updating performance baselines for {pipeline_name}")
        
        # Get recent metrics (last 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Get all metric types for this pipeline
            cursor.execute('''
                SELECT DISTINCT metric_type FROM performance_metrics 
                WHERE pipeline_name = ? AND timestamp >= ?
            ''', (pipeline_name, cutoff_date.isoformat()))
            
            metric_types = [row[0] for row in cursor.fetchall()]
            
            for metric_type in metric_types:
                # Get values for this metric type
                cursor.execute('''
                    SELECT value FROM performance_metrics 
                    WHERE pipeline_name = ? AND metric_type = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                ''', (pipeline_name, metric_type, cutoff_date.isoformat()))
                
                values = [row[0] for row in cursor.fetchall() if row[0] > 0]
                
                if len(values) >= self.min_samples_for_baseline:
                    # Calculate baseline statistics
                    baseline_value = np.percentile(values, 75)  # Use 75th percentile
                    
                    # Calculate confidence interval
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    confidence_margin = 1.96 * (std_val / np.sqrt(len(values)))  # 95% CI
                    confidence_interval = (mean_val - confidence_margin, mean_val + confidence_margin)
                    
                    # Calculate stability score (inverse of coefficient of variation)
                    cv = std_val / mean_val if mean_val > 0 else 1
                    stability_score = max(0, 1 - cv)
                    
                    # Update baseline in database
                    cursor.execute('''
                        INSERT OR REPLACE INTO performance_baselines 
                        (pipeline_name, metric_type, baseline_value, confidence_lower, confidence_upper,
                         sample_size, last_updated, stability_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        pipeline_name, metric_type, baseline_value,
                        confidence_interval[0], confidence_interval[1],
                        len(values), datetime.now().isoformat(), stability_score
                    ))
                    
                    # Update in-memory baseline
                    if pipeline_name not in self.baselines:
                        self.baselines[pipeline_name] = {}
                    
                    self.baselines[pipeline_name][metric_type] = PerformanceBaseline(
                        pipeline_name=pipeline_name,
                        metric_type=metric_type,
                        baseline_value=baseline_value,
                        confidence_interval=confidence_interval,
                        sample_size=len(values),
                        last_updated=datetime.now(),
                        stability_score=stability_score
                    )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update baselines: {e}")

    def _store_performance_profile(self, profile: PipelinePerformanceProfile):
        """Store performance profile in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            profile_data = {
                'pipeline_name': profile.pipeline_name,
                'baseline_execution_time': profile.baseline_execution_time,
                'baseline_memory_usage': profile.baseline_memory_usage,
                'baseline_cpu_usage': profile.baseline_cpu_usage,
                'performance_trends': profile.performance_trends,
                'regression_alerts': profile.regression_alerts,
                'optimization_opportunities': profile.optimization_opportunities,
                'resource_efficiency_score': profile.resource_efficiency_score
            }
            
            cursor.execute('''
                INSERT INTO performance_profiles (pipeline_name, profile_data, timestamp)
                VALUES (?, ?, ?)
            ''', (
                profile.pipeline_name,
                json.dumps(profile_data, default=str),
                profile.timestamp.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store performance profile: {e}")

    def get_performance_summary(self, pipeline_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for pipeline(s)."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            if pipeline_name:
                # Get summary for specific pipeline
                cursor.execute('''
                    SELECT metric_type, AVG(value), MIN(value), MAX(value), COUNT(*)
                    FROM performance_metrics 
                    WHERE pipeline_name = ?
                    GROUP BY metric_type
                ''', (pipeline_name,))
            else:
                # Get summary for all pipelines
                cursor.execute('''
                    SELECT pipeline_name, metric_type, AVG(value), MIN(value), MAX(value), COUNT(*)
                    FROM performance_metrics 
                    GROUP BY pipeline_name, metric_type
                ''')
            
            results = cursor.fetchall()
            conn.close()
            
            if pipeline_name:
                summary = {}
                for row in results:
                    summary[row[0]] = {
                        'average': row[1],
                        'minimum': row[2],
                        'maximum': row[3],
                        'sample_count': row[4]
                    }
                return {pipeline_name: summary}
            else:
                summary = defaultdict(dict)
                for row in results:
                    pipeline = row[0]
                    metric_type = row[1]
                    summary[pipeline][metric_type] = {
                        'average': row[2],
                        'minimum': row[3],
                        'maximum': row[4],
                        'sample_count': row[5]
                    }
                return dict(summary)
                
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}

    def cleanup_old_metrics(self, days_to_keep: int = 90):
        """Clean up old performance metrics to prevent database bloat."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            cursor.execute('''
                DELETE FROM performance_metrics 
                WHERE timestamp < ?
            ''', (cutoff_date.isoformat(),))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up {deleted_count} old performance metrics")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Monitor")
    parser.add_argument("--root", default=".", help="Repository root path")
    parser.add_argument("--pipeline", help="Pipeline name to monitor")
    parser.add_argument("--summary", action='store_true', help="Show performance summary")
    parser.add_argument("--cleanup", type=int, help="Cleanup metrics older than N days")
    parser.add_argument("--baseline", action='store_true', help="Update baselines")
    parser.add_argument("--monitor-interval", type=float, default=1.0, help="Monitoring interval in seconds")
    parser.add_argument("--verbose", action='store_true', help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    monitor = PerformanceMonitor(args.root, args.monitor_interval)
    
    if args.summary:
        summary = monitor.get_performance_summary(args.pipeline)
        if summary:
            print(json.dumps(summary, indent=2, default=str))
        else:
            print("No performance data available")
    
    elif args.cleanup:
        monitor.cleanup_old_metrics(args.cleanup)
        print(f"Cleaned up metrics older than {args.cleanup} days")
    
    elif args.baseline:
        if args.pipeline:
            monitor._update_baselines(args.pipeline)
            print(f"Updated baselines for {args.pipeline}")
        else:
            print("Please specify --pipeline for baseline updates")
    
    else:
        print("Use --summary for performance summary")
        print("Use --cleanup N to cleanup old metrics")
        print("Use --baseline --pipeline NAME to update baselines")


if __name__ == "__main__":
    main()