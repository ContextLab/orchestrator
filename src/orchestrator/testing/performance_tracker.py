"""Historical performance tracking and trend analysis system."""

import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    _VISUALIZATION_AVAILABLE = True
except ImportError:
    _VISUALIZATION_AVAILABLE = False

from .performance_monitor import ExecutionMetrics, PerformanceBaseline, PerformanceMonitor
from .regression_detector import RegressionAlert, RegressionDetector, RegressionDetectionConfig

logger = logging.getLogger(__name__)


@dataclass
class PerformanceTrend:
    """Performance trend analysis result."""
    
    pipeline_name: str
    metric_name: str
    trend_period_days: int
    
    # Trend statistics
    trend_direction: str  # "improving", "degrading", "stable"
    trend_strength: float  # 0.0 to 1.0
    change_percent: float
    change_absolute: float
    
    # Statistical measures
    correlation_coefficient: float
    p_value: float
    confidence_interval: Tuple[float, float]
    
    # Data points
    sample_count: int
    first_value: float
    last_value: float
    min_value: float
    max_value: float
    mean_value: float
    
    # Trend assessment
    is_significant: bool
    significance_threshold: float = 0.05
    trend_quality: str = "good"  # "good", "concerning", "critical"
    
    @property
    def trend_summary(self) -> str:
        """Generate human-readable trend summary."""
        direction_word = {
            "improving": "improved",
            "degrading": "degraded", 
            "stable": "remained stable"
        }.get(self.trend_direction, "changed")
        
        return (f"{self.metric_name.replace('_', ' ').title()} has {direction_word} by "
               f"{abs(self.change_percent):.1f}% over {self.trend_period_days} days "
               f"({self.trend_quality} trend)")


@dataclass
class PipelinePerformanceProfile:
    """Comprehensive performance profile for a pipeline."""
    
    # Required fields (no defaults)
    pipeline_name: str
    profile_date: datetime
    analysis_period_days: int
    total_executions: int
    successful_executions: int
    failed_executions: int
    success_rate: float
    avg_execution_time: float
    median_execution_time: float
    p95_execution_time: float
    avg_cost: float
    total_cost: float
    avg_memory_usage: float
    peak_memory_usage: float
    avg_throughput_tokens_per_sec: float
    
    # Optional fields (with defaults)
    execution_time_trend: Optional[PerformanceTrend] = None
    cost_trend: Optional[PerformanceTrend] = None
    memory_trend: Optional[PerformanceTrend] = None
    avg_quality_score: Optional[float] = None
    quality_trend: Optional[PerformanceTrend] = None
    throughput_trend: Optional[PerformanceTrend] = None
    active_regressions: List[RegressionAlert] = field(default_factory=list)
    resolved_regressions: List[RegressionAlert] = field(default_factory=list)
    execution_time_stability: float = 1.0  # Coefficient of variation (lower = more stable)
    cost_stability: float = 1.0
    has_baseline: bool = False
    baseline_age_days: int = 0
    baseline_confidence: float = 0.0
    
    @property
    def overall_health_score(self) -> float:
        """Calculate overall performance health score (0-100)."""
        score = 50.0  # Start with neutral score
        
        # Success rate contribution (30 points)
        score += (self.success_rate * 30)
        
        # Trend contribution (25 points)
        trend_bonus = 0
        for trend in [self.execution_time_trend, self.cost_trend, self.quality_trend]:
            if trend:
                if trend.trend_direction == "improving":
                    trend_bonus += 8
                elif trend.trend_direction == "stable":
                    trend_bonus += 5
                elif trend.trend_quality == "concerning":
                    trend_bonus -= 3
                elif trend.trend_quality == "critical":
                    trend_bonus -= 8
        
        score += min(25, max(-25, trend_bonus))
        
        # Stability contribution (20 points)
        stability_score = 20 * (2 - min(2, max(0, self.execution_time_stability)))
        score += stability_score
        
        # Regression penalty (up to -15 points)
        critical_regressions = len([a for a in self.active_regressions 
                                   if a.severity.value in ['critical', 'high']])
        score -= min(15, critical_regressions * 7)
        
        # Quality bonus (15 points)
        if self.avg_quality_score:
            quality_bonus = (self.avg_quality_score / 100) * 15
            score += quality_bonus
        
        return max(0.0, min(100.0, score))
    
    @property
    def performance_status(self) -> str:
        """Get overall performance status."""
        health_score = self.overall_health_score
        
        if health_score >= 85:
            return "excellent"
        elif health_score >= 70:
            return "good"
        elif health_score >= 50:
            return "fair"
        elif health_score >= 30:
            return "poor"
        else:
            return "critical"


class PerformanceTracker:
    """
    Comprehensive performance tracking and analysis system.
    
    Features:
    - Historical performance data analysis
    - Trend detection and analysis
    - Performance profiling for pipelines
    - Regression tracking and resolution
    - Performance visualization and reporting
    - Baseline management and updates
    """
    
    def __init__(self, 
                 performance_monitor: PerformanceMonitor,
                 regression_detector: Optional[RegressionDetector] = None,
                 enable_visualization: bool = True):
        """
        Initialize performance tracker.
        
        Args:
            performance_monitor: PerformanceMonitor instance
            regression_detector: RegressionDetector instance (optional)
            enable_visualization: Enable performance visualization features
        """
        self.performance_monitor = performance_monitor
        self.regression_detector = regression_detector or RegressionDetector()
        self.enable_visualization = enable_visualization
        
        # Tracking configuration
        self.default_analysis_period = 30  # days
        self.min_samples_for_trend = 5
        self.trend_significance_threshold = 0.05
        
        # Cache for performance profiles
        self._performance_profiles: Dict[str, PipelinePerformanceProfile] = {}
        self._last_profile_update: Dict[str, datetime] = {}
        
        logger.info("Initialized PerformanceTracker")
    
    def track_pipeline_performance(self, 
                                 pipeline_name: str,
                                 analysis_period_days: Optional[int] = None,
                                 force_refresh: bool = False) -> PipelinePerformanceProfile:
        """
        Create comprehensive performance profile for a pipeline.
        
        Args:
            pipeline_name: Name of the pipeline to analyze
            analysis_period_days: Days to analyze (default: 30)
            force_refresh: Force refresh of cached data
            
        Returns:
            PipelinePerformanceProfile: Comprehensive performance analysis
        """
        analysis_period = analysis_period_days or self.default_analysis_period
        
        # Check if we have recent cached data
        cache_key = f"{pipeline_name}_{analysis_period}"
        if (not force_refresh and 
            pipeline_name in self._performance_profiles and
            pipeline_name in self._last_profile_update):
            
            last_update = self._last_profile_update[pipeline_name]
            if (datetime.now() - last_update).total_seconds() < 3600:  # 1 hour cache
                logger.info(f"Using cached performance profile for {pipeline_name}")
                return self._performance_profiles[pipeline_name]
        
        logger.info(f"Analyzing performance for {pipeline_name} over {analysis_period} days")
        
        # Get execution history
        executions = self.performance_monitor.get_execution_history(
            pipeline_name=pipeline_name,
            days_back=analysis_period,
            include_failed=True
        )
        
        if not executions:
            logger.warning(f"No execution data found for {pipeline_name}")
            return self._create_empty_profile(pipeline_name, analysis_period)
        
        # Calculate basic statistics
        successful_executions = [e for e in executions if e.success]
        failed_executions = [e for e in executions if not e.success]
        
        success_rate = len(successful_executions) / len(executions) if executions else 0.0
        
        # Performance metrics (from successful executions only)
        if successful_executions:
            exec_times = [e.execution_time_seconds for e in successful_executions]
            costs = [e.estimated_cost_usd for e in successful_executions if e.estimated_cost_usd > 0]
            memory_values = [e.peak_memory_mb for e in successful_executions if e.peak_memory_mb > 0]
            quality_scores = [e.quality_score for e in successful_executions 
                            if e.quality_score is not None]
            throughput_values = [e.throughput_tokens_per_second for e in successful_executions 
                               if e.throughput_tokens_per_second > 0]
            
            # Calculate statistics
            avg_execution_time = mean(exec_times) if exec_times else 0.0
            median_execution_time = median(exec_times) if exec_times else 0.0
            p95_execution_time = self._calculate_percentile(exec_times, 95) if exec_times else 0.0
            execution_time_stability = (stdev(exec_times) / mean(exec_times)) if len(exec_times) > 1 else 0.0
            
            avg_cost = mean(costs) if costs else 0.0
            total_cost = sum(costs) if costs else 0.0
            cost_stability = (stdev(costs) / mean(costs)) if len(costs) > 1 and mean(costs) > 0 else 0.0
            
            avg_memory_usage = mean(memory_values) if memory_values else 0.0
            peak_memory_usage = max(memory_values) if memory_values else 0.0
            
            avg_quality_score = mean(quality_scores) if quality_scores else None
            avg_throughput_tokens_per_sec = mean(throughput_values) if throughput_values else 0.0
            
        else:
            # No successful executions
            avg_execution_time = median_execution_time = p95_execution_time = 0.0
            execution_time_stability = 0.0
            avg_cost = total_cost = cost_stability = 0.0
            avg_memory_usage = peak_memory_usage = 0.0
            avg_quality_score = None
            avg_throughput_tokens_per_sec = 0.0
        
        # Analyze trends
        execution_time_trend = self._analyze_metric_trend(
            executions, "execution_time_seconds", analysis_period, "execution_time"
        )
        cost_trend = self._analyze_metric_trend(
            executions, "estimated_cost_usd", analysis_period, "cost"
        )
        memory_trend = self._analyze_metric_trend(
            executions, "peak_memory_mb", analysis_period, "memory_usage"
        )
        quality_trend = self._analyze_metric_trend(
            executions, "quality_score", analysis_period, "quality_score"
        )
        throughput_trend = self._analyze_metric_trend(
            executions, "throughput_tokens_per_second", analysis_period, "throughput"
        )
        
        # Get baseline information
        baseline = self.performance_monitor.get_baseline(pipeline_name)
        has_baseline = baseline is not None
        baseline_age_days = 0
        baseline_confidence = 0.0
        
        if baseline:
            baseline_age_days = (datetime.now() - baseline.baseline_date).days
            baseline_confidence = baseline.baseline_confidence
        
        # Detect active regressions
        active_regressions = []
        if baseline and successful_executions:
            recent_executions = sorted(executions, key=lambda e: e.start_time)[-10:]  # Last 10 executions
            active_regressions = self.regression_detector.detect_regressions(
                pipeline_name, recent_executions, baseline
            )
        
        # Create performance profile
        profile = PipelinePerformanceProfile(
            pipeline_name=pipeline_name,
            profile_date=datetime.now(),
            analysis_period_days=analysis_period,
            total_executions=len(executions),
            successful_executions=len(successful_executions),
            failed_executions=len(failed_executions),
            success_rate=success_rate,
            avg_execution_time=avg_execution_time,
            median_execution_time=median_execution_time,
            p95_execution_time=p95_execution_time,
            execution_time_trend=execution_time_trend,
            avg_cost=avg_cost,
            total_cost=total_cost,
            cost_trend=cost_trend,
            avg_memory_usage=avg_memory_usage,
            peak_memory_usage=peak_memory_usage,
            memory_trend=memory_trend,
            avg_quality_score=avg_quality_score,
            quality_trend=quality_trend,
            avg_throughput_tokens_per_sec=avg_throughput_tokens_per_sec,
            throughput_trend=throughput_trend,
            active_regressions=active_regressions,
            execution_time_stability=execution_time_stability,
            cost_stability=cost_stability,
            has_baseline=has_baseline,
            baseline_age_days=baseline_age_days,
            baseline_confidence=baseline_confidence
        )
        
        # Cache the profile
        self._performance_profiles[pipeline_name] = profile
        self._last_profile_update[pipeline_name] = datetime.now()
        
        logger.info(f"Generated performance profile for {pipeline_name} "
                   f"(health score: {profile.overall_health_score:.1f}, "
                   f"status: {profile.performance_status})")
        
        return profile
    
    def _analyze_metric_trend(self, 
                            executions: List[ExecutionMetrics],
                            metric_attr: str,
                            period_days: int,
                            metric_name: str) -> Optional[PerformanceTrend]:
        """Analyze trend for a specific metric."""
        # Filter to successful executions and extract metric values
        successful = [e for e in executions if e.success]
        values = []
        timestamps = []
        
        for execution in successful:
            value = getattr(execution, metric_attr, None)
            if value is not None and (metric_attr != "estimated_cost_usd" or value > 0):
                values.append(float(value))
                timestamps.append(execution.start_time)
        
        if len(values) < self.min_samples_for_trend:
            return None
        
        # Sort by timestamp
        paired_data = list(zip(timestamps, values))
        paired_data.sort(key=lambda x: x[0])
        timestamps, values = zip(*paired_data)
        
        # Convert timestamps to numeric values (days since first execution)
        first_timestamp = timestamps[0]
        x_values = [(ts - first_timestamp).total_seconds() / 86400 for ts in timestamps]
        
        # Calculate trend statistics
        correlation = self._calculate_correlation(x_values, values)
        trend_slope = self._calculate_trend_slope(x_values, values)
        
        # Determine trend direction and strength
        trend_strength = abs(correlation)
        if trend_strength < 0.3:
            trend_direction = "stable"
        elif trend_slope > 0:
            trend_direction = "degrading" if metric_name in ["execution_time", "cost", "memory_usage"] else "improving"
        else:
            trend_direction = "improving" if metric_name in ["execution_time", "cost", "memory_usage"] else "degrading"
        
        # Calculate change metrics
        first_value = values[0]
        last_value = values[-1]
        change_absolute = last_value - first_value
        change_percent = (change_absolute / first_value * 100) if first_value != 0 else 0.0
        
        # Statistical significance (simplified)
        is_significant = trend_strength > 0.5 and len(values) >= 10
        
        # Assess trend quality
        if trend_strength >= 0.8 and abs(change_percent) >= 25:
            trend_quality = "critical"
        elif trend_strength >= 0.6 and abs(change_percent) >= 15:
            trend_quality = "concerning"
        else:
            trend_quality = "good"
        
        return PerformanceTrend(
            pipeline_name=executions[0].pipeline_name if executions else "unknown",
            metric_name=metric_name,
            trend_period_days=period_days,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            change_percent=change_percent,
            change_absolute=change_absolute,
            correlation_coefficient=correlation,
            p_value=0.0,  # Simplified for now
            confidence_interval=(min(values), max(values)),
            sample_count=len(values),
            first_value=first_value,
            last_value=last_value,
            min_value=min(values),
            max_value=max(values),
            mean_value=mean(values),
            is_significant=is_significant,
            trend_quality=trend_quality
        )
    
    def get_performance_summary(self, 
                              pipeline_names: Optional[List[str]] = None,
                              analysis_period_days: int = 30) -> Dict[str, Any]:
        """
        Get performance summary for multiple pipelines.
        
        Args:
            pipeline_names: List of pipeline names (None for all)
            analysis_period_days: Analysis period in days
            
        Returns:
            Dict: Performance summary across pipelines
        """
        if pipeline_names is None:
            # Get all pipelines with recent execution data
            all_executions = self.performance_monitor.get_execution_history(
                days_back=analysis_period_days
            )
            pipeline_names = list(set(e.pipeline_name for e in all_executions))
        
        profiles = {}
        summary_stats = {
            "total_pipelines": len(pipeline_names),
            "healthy_pipelines": 0,
            "concerning_pipelines": 0,
            "critical_pipelines": 0,
            "total_executions": 0,
            "total_cost": 0.0,
            "average_success_rate": 0.0,
            "pipelines_with_regressions": 0
        }
        
        health_scores = []
        success_rates = []
        
        for pipeline_name in pipeline_names:
            try:
                profile = self.track_pipeline_performance(pipeline_name, analysis_period_days)
                profiles[pipeline_name] = {
                    "health_score": profile.overall_health_score,
                    "status": profile.performance_status,
                    "success_rate": profile.success_rate,
                    "total_executions": profile.total_executions,
                    "total_cost": profile.total_cost,
                    "active_regressions": len(profile.active_regressions),
                    "execution_time_trend": profile.execution_time_trend.trend_direction if profile.execution_time_trend else "stable",
                    "cost_trend": profile.cost_trend.trend_direction if profile.cost_trend else "stable"
                }
                
                # Update summary stats
                health_scores.append(profile.overall_health_score)
                success_rates.append(profile.success_rate)
                summary_stats["total_executions"] += profile.total_executions
                summary_stats["total_cost"] += profile.total_cost
                
                if profile.performance_status in ["excellent", "good"]:
                    summary_stats["healthy_pipelines"] += 1
                elif profile.performance_status in ["fair", "poor"]:
                    summary_stats["concerning_pipelines"] += 1
                else:
                    summary_stats["critical_pipelines"] += 1
                
                if profile.active_regressions:
                    summary_stats["pipelines_with_regressions"] += 1
            
            except Exception as e:
                logger.error(f"Failed to analyze {pipeline_name}: {e}")
                profiles[pipeline_name] = {"error": str(e)}
        
        # Calculate aggregate metrics
        if health_scores:
            summary_stats["average_health_score"] = mean(health_scores)
            summary_stats["min_health_score"] = min(health_scores)
            summary_stats["max_health_score"] = max(health_scores)
        
        if success_rates:
            summary_stats["average_success_rate"] = mean(success_rates)
        
        return {
            "summary": summary_stats,
            "pipeline_profiles": profiles,
            "analysis_period_days": analysis_period_days,
            "generated_at": datetime.now().isoformat()
        }
    
    def generate_performance_report(self, 
                                  pipeline_name: str,
                                  output_path: Optional[Path] = None,
                                  include_visualizations: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive performance report for a pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            output_path: Path to save report files
            include_visualizations: Include charts and graphs
            
        Returns:
            Dict: Report data
        """
        profile = self.track_pipeline_performance(pipeline_name)
        
        # Build report data
        report = {
            "pipeline_name": pipeline_name,
            "report_generated": datetime.now().isoformat(),
            "analysis_period_days": profile.analysis_period_days,
            "performance_summary": {
                "health_score": profile.overall_health_score,
                "status": profile.performance_status,
                "success_rate": profile.success_rate,
                "total_executions": profile.total_executions,
                "total_cost": profile.total_cost,
                "avg_execution_time": profile.avg_execution_time,
                "stability_score": 1.0 - min(1.0, profile.execution_time_stability)
            },
            "trends": {
                "execution_time": asdict(profile.execution_time_trend) if profile.execution_time_trend else None,
                "cost": asdict(profile.cost_trend) if profile.cost_trend else None,
                "memory": asdict(profile.memory_trend) if profile.memory_trend else None,
                "quality": asdict(profile.quality_trend) if profile.quality_trend else None,
                "throughput": asdict(profile.throughput_trend) if profile.throughput_trend else None
            },
            "regressions": {
                "active_count": len(profile.active_regressions),
                "active_alerts": [asdict(alert) for alert in profile.active_regressions],
                "critical_alerts": [asdict(alert) for alert in profile.active_regressions 
                                   if alert.severity.value in ['critical', 'high']]
            },
            "baseline_info": {
                "has_baseline": profile.has_baseline,
                "baseline_age_days": profile.baseline_age_days,
                "baseline_confidence": profile.baseline_confidence
            },
            "recommendations": self._generate_recommendations(profile)
        }
        
        # Generate visualizations if requested
        if include_visualizations and self.enable_visualization and output_path:
            try:
                viz_paths = self._create_performance_visualizations(pipeline_name, profile, output_path)
                report["visualizations"] = viz_paths
            except Exception as e:
                logger.warning(f"Failed to generate visualizations: {e}")
        
        # Save report to file if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            report_file = output_path / f"{pipeline_name}_performance_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Performance report saved to {report_file}")
        
        return report
    
    def _create_empty_profile(self, pipeline_name: str, analysis_period: int) -> PipelinePerformanceProfile:
        """Create empty performance profile when no data is available."""
        return PipelinePerformanceProfile(
            pipeline_name=pipeline_name,
            profile_date=datetime.now(),
            analysis_period_days=analysis_period,
            total_executions=0,
            successful_executions=0,
            failed_executions=0,
            success_rate=0.0,
            avg_execution_time=0.0,
            median_execution_time=0.0,
            p95_execution_time=0.0,
            avg_cost=0.0,
            total_cost=0.0,
            avg_memory_usage=0.0,
            peak_memory_usage=0.0,
            avg_throughput_tokens_per_sec=0.0
        )
    
    def _calculate_percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of a dataset."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _calculate_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        n = len(x_values)
        if n < 2:
            return 0.0
        
        mean_x = mean(x_values)
        mean_y = mean(y_values)
        
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
        
        sum_sq_x = sum((x - mean_x) ** 2 for x in x_values)
        sum_sq_y = sum((y - mean_y) ** 2 for y in y_values)
        
        denominator = (sum_sq_x * sum_sq_y) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_trend_slope(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate linear trend slope."""
        n = len(x_values)
        if n < 2:
            return 0.0
        
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        return (n * sum_xy - sum_x * sum_y) / denominator
    
    def _generate_recommendations(self, profile: PipelinePerformanceProfile) -> List[str]:
        """Generate actionable recommendations based on performance profile."""
        recommendations = []
        
        # Success rate recommendations
        if profile.success_rate < 0.8:
            recommendations.append(
                f"Success rate is {profile.success_rate:.1%}. Investigate and address "
                f"failure patterns to improve reliability."
            )
        
        # Performance trend recommendations
        if profile.execution_time_trend and profile.execution_time_trend.trend_direction == "degrading":
            recommendations.append(
                f"Execution time is trending upward ({profile.execution_time_trend.change_percent:.1f}% increase). "
                f"Consider optimizing pipeline logic or reviewing recent changes."
            )
        
        if profile.cost_trend and profile.cost_trend.trend_direction == "degrading":
            recommendations.append(
                f"Costs are increasing ({profile.cost_trend.change_percent:.1f}% increase). "
                f"Review model usage and consider optimization strategies."
            )
        
        # Active regression recommendations
        if profile.active_regressions:
            critical_regressions = [r for r in profile.active_regressions 
                                  if r.severity.value in ['critical', 'high']]
            if critical_regressions:
                recommendations.append(
                    f"URGENT: {len(critical_regressions)} critical/high severity regressions detected. "
                    f"Immediate investigation required."
                )
        
        # Baseline recommendations
        if not profile.has_baseline:
            recommendations.append(
                "No performance baseline established. Run more executions to establish baseline "
                "for better regression detection."
            )
        elif profile.baseline_age_days > 30:
            recommendations.append(
                f"Performance baseline is {profile.baseline_age_days} days old. "
                f"Consider updating baseline with recent data."
            )
        
        # Stability recommendations
        if profile.execution_time_stability > 1.0:
            recommendations.append(
                "High execution time variability detected. Consider investigating "
                "sources of performance inconsistency."
            )
        
        return recommendations
    
    def _create_performance_visualizations(self, 
                                         pipeline_name: str,
                                         profile: PipelinePerformanceProfile,
                                         output_path: Path) -> Dict[str, str]:
        """Create performance visualization charts."""
        viz_paths = {}
        
        try:
            if not _VISUALIZATION_AVAILABLE:
                logger.warning("Visualization libraries not available, skipping charts")
                return viz_paths
                
            # Get historical execution data
            executions = self.performance_monitor.get_execution_history(
                pipeline_name=pipeline_name,
                days_back=profile.analysis_period_days,
                include_failed=False
            )
            
            if not executions:
                return viz_paths
            
            # Convert to DataFrame for easier plotting
            execution_data = []
            for execution in executions:
                execution_data.append({
                    'timestamp': execution.start_time,
                    'execution_time': execution.execution_time_seconds,
                    'cost': execution.estimated_cost_usd,
                    'memory_mb': execution.peak_memory_mb,
                    'quality_score': execution.quality_score,
                    'success': execution.success
                })
            
            df = pd.DataFrame(execution_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Set style
            try:
                plt.style.use('seaborn-v0_8')
            except:
                try:
                    plt.style.use('seaborn')
                except:
                    pass  # Use default style
            
            # 1. Execution Time Trend Chart
            if not df['execution_time'].empty:
                plt.figure(figsize=(12, 6))
                plt.plot(df['timestamp'], df['execution_time'], marker='o', alpha=0.7)
                plt.title(f'{pipeline_name} - Execution Time Trend')
                plt.xlabel('Date')
                plt.ylabel('Execution Time (seconds)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                exec_time_path = output_path / f"{pipeline_name}_execution_time_trend.png"
                plt.savefig(exec_time_path, dpi=300, bbox_inches='tight')
                plt.close()
                viz_paths['execution_time_trend'] = str(exec_time_path)
            
            # 2. Cost Trend Chart
            cost_data = df[df['cost'] > 0]
            if not cost_data.empty:
                plt.figure(figsize=(12, 6))
                plt.plot(cost_data['timestamp'], cost_data['cost'], marker='s', alpha=0.7, color='orange')
                plt.title(f'{pipeline_name} - Cost Trend')
                plt.xlabel('Date')
                plt.ylabel('Cost (USD)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                cost_path = output_path / f"{pipeline_name}_cost_trend.png"
                plt.savefig(cost_path, dpi=300, bbox_inches='tight')
                plt.close()
                viz_paths['cost_trend'] = str(cost_path)
            
            # 3. Performance Dashboard (multi-metric)
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{pipeline_name} - Performance Dashboard', fontsize=16)
            
            # Execution time subplot
            if not df['execution_time'].empty:
                axes[0, 0].plot(df['timestamp'], df['execution_time'], 'b-', alpha=0.7)
                axes[0, 0].set_title('Execution Time')
                axes[0, 0].set_ylabel('Seconds')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Cost subplot
            if not cost_data.empty:
                axes[0, 1].plot(cost_data['timestamp'], cost_data['cost'], 'orange', alpha=0.7)
                axes[0, 1].set_title('Cost')
                axes[0, 1].set_ylabel('USD')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Memory usage subplot
            memory_data = df[df['memory_mb'] > 0]
            if not memory_data.empty:
                axes[1, 0].plot(memory_data['timestamp'], memory_data['memory_mb'], 'g-', alpha=0.7)
                axes[1, 0].set_title('Memory Usage')
                axes[1, 0].set_ylabel('MB')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Quality score subplot
            quality_data = df[df['quality_score'].notna()]
            if not quality_data.empty:
                axes[1, 1].plot(quality_data['timestamp'], quality_data['quality_score'], 'r-', alpha=0.7)
                axes[1, 1].set_title('Quality Score')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            dashboard_path = output_path / f"{pipeline_name}_performance_dashboard.png"
            plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths['performance_dashboard'] = str(dashboard_path)
        
        except ImportError:
            logger.warning("Matplotlib/Pandas not available, skipping visualizations")
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
        
        return viz_paths