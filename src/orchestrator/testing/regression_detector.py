"""Advanced regression detection system for pipeline performance monitoring."""

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from .performance_monitor import ExecutionMetrics, PerformanceBaseline

logger = logging.getLogger(__name__)


class RegressionSeverity(Enum):
    """Severity levels for performance regressions."""
    
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RegressionType(Enum):
    """Types of performance regressions that can be detected."""
    
    EXECUTION_TIME = "execution_time"
    COST_INCREASE = "cost_increase" 
    MEMORY_USAGE = "memory_usage"
    SUCCESS_RATE = "success_rate"
    QUALITY_DEGRADATION = "quality_degradation"
    THROUGHPUT_DECREASE = "throughput_decrease"
    RESOURCE_EFFICIENCY = "resource_efficiency"


@dataclass
class RegressionAlert:
    """Alert for detected performance regression."""
    
    pipeline_name: str
    regression_type: RegressionType
    severity: RegressionSeverity
    detected_at: datetime
    
    # Metrics comparison
    baseline_value: float
    current_value: float
    change_percent: float
    change_absolute: float
    
    # Alert details
    description: str
    recommendation: str
    confidence: float = 1.0  # 0.0 to 1.0
    
    # Supporting data
    sample_size: int = 1
    trend_direction: str = "unknown"  # "increasing", "decreasing", "stable"
    statistical_significance: float = 0.0  # p-value if applicable
    
    # Thresholds used for detection
    threshold_used: float = 0.0
    threshold_type: str = "percent_change"
    
    
    @property
    def is_actionable(self) -> bool:
        """Determine if this alert requires immediate action."""
        return self.severity in [RegressionSeverity.HIGH, RegressionSeverity.CRITICAL] and self.confidence >= 0.7
    
    @property
    def alert_summary(self) -> str:
        """Generate a concise alert summary."""
        direction = "increased" if self.change_absolute > 0 else "decreased"
        return (f"{self.regression_type.value.replace('_', ' ').title()} {direction} by "
               f"{abs(self.change_percent):.1f}% ({self.severity.value.upper()})")


@dataclass
class RegressionDetectionConfig:
    """Configuration for regression detection thresholds and sensitivity."""
    
    # Execution time regression thresholds (percentage increase)
    execution_time_warning_threshold: float = 30.0  # 30% increase
    execution_time_critical_threshold: float = 100.0  # 100% increase
    
    # Cost regression thresholds (percentage increase)
    cost_warning_threshold: float = 25.0  # 25% increase
    cost_critical_threshold: float = 50.0  # 50% increase
    
    # Memory usage regression thresholds (percentage increase)
    memory_warning_threshold: float = 40.0  # 40% increase
    memory_critical_threshold: float = 80.0  # 80% increase
    
    # Success rate regression thresholds (percentage decrease)
    success_rate_warning_threshold: float = 10.0  # 10% decrease
    success_rate_critical_threshold: float = 25.0  # 25% decrease
    
    # Quality score regression thresholds (absolute point decrease)
    quality_warning_threshold: float = 10.0  # 10 point decrease
    quality_critical_threshold: float = 20.0  # 20 point decrease
    
    # Throughput regression thresholds (percentage decrease)
    throughput_warning_threshold: float = 20.0  # 20% decrease
    throughput_critical_threshold: float = 40.0  # 40% decrease
    
    # Statistical significance requirements
    min_sample_size_for_significance: int = 5
    confidence_threshold_for_alerts: float = 0.6
    
    # Baseline requirements
    min_baseline_confidence: float = 0.5
    max_baseline_age_days: int = 30
    
    # Trend analysis parameters
    trend_analysis_window: int = 10  # Number of recent executions to analyze
    trend_significance_threshold: float = 0.7  # Confidence threshold for trend detection
    
    def get_thresholds_for_type(self, regression_type: RegressionType) -> Tuple[float, float]:
        """Get warning and critical thresholds for a specific regression type."""
        threshold_map = {
            RegressionType.EXECUTION_TIME: (self.execution_time_warning_threshold, 
                                           self.execution_time_critical_threshold),
            RegressionType.COST_INCREASE: (self.cost_warning_threshold, 
                                         self.cost_critical_threshold),
            RegressionType.MEMORY_USAGE: (self.memory_warning_threshold, 
                                        self.memory_critical_threshold),
            RegressionType.SUCCESS_RATE: (self.success_rate_warning_threshold, 
                                        self.success_rate_critical_threshold),
            RegressionType.QUALITY_DEGRADATION: (self.quality_warning_threshold, 
                                               self.quality_critical_threshold),
            RegressionType.THROUGHPUT_DECREASE: (self.throughput_warning_threshold, 
                                              self.throughput_critical_threshold)
        }
        return threshold_map.get(regression_type, (10.0, 25.0))


class RegressionDetector:
    """
    Advanced regression detection system with configurable thresholds and statistical analysis.
    
    Features:
    - Multiple regression type detection
    - Statistical significance testing
    - Trend analysis
    - Configurable sensitivity levels
    - False positive reduction through confidence scoring
    - Actionable recommendations for detected regressions
    """
    
    def __init__(self, config: Optional[RegressionDetectionConfig] = None):
        """
        Initialize regression detector.
        
        Args:
            config: Configuration for thresholds and detection parameters
        """
        self.config = config or RegressionDetectionConfig()
        logger.info("Initialized RegressionDetector with configured thresholds")
    
    def detect_regressions(self, 
                         pipeline_name: str,
                         recent_executions: List[ExecutionMetrics],
                         baseline: PerformanceBaseline,
                         include_trends: bool = True) -> List[RegressionAlert]:
        """
        Detect performance regressions by comparing recent executions to baseline.
        
        Args:
            pipeline_name: Name of the pipeline being analyzed
            recent_executions: Recent execution metrics
            baseline: Performance baseline for comparison
            include_trends: Whether to include trend analysis
            
        Returns:
            List[RegressionAlert]: Detected regression alerts
        """
        alerts = []
        
        # Validate inputs
        if not recent_executions:
            logger.warning(f"No recent executions provided for {pipeline_name}")
            return alerts
        
        if not self._is_baseline_valid(baseline):
            logger.warning(f"Baseline for {pipeline_name} is not valid for regression detection")
            return alerts
        
        # Filter to successful executions for most comparisons
        successful_executions = [e for e in recent_executions if e.success]
        
        if not successful_executions:
            # Check for success rate regression if all recent executions failed
            success_rate = 0.0
            if baseline.success_rate > self.config.success_rate_warning_threshold / 100:
                alerts.append(self._create_regression_alert(
                    pipeline_name=pipeline_name,
                    regression_type=RegressionType.SUCCESS_RATE,
                    baseline_value=baseline.success_rate * 100,
                    current_value=success_rate,
                    sample_size=len(recent_executions),
                    description="Pipeline success rate has dropped to 0%",
                    recommendation="Investigate recent pipeline failures and address critical issues"
                ))
            return alerts
        
        # 1. Execution Time Regression Detection
        execution_time_alert = self._detect_execution_time_regression(
            pipeline_name, successful_executions, baseline
        )
        if execution_time_alert:
            alerts.append(execution_time_alert)
        
        # 2. Cost Regression Detection
        cost_alert = self._detect_cost_regression(
            pipeline_name, successful_executions, baseline
        )
        if cost_alert:
            alerts.append(cost_alert)
        
        # 3. Memory Usage Regression Detection
        memory_alert = self._detect_memory_regression(
            pipeline_name, successful_executions, baseline
        )
        if memory_alert:
            alerts.append(memory_alert)
        
        # 4. Success Rate Regression Detection
        success_rate_alert = self._detect_success_rate_regression(
            pipeline_name, recent_executions, baseline
        )
        if success_rate_alert:
            alerts.append(success_rate_alert)
        
        # 5. Quality Degradation Detection
        quality_alert = self._detect_quality_regression(
            pipeline_name, successful_executions, baseline
        )
        if quality_alert:
            alerts.append(quality_alert)
        
        # 6. Throughput Regression Detection
        throughput_alert = self._detect_throughput_regression(
            pipeline_name, successful_executions, baseline
        )
        if throughput_alert:
            alerts.append(throughput_alert)
        
        # 7. Trend Analysis (if requested)
        if include_trends and len(recent_executions) >= self.config.trend_analysis_window:
            trend_alerts = self._analyze_performance_trends(pipeline_name, recent_executions)
            alerts.extend(trend_alerts)
        
        # Sort alerts by severity and confidence
        alerts.sort(key=lambda a: (a.severity == RegressionSeverity.CRITICAL, 
                                  a.severity == RegressionSeverity.HIGH,
                                  a.confidence), reverse=True)
        
        logger.info(f"Detected {len(alerts)} regression alerts for {pipeline_name}")
        for alert in alerts:
            logger.info(f"  - {alert.alert_summary} (confidence: {alert.confidence:.2f})")
        
        return alerts
    
    def _is_baseline_valid(self, baseline: PerformanceBaseline) -> bool:
        """Check if baseline is valid for regression detection."""
        # Check baseline confidence
        if baseline.baseline_confidence < self.config.min_baseline_confidence:
            return False
        
        # Check baseline age
        baseline_age = (datetime.now() - baseline.baseline_date).days
        if baseline_age > self.config.max_baseline_age_days:
            return False
        
        # Check sample size
        if baseline.sample_count < 3:
            return False
        
        return True
    
    def _detect_execution_time_regression(self, 
                                        pipeline_name: str,
                                        executions: List[ExecutionMetrics],
                                        baseline: PerformanceBaseline) -> Optional[RegressionAlert]:
        """Detect execution time regression."""
        if not executions:
            return None
        
        # Calculate current average execution time
        execution_times = [e.execution_time_seconds for e in executions]
        current_avg_time = statistics.mean(execution_times)
        
        # Compare to baseline
        baseline_time = baseline.avg_execution_time
        if baseline_time <= 0:
            return None
        
        percent_increase = ((current_avg_time - baseline_time) / baseline_time) * 100
        
        # Check thresholds
        warning_threshold, critical_threshold = self.config.get_thresholds_for_type(
            RegressionType.EXECUTION_TIME
        )
        
        if percent_increase < warning_threshold:
            return None
        
        # Calculate confidence based on sample size and consistency
        confidence = self._calculate_confidence(execution_times, len(executions))
        
        # Determine severity
        if percent_increase >= critical_threshold:
            severity = RegressionSeverity.CRITICAL
        elif percent_increase >= warning_threshold * 2:
            severity = RegressionSeverity.HIGH
        elif percent_increase >= warning_threshold * 1.5:
            severity = RegressionSeverity.MEDIUM
        else:
            severity = RegressionSeverity.LOW
        
        return self._create_regression_alert(
            pipeline_name=pipeline_name,
            regression_type=RegressionType.EXECUTION_TIME,
            baseline_value=baseline_time,
            current_value=current_avg_time,
            sample_size=len(executions),
            severity=severity,
            confidence=confidence,
            description=f"Execution time increased by {percent_increase:.1f}% from baseline",
            recommendation=self._get_execution_time_recommendation(percent_increase),
            threshold_used=warning_threshold if percent_increase >= warning_threshold else critical_threshold
        )
    
    def _detect_cost_regression(self, 
                              pipeline_name: str,
                              executions: List[ExecutionMetrics],
                              baseline: PerformanceBaseline) -> Optional[RegressionAlert]:
        """Detect cost regression."""
        if not executions or baseline.avg_cost <= 0:
            return None
        
        # Calculate current average cost
        costs = [e.estimated_cost_usd for e in executions if e.estimated_cost_usd > 0]
        if not costs:
            return None
        
        current_avg_cost = statistics.mean(costs)
        baseline_cost = baseline.avg_cost
        
        percent_increase = ((current_avg_cost - baseline_cost) / baseline_cost) * 100
        
        # Check thresholds
        warning_threshold, critical_threshold = self.config.get_thresholds_for_type(
            RegressionType.COST_INCREASE
        )
        
        if percent_increase < warning_threshold:
            return None
        
        confidence = self._calculate_confidence(costs, len(costs))
        
        # Determine severity
        if percent_increase >= critical_threshold:
            severity = RegressionSeverity.CRITICAL
        elif percent_increase >= warning_threshold * 2:
            severity = RegressionSeverity.HIGH  
        elif percent_increase >= warning_threshold * 1.5:
            severity = RegressionSeverity.MEDIUM
        else:
            severity = RegressionSeverity.LOW
        
        return self._create_regression_alert(
            pipeline_name=pipeline_name,
            regression_type=RegressionType.COST_INCREASE,
            baseline_value=baseline_cost,
            current_value=current_avg_cost,
            sample_size=len(costs),
            severity=severity,
            confidence=confidence,
            description=f"Execution cost increased by {percent_increase:.1f}% from baseline",
            recommendation=self._get_cost_recommendation(percent_increase),
            threshold_used=warning_threshold if percent_increase >= warning_threshold else critical_threshold
        )
    
    def _detect_memory_regression(self, 
                                pipeline_name: str,
                                executions: List[ExecutionMetrics],
                                baseline: PerformanceBaseline) -> Optional[RegressionAlert]:
        """Detect memory usage regression."""
        if not executions or baseline.avg_memory_mb <= 0:
            return None
        
        # Calculate current average memory usage
        memory_values = [e.peak_memory_mb for e in executions if e.peak_memory_mb > 0]
        if not memory_values:
            return None
        
        current_avg_memory = statistics.mean(memory_values)
        baseline_memory = baseline.avg_memory_mb
        
        percent_increase = ((current_avg_memory - baseline_memory) / baseline_memory) * 100
        
        # Check thresholds
        warning_threshold, critical_threshold = self.config.get_thresholds_for_type(
            RegressionType.MEMORY_USAGE
        )
        
        if percent_increase < warning_threshold:
            return None
        
        confidence = self._calculate_confidence(memory_values, len(memory_values))
        
        # Determine severity
        if percent_increase >= critical_threshold:
            severity = RegressionSeverity.CRITICAL
        elif percent_increase >= warning_threshold * 2:
            severity = RegressionSeverity.HIGH
        else:
            severity = RegressionSeverity.MEDIUM
        
        return self._create_regression_alert(
            pipeline_name=pipeline_name,
            regression_type=RegressionType.MEMORY_USAGE,
            baseline_value=baseline_memory,
            current_value=current_avg_memory,
            sample_size=len(memory_values),
            severity=severity,
            confidence=confidence,
            description=f"Memory usage increased by {percent_increase:.1f}% from baseline",
            recommendation=self._get_memory_recommendation(percent_increase),
            threshold_used=warning_threshold if percent_increase >= warning_threshold else critical_threshold
        )
    
    def _detect_success_rate_regression(self, 
                                       pipeline_name: str,
                                       executions: List[ExecutionMetrics],
                                       baseline: PerformanceBaseline) -> Optional[RegressionAlert]:
        """Detect success rate regression."""
        if not executions:
            return None
        
        # Calculate current success rate
        successful_count = sum(1 for e in executions if e.success)
        current_success_rate = (successful_count / len(executions)) * 100
        baseline_success_rate = baseline.success_rate * 100
        
        percent_decrease = baseline_success_rate - current_success_rate
        
        # Check thresholds
        warning_threshold, critical_threshold = self.config.get_thresholds_for_type(
            RegressionType.SUCCESS_RATE
        )
        
        if percent_decrease < warning_threshold:
            return None
        
        confidence = self._calculate_success_rate_confidence(len(executions), successful_count)
        
        # Determine severity
        if percent_decrease >= critical_threshold or current_success_rate == 0:
            severity = RegressionSeverity.CRITICAL
        elif percent_decrease >= warning_threshold * 2:
            severity = RegressionSeverity.HIGH
        else:
            severity = RegressionSeverity.MEDIUM
        
        return self._create_regression_alert(
            pipeline_name=pipeline_name,
            regression_type=RegressionType.SUCCESS_RATE,
            baseline_value=baseline_success_rate,
            current_value=current_success_rate,
            sample_size=len(executions),
            severity=severity,
            confidence=confidence,
            description=f"Success rate decreased by {percent_decrease:.1f}% from baseline",
            recommendation=self._get_success_rate_recommendation(current_success_rate),
            threshold_used=warning_threshold if percent_decrease >= warning_threshold else critical_threshold
        )
    
    def _detect_quality_regression(self, 
                                 pipeline_name: str,
                                 executions: List[ExecutionMetrics],
                                 baseline: PerformanceBaseline) -> Optional[RegressionAlert]:
        """Detect quality score regression."""
        if not executions or baseline.avg_quality_score is None:
            return None
        
        # Calculate current average quality score
        quality_scores = [e.quality_score for e in executions 
                         if e.quality_score is not None]
        if not quality_scores:
            return None
        
        current_avg_quality = statistics.mean(quality_scores)
        baseline_quality = baseline.avg_quality_score
        
        quality_decrease = baseline_quality - current_avg_quality
        
        # Check thresholds
        warning_threshold, critical_threshold = self.config.get_thresholds_for_type(
            RegressionType.QUALITY_DEGRADATION
        )
        
        if quality_decrease < warning_threshold:
            return None
        
        confidence = self._calculate_confidence(quality_scores, len(quality_scores))
        
        # Determine severity
        if quality_decrease >= critical_threshold:
            severity = RegressionSeverity.CRITICAL
        elif quality_decrease >= warning_threshold * 2:
            severity = RegressionSeverity.HIGH
        else:
            severity = RegressionSeverity.MEDIUM
        
        return self._create_regression_alert(
            pipeline_name=pipeline_name,
            regression_type=RegressionType.QUALITY_DEGRADATION,
            baseline_value=baseline_quality,
            current_value=current_avg_quality,
            sample_size=len(quality_scores),
            severity=severity,
            confidence=confidence,
            description=f"Quality score decreased by {quality_decrease:.1f} points from baseline",
            recommendation=self._get_quality_recommendation(quality_decrease),
            threshold_used=warning_threshold if quality_decrease >= warning_threshold else critical_threshold
        )
    
    def _detect_throughput_regression(self, 
                                    pipeline_name: str,
                                    executions: List[ExecutionMetrics],
                                    baseline: PerformanceBaseline) -> Optional[RegressionAlert]:
        """Detect throughput regression."""
        if not executions:
            return None
        
        # Calculate current average throughput (tokens per second)
        throughputs = [e.throughput_tokens_per_second for e in executions 
                      if e.throughput_tokens_per_second > 0]
        if not throughputs:
            return None
        
        current_avg_throughput = statistics.mean(throughputs)
        
        # Use baseline throughput or calculate from baseline metrics
        baseline_throughput = baseline.avg_tokens_per_second
        if baseline_throughput <= 0:
            # Fallback: estimate from baseline execution time if tokens data available
            return None
        
        percent_decrease = ((baseline_throughput - current_avg_throughput) / baseline_throughput) * 100
        
        # Check thresholds
        warning_threshold, critical_threshold = self.config.get_thresholds_for_type(
            RegressionType.THROUGHPUT_DECREASE
        )
        
        if percent_decrease < warning_threshold:
            return None
        
        confidence = self._calculate_confidence(throughputs, len(throughputs))
        
        # Determine severity
        if percent_decrease >= critical_threshold:
            severity = RegressionSeverity.CRITICAL
        elif percent_decrease >= warning_threshold * 2:
            severity = RegressionSeverity.HIGH
        else:
            severity = RegressionSeverity.MEDIUM
        
        return self._create_regression_alert(
            pipeline_name=pipeline_name,
            regression_type=RegressionType.THROUGHPUT_DECREASE,
            baseline_value=baseline_throughput,
            current_value=current_avg_throughput,
            sample_size=len(throughputs),
            severity=severity,
            confidence=confidence,
            description=f"Throughput decreased by {percent_decrease:.1f}% from baseline",
            recommendation=self._get_throughput_recommendation(percent_decrease),
            threshold_used=warning_threshold if percent_decrease >= warning_threshold else critical_threshold
        )
    
    def _analyze_performance_trends(self, 
                                  pipeline_name: str,
                                  executions: List[ExecutionMetrics]) -> List[RegressionAlert]:
        """Analyze performance trends over recent executions."""
        trend_alerts = []
        
        if len(executions) < self.config.trend_analysis_window:
            return trend_alerts
        
        # Sort executions by start time
        sorted_executions = sorted(executions, key=lambda e: e.start_time)
        recent_window = sorted_executions[-self.config.trend_analysis_window:]
        
        # Analyze execution time trend
        execution_times = [e.execution_time_seconds for e in recent_window if e.success]
        if len(execution_times) >= 5:
            trend_alert = self._analyze_metric_trend(
                pipeline_name=pipeline_name,
                metric_name="execution_time",
                values=execution_times,
                regression_type=RegressionType.EXECUTION_TIME,
                is_negative_trend_bad=True
            )
            if trend_alert:
                trend_alerts.append(trend_alert)
        
        # Analyze cost trend
        costs = [e.estimated_cost_usd for e in recent_window 
                if e.success and e.estimated_cost_usd > 0]
        if len(costs) >= 5:
            trend_alert = self._analyze_metric_trend(
                pipeline_name=pipeline_name,
                metric_name="cost",
                values=costs,
                regression_type=RegressionType.COST_INCREASE,
                is_negative_trend_bad=False
            )
            if trend_alert:
                trend_alerts.append(trend_alert)
        
        return trend_alerts
    
    def _analyze_metric_trend(self, 
                            pipeline_name: str,
                            metric_name: str,
                            values: List[float],
                            regression_type: RegressionType,
                            is_negative_trend_bad: bool) -> Optional[RegressionAlert]:
        """Analyze trend for a specific metric."""
        if len(values) < 5:
            return None
        
        # Calculate simple linear trend
        x_values = list(range(len(values)))
        trend_slope = self._calculate_trend_slope(x_values, values)
        trend_direction = "increasing" if trend_slope > 0 else "decreasing"
        
        # Calculate trend significance (correlation coefficient)
        correlation = self._calculate_correlation(x_values, values)
        trend_strength = abs(correlation)
        
        # Determine if this trend is concerning
        is_concerning_trend = (
            (is_negative_trend_bad and trend_slope > 0) or  # Increasing bad metric
            (not is_negative_trend_bad and trend_slope < 0)  # Decreasing good metric
        )
        
        if not is_concerning_trend or trend_strength < self.config.trend_significance_threshold:
            return None
        
        # Calculate trend magnitude
        first_half_avg = statistics.mean(values[:len(values)//2])
        second_half_avg = statistics.mean(values[len(values)//2:])
        
        if first_half_avg <= 0:
            return None
        
        trend_change_percent = abs(((second_half_avg - first_half_avg) / first_half_avg) * 100)
        
        # Determine severity based on trend strength and change magnitude
        if trend_strength >= 0.9 and trend_change_percent >= 25:
            severity = RegressionSeverity.HIGH
        elif trend_strength >= 0.8 and trend_change_percent >= 15:
            severity = RegressionSeverity.MEDIUM
        else:
            severity = RegressionSeverity.LOW
        
        return self._create_regression_alert(
            pipeline_name=pipeline_name,
            regression_type=regression_type,
            baseline_value=first_half_avg,
            current_value=second_half_avg,
            sample_size=len(values),
            severity=severity,
            confidence=trend_strength,
            description=f"Concerning {trend_direction} trend detected in {metric_name} "
                       f"(change: {trend_change_percent:.1f}%)",
            recommendation=f"Monitor {metric_name} closely; consider investigating root cause",
            threshold_used=self.config.trend_significance_threshold,
            threshold_type="trend_correlation"
        )
    
    def _calculate_trend_slope(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate linear trend slope using least squares."""
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
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _calculate_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        n = len(x_values)
        if n < 2:
            return 0.0
        
        mean_x = statistics.mean(x_values)
        mean_y = statistics.mean(y_values)
        
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
        
        sum_sq_x = sum((x - mean_x) ** 2 for x in x_values)
        sum_sq_y = sum((y - mean_y) ** 2 for y in y_values)
        
        denominator = (sum_sq_x * sum_sq_y) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        correlation = numerator / denominator
        return correlation
    
    def _calculate_confidence(self, values: List[float], sample_size: int) -> float:
        """Calculate confidence score based on sample size and data consistency."""
        if sample_size < 3:
            return 0.3
        elif sample_size < 5:
            return 0.5
        elif sample_size < 10:
            return 0.7
        
        # Factor in data consistency (lower coefficient of variation = higher confidence)
        if len(values) > 1 and statistics.mean(values) > 0:
            cv = statistics.stdev(values) / statistics.mean(values)
            consistency_factor = max(0.2, 1.0 - cv)
            return min(1.0, 0.7 + (0.3 * consistency_factor))
        
        return 0.8
    
    def _calculate_success_rate_confidence(self, total_executions: int, successful_executions: int) -> float:
        """Calculate confidence for success rate metrics."""
        if total_executions < 5:
            return 0.4
        elif total_executions < 10:
            return 0.6
        else:
            return 0.8
    
    def _create_regression_alert(self, 
                               pipeline_name: str,
                               regression_type: RegressionType,
                               baseline_value: float,
                               current_value: float,
                               sample_size: int,
                               severity: Optional[RegressionSeverity] = None,
                               confidence: Optional[float] = None,
                               description: str = "",
                               recommendation: str = "",
                               threshold_used: float = 0.0,
                               threshold_type: str = "percent_change") -> RegressionAlert:
        """Create a regression alert with all necessary information."""
        
        if severity is None:
            # Auto-determine severity based on change magnitude
            if baseline_value > 0:
                change_percent = abs(((current_value - baseline_value) / baseline_value) * 100)
                if change_percent >= 50:
                    severity = RegressionSeverity.CRITICAL
                elif change_percent >= 25:
                    severity = RegressionSeverity.HIGH
                elif change_percent >= 15:
                    severity = RegressionSeverity.MEDIUM
                else:
                    severity = RegressionSeverity.LOW
            else:
                severity = RegressionSeverity.MEDIUM
        
        if confidence is None:
            confidence = self._calculate_confidence([current_value], sample_size)
        
        # Calculate change metrics
        if baseline_value > 0:
            change_percent = ((current_value - baseline_value) / baseline_value) * 100
        else:
            change_percent = float('inf') if current_value > 0 else 0.0
        change_absolute = current_value - baseline_value
        
        return RegressionAlert(
            pipeline_name=pipeline_name,
            regression_type=regression_type,
            severity=severity,
            detected_at=datetime.now(),
            baseline_value=baseline_value,
            current_value=current_value,
            change_percent=change_percent,
            change_absolute=change_absolute,
            sample_size=sample_size,
            description=description,
            recommendation=recommendation,
            confidence=confidence,
            threshold_used=threshold_used,
            threshold_type=threshold_type
        )
    
    def _get_execution_time_recommendation(self, percent_increase: float) -> str:
        """Generate recommendation for execution time regression."""
        if percent_increase >= 100:
            return ("URGENT: Execution time doubled. Check for infinite loops, "
                   "resource contention, or inefficient API calls.")
        elif percent_increase >= 50:
            return ("Investigate recent changes to pipeline logic, model selection, "
                   "or input data complexity.")
        else:
            return ("Monitor execution time trends and consider optimizing "
                   "computationally expensive operations.")
    
    def _get_cost_recommendation(self, percent_increase: float) -> str:
        """Generate recommendation for cost regression."""
        if percent_increase >= 50:
            return ("Review model usage patterns, token consumption, and API call efficiency. "
                   "Consider using smaller models where appropriate.")
        else:
            return ("Monitor cost trends and optimize token usage through better prompting "
                   "or result caching.")
    
    def _get_memory_recommendation(self, percent_increase: float) -> str:
        """Generate recommendation for memory regression."""
        if percent_increase >= 80:
            return ("Check for memory leaks, large data structures, or inefficient "
                   "resource management.")
        else:
            return ("Monitor memory usage patterns and consider implementing "
                   "memory optimization strategies.")
    
    def _get_success_rate_recommendation(self, current_success_rate: float) -> str:
        """Generate recommendation for success rate regression."""
        if current_success_rate == 0:
            return ("CRITICAL: All recent executions failed. Investigate error logs "
                   "and address blocking issues immediately.")
        elif current_success_rate < 50:
            return ("High failure rate detected. Review error patterns and "
                   "implement error handling improvements.")
        else:
            return ("Monitor error patterns and investigate intermittent failures.")
    
    def _get_quality_recommendation(self, quality_decrease: float) -> str:
        """Generate recommendation for quality regression."""
        if quality_decrease >= 20:
            return ("Significant quality degradation detected. Review output quality, "
                   "prompt engineering, and model performance.")
        else:
            return ("Monitor output quality and consider quality assurance improvements.")
    
    def _get_throughput_recommendation(self, percent_decrease: float) -> str:
        """Generate recommendation for throughput regression."""
        if percent_decrease >= 40:
            return ("Significant throughput decrease. Check for API throttling, "
                   "network issues, or processing bottlenecks.")
        else:
            return ("Monitor throughput trends and optimize processing efficiency.")
    
    def get_regression_summary(self, alerts: List[RegressionAlert]) -> Dict[str, Any]:
        """Generate a summary of regression alerts."""
        if not alerts:
            return {"status": "healthy", "total_alerts": 0}
        
        # Group by severity
        severity_counts = {}
        for severity in RegressionSeverity:
            severity_counts[severity.value] = len([a for a in alerts if a.severity == severity])
        
        # Group by type
        type_counts = {}
        for reg_type in RegressionType:
            type_counts[reg_type.value] = len([a for a in alerts if a.regression_type == reg_type])
        
        # Get actionable alerts
        actionable_alerts = [a for a in alerts if a.is_actionable]
        
        return {
            "status": "regression_detected" if actionable_alerts else "warning",
            "total_alerts": len(alerts),
            "actionable_alerts": len(actionable_alerts),
            "severity_breakdown": severity_counts,
            "type_breakdown": type_counts,
            "highest_severity": max(alerts, key=lambda a: list(RegressionSeverity).index(a.severity)).severity.value,
            "average_confidence": statistics.mean([a.confidence for a in alerts]),
            "recommendations": [a.recommendation for a in actionable_alerts]
        }