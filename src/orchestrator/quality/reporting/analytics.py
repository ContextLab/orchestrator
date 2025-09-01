"""
Quality Analytics Engine with Trend Analysis and Insights Generation

This module provides sophisticated analytics capabilities for quality control
reporting. It analyzes quality metrics over time, identifies trends, generates
actionable insights, and provides predictive analysis for quality improvement.

Key Features:
- Time-series trend analysis and pattern detection
- Statistical analysis of quality metrics
- Anomaly detection and quality degradation alerts  
- Predictive quality modeling and forecasting
- Actionable insight generation with recommendations
- Performance correlation analysis
- Quality score calculations and benchmarking
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import time
import threading
from pathlib import Path
import json

from .metrics import QualityMetricsCollector, TimeSeriesMetric, MetricsSnapshot, QualityMetric, MetricType
from ..validation.engine import ValidationSession
from ..validation.rules import RuleSeverity, RuleCategory
from ..logging.logger import StructuredLogger, get_logger, LogCategory


class TrendDirection(Enum):
    """Direction of trend analysis."""
    IMPROVING = "improving"
    DECLINING = "declining" 
    STABLE = "stable"
    VOLATILE = "volatile"
    INSUFFICIENT_DATA = "insufficient_data"


class InsightType(Enum):
    """Types of quality insights."""
    TREND = "trend"
    ANOMALY = "anomaly"
    RECOMMENDATION = "recommendation"
    WARNING = "warning"
    PERFORMANCE = "performance"
    THRESHOLD = "threshold"
    PATTERN = "pattern"
    CORRELATION = "correlation"


@dataclass
class TrendAnalysis:
    """Analysis of metric trends over time."""
    metric_name: str
    direction: TrendDirection
    confidence: float  # 0.0 to 1.0
    slope: float       # Rate of change
    r_squared: float   # Correlation coefficient
    time_window_hours: int
    data_points: int
    start_value: float
    end_value: float
    average_value: float
    volatility: float  # Standard deviation / mean
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def percentage_change(self) -> float:
        """Calculate percentage change over the trend period."""
        if self.start_value == 0:
            return 0.0
        return ((self.end_value - self.start_value) / abs(self.start_value)) * 100
    
    @property
    def change_description(self) -> str:
        """Human-readable description of the change."""
        pct_change = abs(self.percentage_change)
        
        if self.direction == TrendDirection.IMPROVING:
            return f"improving by {pct_change:.1f}%"
        elif self.direction == TrendDirection.DECLINING:
            return f"declining by {pct_change:.1f}%"
        elif self.direction == TrendDirection.STABLE:
            return f"stable (±{pct_change:.1f}%)"
        elif self.direction == TrendDirection.VOLATILE:
            return f"volatile (σ={self.volatility:.2f})"
        else:
            return "insufficient data for trend analysis"


@dataclass
class QualityInsight:
    """Actionable quality insight with recommendations."""
    insight_type: InsightType
    severity: str  # info, warning, error, critical
    title: str
    description: str
    metric_name: Optional[str] = None
    current_value: Optional[Union[float, int]] = None
    threshold_value: Optional[Union[float, int]] = None
    confidence: float = 1.0  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)
    recommendations: List[str] = field(default_factory=list)
    related_metrics: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def datetime(self) -> datetime:
        """Get insight timestamp as datetime."""
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)


@dataclass
class AnalyticsResult:
    """Comprehensive analytics results."""
    analysis_timestamp: float
    time_window_hours: int
    trend_analyses: List[TrendAnalysis] = field(default_factory=list)
    quality_insights: List[QualityInsight] = field(default_factory=list)
    summary_statistics: Dict[str, Any] = field(default_factory=dict)
    quality_score: Optional[float] = None
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def datetime(self) -> datetime:
        """Get analysis timestamp as datetime."""
        return datetime.fromtimestamp(self.analysis_timestamp, tz=timezone.utc)


class QualityAnalytics:
    """
    Comprehensive quality analytics engine with trend analysis and insights.
    
    Provides sophisticated analysis of quality metrics, trend detection,
    anomaly identification, and actionable insights for quality improvement.
    """
    
    def __init__(
        self,
        metrics_collector: QualityMetricsCollector,
        logger: Optional[StructuredLogger] = None,
        trend_analysis_window_hours: int = 24,
        trend_confidence_threshold: float = 0.7,
        anomaly_threshold_std_devs: float = 2.0
    ):
        """
        Initialize quality analytics engine.
        
        Args:
            metrics_collector: Metrics collector instance
            logger: Optional structured logger
            trend_analysis_window_hours: Window for trend analysis
            trend_confidence_threshold: Minimum confidence for trend detection
            anomaly_threshold_std_devs: Standard deviations for anomaly detection
        """
        self.metrics_collector = metrics_collector
        self.logger = logger or get_logger("quality_analytics")
        self.trend_analysis_window_hours = trend_analysis_window_hours
        self.trend_confidence_threshold = trend_confidence_threshold
        self.anomaly_threshold_std_devs = anomaly_threshold_std_devs
        
        # Analysis state
        self._analysis_cache: Dict[str, AnalyticsResult] = {}
        self._cache_lock = threading.RLock()
        self._insight_generators: List[Callable] = []
        
        # Quality thresholds for scoring
        self._quality_thresholds = {
            'quality_score_excellent': 90.0,
            'quality_score_good': 75.0,
            'quality_score_fair': 60.0,
            'violation_rate_warning': 0.1,  # 10% violation rate warning
            'violation_rate_critical': 0.25,  # 25% violation rate critical
            'success_rate_warning': 0.9,  # 90% success rate warning
            'success_rate_critical': 0.75   # 75% success rate critical
        }
        
        # Register default insight generators
        self._register_default_insight_generators()
        
        self.logger.info("Initialized QualityAnalytics engine", category=LogCategory.MONITORING)
    
    def _register_default_insight_generators(self) -> None:
        """Register default insight generation functions."""
        self._insight_generators.extend([
            self._generate_trend_insights,
            self._generate_threshold_insights,
            self._generate_performance_insights,
            self._generate_anomaly_insights,
            self._generate_pattern_insights
        ])
    
    def add_insight_generator(self, generator: Callable) -> None:
        """Add custom insight generator function."""
        self._insight_generators.append(generator)
    
    def analyze_quality_trends(
        self,
        time_window_hours: Optional[int] = None,
        metric_patterns: Optional[List[str]] = None
    ) -> AnalyticsResult:
        """
        Perform comprehensive quality trend analysis.
        
        Args:
            time_window_hours: Analysis time window (default from config)
            metric_patterns: Optional patterns to filter metrics
            
        Returns:
            AnalyticsResult with trend analysis and insights
        """
        analysis_start = time.perf_counter()
        window_hours = time_window_hours or self.trend_analysis_window_hours
        analysis_timestamp = time.time()
        
        self.logger.info(f"Starting quality trend analysis (window: {window_hours}h)", category=LogCategory.MONITORING)
        
        try:
            # Get metrics for analysis
            all_metrics = self.metrics_collector.get_all_metrics()
            
            # Filter metrics if patterns provided
            if metric_patterns:
                filtered_metrics = {}
                import fnmatch
                for pattern in metric_patterns:
                    for key, metric in all_metrics.items():
                        if fnmatch.fnmatch(metric.name, pattern):
                            filtered_metrics[key] = metric
                all_metrics = filtered_metrics
            
            # Perform trend analysis on each metric
            trend_analyses = []
            for key, time_series in all_metrics.items():
                trend_analysis = self._analyze_metric_trend(time_series, window_hours)
                if trend_analysis and trend_analysis.confidence >= self.trend_confidence_threshold:
                    trend_analyses.append(trend_analysis)
            
            # Generate insights
            insights = []
            for generator in self._insight_generators:
                try:
                    generated_insights = generator(all_metrics, trend_analyses, window_hours)
                    if generated_insights:
                        insights.extend(generated_insights)
                except Exception as e:
                    self.logger.error(f"Insight generator failed: {e}", category=LogCategory.MONITORING, exception=e)
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_statistics(all_metrics, trend_analyses, window_hours)
            
            # Calculate overall quality score
            quality_score = self._calculate_overall_quality_score(all_metrics, trend_analyses, window_hours)
            
            # Generate top-level recommendations
            recommendations = self._generate_recommendations(trend_analyses, insights, quality_score)
            
            # Create result
            result = AnalyticsResult(
                analysis_timestamp=analysis_timestamp,
                time_window_hours=window_hours,
                trend_analyses=trend_analyses,
                quality_insights=insights,
                summary_statistics=summary_stats,
                quality_score=quality_score,
                recommendations=recommendations
            )
            
            # Cache result
            with self._cache_lock:
                cache_key = f"{window_hours}h_{int(analysis_timestamp/300)*300}"  # 5-minute buckets
                self._analysis_cache[cache_key] = result
            
            analysis_duration = (time.perf_counter() - analysis_start) * 1000
            self.logger.info(
                f"Completed quality trend analysis in {analysis_duration:.2f}ms",
                category=LogCategory.MONITORING,
                metadata={
                    'trends_analyzed': len(trend_analyses),
                    'insights_generated': len(insights),
                    'quality_score': quality_score
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Quality trend analysis failed: {e}", category=LogCategory.MONITORING, exception=e)
            raise
    
    def _analyze_metric_trend(self, time_series: TimeSeriesMetric, window_hours: int) -> Optional[TrendAnalysis]:
        """Analyze trend for a single metric."""
        if not time_series.values:
            return None
        
        # Get values in time window
        cutoff_time = time.time() - (window_hours * 3600)
        window_values = [(ts, val) for ts, val in time_series.values if ts >= cutoff_time]
        
        if len(window_values) < 3:  # Need at least 3 points for trend analysis
            return TrendAnalysis(
                metric_name=time_series.name,
                direction=TrendDirection.INSUFFICIENT_DATA,
                confidence=0.0,
                slope=0.0,
                r_squared=0.0,
                time_window_hours=window_hours,
                data_points=len(window_values),
                start_value=0.0,
                end_value=0.0,
                average_value=0.0,
                volatility=0.0
            )
        
        # Extract timestamps and values
        timestamps = [ts for ts, _ in window_values]
        values = [val for _, val in window_values]
        
        # Normalize timestamps relative to first timestamp
        normalized_times = [(ts - timestamps[0]) / 3600 for ts in timestamps]  # Hours from start
        
        # Calculate linear regression
        slope, r_squared = self._calculate_linear_regression(normalized_times, values)
        
        # Determine trend direction
        direction = self._determine_trend_direction(slope, r_squared, values)
        
        # Calculate statistics
        start_value = values[0]
        end_value = values[-1] 
        average_value = statistics.mean(values)
        volatility = statistics.stdev(values) / abs(average_value) if average_value != 0 else 0.0
        
        # Calculate confidence based on R-squared and data quality
        confidence = min(1.0, r_squared * (len(values) / 10.0))  # Scale by data points
        
        return TrendAnalysis(
            metric_name=time_series.name,
            direction=direction,
            confidence=confidence,
            slope=slope,
            r_squared=r_squared,
            time_window_hours=window_hours,
            data_points=len(window_values),
            start_value=start_value,
            end_value=end_value,
            average_value=average_value,
            volatility=volatility,
            metadata={
                'labels': time_series.labels,
                'metric_type': time_series.metric_type.value
            }
        )
    
    def _calculate_linear_regression(self, x_values: List[float], y_values: List[float]) -> Tuple[float, float]:
        """Calculate linear regression slope and R-squared."""
        n = len(x_values)
        if n < 2:
            return 0.0, 0.0
        
        # Calculate means
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        # Calculate slope and intercept
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return 0.0, 0.0
        
        slope = numerator / denominator
        
        # Calculate R-squared
        y_pred = [slope * (x - x_mean) + y_mean for x in x_values]
        ss_res = sum((y - y_p) ** 2 for y, y_p in zip(y_values, y_pred))
        ss_tot = sum((y - y_mean) ** 2 for y in y_values)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return slope, max(0.0, r_squared)  # Ensure R-squared is non-negative
    
    def _determine_trend_direction(self, slope: float, r_squared: float, values: List[float]) -> TrendDirection:
        """Determine trend direction from slope and R-squared."""
        if r_squared < 0.3:  # Low correlation indicates volatility
            return TrendDirection.VOLATILE
        
        # Calculate volatility
        if len(values) > 1:
            mean_val = statistics.mean(values)
            if mean_val != 0:
                volatility = statistics.stdev(values) / abs(mean_val)
                if volatility > 0.3:  # High volatility relative to mean
                    return TrendDirection.VOLATILE
        
        # Determine direction based on slope
        slope_threshold = 0.01  # Minimum slope for trend detection
        
        if abs(slope) < slope_threshold:
            return TrendDirection.STABLE
        elif slope > 0:
            return TrendDirection.IMPROVING
        else:
            return TrendDirection.DECLINING
    
    def _generate_trend_insights(
        self,
        metrics: Dict[str, TimeSeriesMetric],
        trends: List[TrendAnalysis],
        window_hours: int
    ) -> List[QualityInsight]:
        """Generate insights based on trend analysis."""
        insights = []
        
        for trend in trends:
            if trend.confidence < self.trend_confidence_threshold:
                continue
            
            # Quality score trends
            if 'quality' in trend.metric_name.lower() and 'score' in trend.metric_name.lower():
                if trend.direction == TrendDirection.DECLINING:
                    insights.append(QualityInsight(
                        insight_type=InsightType.TREND,
                        severity='warning' if trend.percentage_change < -10 else 'info',
                        title=f"Quality Score Declining",
                        description=f"Quality score is {trend.change_description} over {window_hours}h",
                        metric_name=trend.metric_name,
                        current_value=trend.end_value,
                        confidence=trend.confidence,
                        recommendations=[
                            "Review recent validation rule changes",
                            "Analyze failing validation rules",
                            "Check for data quality issues in inputs"
                        ],
                        metadata={'trend_analysis': trend}
                    ))
                elif trend.direction == TrendDirection.IMPROVING:
                    insights.append(QualityInsight(
                        insight_type=InsightType.TREND,
                        severity='info',
                        title=f"Quality Score Improving",
                        description=f"Quality score is {trend.change_description} over {window_hours}h",
                        metric_name=trend.metric_name,
                        current_value=trend.end_value,
                        confidence=trend.confidence,
                        recommendations=[
                            "Monitor continued improvement",
                            "Document successful quality practices"
                        ],
                        metadata={'trend_analysis': trend}
                    ))
            
            # Violation trends
            elif 'violation' in trend.metric_name.lower():
                if trend.direction == TrendDirection.IMPROVING and trend.end_value < trend.start_value:
                    insights.append(QualityInsight(
                        insight_type=InsightType.TREND,
                        severity='info',
                        title="Violations Decreasing",
                        description=f"Violations are {trend.change_description} over {window_hours}h",
                        metric_name=trend.metric_name,
                        current_value=trend.end_value,
                        confidence=trend.confidence,
                        recommendations=["Continue current quality practices"]
                    ))
                elif trend.direction == TrendDirection.DECLINING and trend.end_value > trend.start_value:
                    insights.append(QualityInsight(
                        insight_type=InsightType.TREND,
                        severity='warning',
                        title="Violations Increasing",
                        description=f"Violations are {trend.change_description} over {window_hours}h",
                        metric_name=trend.metric_name,
                        current_value=trend.end_value,
                        confidence=trend.confidence,
                        recommendations=[
                            "Review validation rule effectiveness",
                            "Check for new error patterns",
                            "Analyze failing validation categories"
                        ]
                    ))
        
        return insights
    
    def _generate_threshold_insights(
        self,
        metrics: Dict[str, TimeSeriesMetric],
        trends: List[TrendAnalysis],
        window_hours: int
    ) -> List[QualityInsight]:
        """Generate insights based on threshold violations."""
        insights = []
        
        for key, time_series in metrics.items():
            latest_value = time_series.get_latest_value()
            if latest_value is None:
                continue
            
            # Quality score thresholds
            if 'quality' in time_series.name.lower() and 'score' in time_series.name.lower():
                if latest_value < self._quality_thresholds['quality_score_fair']:
                    insights.append(QualityInsight(
                        insight_type=InsightType.THRESHOLD,
                        severity='critical',
                        title="Quality Score Critical",
                        description=f"Quality score ({latest_value:.1f}) is below critical threshold",
                        metric_name=time_series.name,
                        current_value=latest_value,
                        threshold_value=self._quality_thresholds['quality_score_fair'],
                        recommendations=[
                            "Immediate quality review required",
                            "Check validation rule configuration",
                            "Review recent pipeline changes"
                        ]
                    ))
                elif latest_value < self._quality_thresholds['quality_score_good']:
                    insights.append(QualityInsight(
                        insight_type=InsightType.THRESHOLD,
                        severity='warning',
                        title="Quality Score Below Target",
                        description=f"Quality score ({latest_value:.1f}) is below target threshold",
                        metric_name=time_series.name,
                        current_value=latest_value,
                        threshold_value=self._quality_thresholds['quality_score_good'],
                        recommendations=[
                            "Review validation results",
                            "Optimize content quality processes"
                        ]
                    ))
            
            # Success rate thresholds
            if 'success_rate' in time_series.name.lower():
                success_rate = latest_value / 100.0 if latest_value > 1.0 else latest_value
                if success_rate < self._quality_thresholds['success_rate_critical']:
                    insights.append(QualityInsight(
                        insight_type=InsightType.THRESHOLD,
                        severity='critical',
                        title="Low Success Rate Critical",
                        description=f"Success rate ({success_rate*100:.1f}%) is critically low",
                        metric_name=time_series.name,
                        current_value=latest_value,
                        threshold_value=self._quality_thresholds['success_rate_critical'] * 100,
                        recommendations=[
                            "Investigate validation rule failures",
                            "Check system health and performance",
                            "Review error logs immediately"
                        ]
                    ))
                elif success_rate < self._quality_thresholds['success_rate_warning']:
                    insights.append(QualityInsight(
                        insight_type=InsightType.THRESHOLD,
                        severity='warning',
                        title="Low Success Rate Warning",
                        description=f"Success rate ({success_rate*100:.1f}%) is below warning threshold",
                        metric_name=time_series.name,
                        current_value=latest_value,
                        threshold_value=self._quality_thresholds['success_rate_warning'] * 100,
                        recommendations=[
                            "Monitor validation rule performance",
                            "Review recent validation changes"
                        ]
                    ))
        
        return insights
    
    def _generate_performance_insights(
        self,
        metrics: Dict[str, TimeSeriesMetric],
        trends: List[TrendAnalysis],
        window_hours: int
    ) -> List[QualityInsight]:
        """Generate performance-related insights."""
        insights = []
        
        # Look for performance degradation trends
        performance_trends = [t for t in trends if 'duration' in t.metric_name.lower() or 'time' in t.metric_name.lower()]
        
        for trend in performance_trends:
            if trend.direction == TrendDirection.DECLINING and trend.percentage_change > 50:  # >50% slower
                insights.append(QualityInsight(
                    insight_type=InsightType.PERFORMANCE,
                    severity='warning',
                    title="Performance Degradation",
                    description=f"Validation performance is {trend.change_description}",
                    metric_name=trend.metric_name,
                    current_value=trend.end_value,
                    confidence=trend.confidence,
                    recommendations=[
                        "Review validation rule complexity",
                        "Check system resource utilization",
                        "Consider parallel execution optimization"
                    ]
                ))
        
        return insights
    
    def _generate_anomaly_insights(
        self,
        metrics: Dict[str, TimeSeriesMetric],
        trends: List[TrendAnalysis],
        window_hours: int
    ) -> List[QualityInsight]:
        """Generate anomaly detection insights."""
        insights = []
        
        # Look for highly volatile metrics
        volatile_trends = [t for t in trends if t.direction == TrendDirection.VOLATILE and t.volatility > 0.5]
        
        for trend in volatile_trends:
            insights.append(QualityInsight(
                insight_type=InsightType.ANOMALY,
                severity='info',
                title="High Metric Volatility",
                description=f"Metric shows high volatility (σ={trend.volatility:.2f})",
                metric_name=trend.metric_name,
                current_value=trend.end_value,
                confidence=trend.confidence,
                recommendations=[
                    "Investigate sources of variability",
                    "Consider metric smoothing or aggregation",
                    "Review underlying data quality"
                ]
            ))
        
        return insights
    
    def _generate_pattern_insights(
        self,
        metrics: Dict[str, TimeSeriesMetric],
        trends: List[TrendAnalysis],
        window_hours: int
    ) -> List[QualityInsight]:
        """Generate pattern-based insights."""
        insights = []
        
        # Look for correlated issues across multiple metrics
        declining_trends = [t for t in trends if t.direction == TrendDirection.DECLINING]
        
        if len(declining_trends) >= 3:  # Multiple metrics declining
            insights.append(QualityInsight(
                insight_type=InsightType.PATTERN,
                severity='warning',
                title="Multiple Quality Metrics Declining",
                description=f"{len(declining_trends)} metrics showing declining trends",
                current_value=len(declining_trends),
                recommendations=[
                    "Comprehensive quality review needed",
                    "Check for systemic issues",
                    "Review recent system changes"
                ],
                related_metrics=[t.metric_name for t in declining_trends]
            ))
        
        return insights
    
    def _calculate_summary_statistics(
        self,
        metrics: Dict[str, TimeSeriesMetric],
        trends: List[TrendAnalysis],
        window_hours: int
    ) -> Dict[str, Any]:
        """Calculate summary statistics for analytics result."""
        stats = {
            'total_metrics_analyzed': len(metrics),
            'trends_detected': len(trends),
            'time_window_hours': window_hours,
            'analysis_timestamp': time.time()
        }
        
        # Trend direction distribution
        trend_directions = defaultdict(int)
        for trend in trends:
            trend_directions[trend.direction.value] += 1
        stats['trend_directions'] = dict(trend_directions)
        
        # Quality score statistics
        quality_scores = []
        for time_series in metrics.values():
            if 'quality' in time_series.name.lower() and 'score' in time_series.name.lower():
                latest_value = time_series.get_latest_value()
                if latest_value is not None:
                    quality_scores.append(latest_value)
        
        if quality_scores:
            stats['quality_score_stats'] = {
                'count': len(quality_scores),
                'mean': statistics.mean(quality_scores),
                'median': statistics.median(quality_scores),
                'min': min(quality_scores),
                'max': max(quality_scores),
                'stdev': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0
            }
        
        # Violation statistics
        violation_counts = []
        for time_series in metrics.values():
            if 'violation' in time_series.name.lower():
                latest_value = time_series.get_latest_value()
                if latest_value is not None:
                    violation_counts.append(latest_value)
        
        if violation_counts:
            stats['violation_stats'] = {
                'total': sum(violation_counts),
                'mean': statistics.mean(violation_counts),
                'max': max(violation_counts)
            }
        
        return stats
    
    def _calculate_overall_quality_score(
        self,
        metrics: Dict[str, TimeSeriesMetric],
        trends: List[TrendAnalysis],
        window_hours: int
    ) -> Optional[float]:
        """Calculate overall quality score from available metrics."""
        quality_scores = []
        success_rates = []
        violation_rates = []
        
        # Collect relevant metrics
        for time_series in metrics.values():
            latest_value = time_series.get_latest_value()
            if latest_value is None:
                continue
            
            if 'quality' in time_series.name.lower() and 'score' in time_series.name.lower():
                quality_scores.append(latest_value)
            elif 'success_rate' in time_series.name.lower():
                success_rates.append(latest_value)
            elif 'violation' in time_series.name.lower() and 'total' in time_series.name.lower():
                violation_rates.append(latest_value)
        
        if not quality_scores and not success_rates:
            return None
        
        # Calculate weighted overall score
        total_score = 0.0
        total_weight = 0.0
        
        # Quality scores (weight: 0.5)
        if quality_scores:
            avg_quality_score = statistics.mean(quality_scores)
            total_score += avg_quality_score * 0.5
            total_weight += 0.5
        
        # Success rates (weight: 0.3)
        if success_rates:
            avg_success_rate = statistics.mean(success_rates)
            # Convert to 0-100 scale if needed
            if avg_success_rate <= 1.0:
                avg_success_rate *= 100
            total_score += avg_success_rate * 0.3
            total_weight += 0.3
        
        # Violation penalty (weight: 0.2)
        if violation_rates:
            avg_violations = statistics.mean(violation_rates)
            # Penalty based on violations (more violations = lower score)
            violation_penalty = max(0, 100 - (avg_violations * 5))  # 5 points per violation
            total_score += violation_penalty * 0.2
            total_weight += 0.2
        
        if total_weight > 0:
            return total_score / total_weight
        
        return None
    
    def _generate_recommendations(
        self,
        trends: List[TrendAnalysis],
        insights: List[QualityInsight],
        quality_score: Optional[float]
    ) -> List[str]:
        """Generate top-level recommendations based on analysis."""
        recommendations = []
        
        # Quality score recommendations
        if quality_score is not None:
            if quality_score < 60:
                recommendations.append("Critical quality review required - overall score below acceptable threshold")
            elif quality_score < 75:
                recommendations.append("Quality improvement initiatives recommended")
            elif quality_score > 90:
                recommendations.append("Excellent quality - maintain current practices")
        
        # Trend-based recommendations
        declining_trends = [t for t in trends if t.direction == TrendDirection.DECLINING]
        if len(declining_trends) > len(trends) * 0.5:  # More than half declining
            recommendations.append("Multiple declining trends detected - comprehensive system review needed")
        
        # Insight-based recommendations
        critical_insights = [i for i in insights if i.severity == 'critical']
        if critical_insights:
            recommendations.append("Critical issues detected - immediate attention required")
        
        warning_insights = [i for i in insights if i.severity == 'warning']
        if len(warning_insights) > 5:
            recommendations.append("Multiple warnings detected - proactive quality measures recommended")
        
        return recommendations
    
    def get_cached_analysis(self, window_hours: int) -> Optional[AnalyticsResult]:
        """Get cached analysis result if available."""
        cache_key = f"{window_hours}h_{int(time.time()/300)*300}"
        
        with self._cache_lock:
            return self._analysis_cache.get(cache_key)
    
    def export_analysis(
        self,
        result: AnalyticsResult,
        output_path: Path,
        format: str = "json"
    ) -> None:
        """Export analysis results to file."""
        export_data = {
            'analysis_timestamp': result.analysis_timestamp,
            'time_window_hours': result.time_window_hours,
            'quality_score': result.quality_score,
            'summary_statistics': result.summary_statistics,
            'recommendations': result.recommendations,
            'trend_analyses': [
                {
                    'metric_name': t.metric_name,
                    'direction': t.direction.value,
                    'confidence': t.confidence,
                    'slope': t.slope,
                    'percentage_change': t.percentage_change,
                    'change_description': t.change_description,
                    'data_points': t.data_points,
                    'metadata': t.metadata
                }
                for t in result.trend_analyses
            ],
            'quality_insights': [
                {
                    'insight_type': i.insight_type.value,
                    'severity': i.severity,
                    'title': i.title,
                    'description': i.description,
                    'metric_name': i.metric_name,
                    'current_value': i.current_value,
                    'confidence': i.confidence,
                    'recommendations': i.recommendations,
                    'related_metrics': i.related_metrics
                }
                for i in result.quality_insights
            ]
        }
        
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported quality analysis to {output_path}", category=LogCategory.MONITORING)