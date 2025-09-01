"""
Comprehensive tests for quality analytics system.

Tests all aspects of the QualityAnalytics engine including:
- Trend analysis and pattern detection
- Quality insight generation  
- Statistical analysis and scoring
- Analytics result export
- Integration with metrics collector
"""

import pytest
import time
import tempfile
import json
import statistics
from pathlib import Path
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import List, Dict, Any

# Import the system under test
from src.orchestrator.quality.reporting.analytics import (
    QualityAnalytics,
    TrendDirection,
    InsightType,
    TrendAnalysis,
    QualityInsight,
    AnalyticsResult
)
from src.orchestrator.quality.reporting.metrics import (
    QualityMetricsCollector,
    TimeSeriesMetric,
    MetricType
)
from src.orchestrator.quality.validation.rules import RuleSeverity


@pytest.fixture
def mock_logger():
    """Create mock logger for testing."""
    return Mock()


@pytest.fixture
def mock_metrics_collector():
    """Create mock metrics collector with test data."""
    collector = Mock(spec=QualityMetricsCollector)
    
    # Create sample time series data
    test_metrics = {}
    
    # Quality score metric with improving trend
    quality_metric = TimeSeriesMetric("quality.overall.score", MetricType.GAUGE)
    base_time = time.time() - (24 * 3600)  # 24 hours ago
    for i in range(24):  # Hourly data points
        timestamp = base_time + (i * 3600)
        value = 60 + (i * 1.5)  # Improving from 60 to ~95
        quality_metric.add_value(value, timestamp)
    test_metrics["quality.overall.score"] = quality_metric
    
    # Violations metric with declining trend (fewer violations)
    violations_metric = TimeSeriesMetric("quality.validation.total_violations", MetricType.GAUGE)
    for i in range(24):
        timestamp = base_time + (i * 3600)
        value = max(0, 50 - (i * 2))  # Declining from 50 to 2
        violations_metric.add_value(value, timestamp)
    test_metrics["quality.validation.total_violations"] = violations_metric
    
    # Success rate metric with stable trend
    success_metric = TimeSeriesMetric("quality.validation.success_rate", MetricType.GAUGE)
    for i in range(24):
        timestamp = base_time + (i * 3600)
        value = 95 + (i % 3) - 1  # Stable around 95% with small variations
        success_metric.add_value(value, timestamp)
    test_metrics["quality.validation.success_rate"] = success_metric
    
    # Performance metric with volatile trend
    perf_metric = TimeSeriesMetric("quality.validation.duration_ms", MetricType.TIMER)
    for i in range(24):
        timestamp = base_time + (i * 3600)
        value = 1000 + (i % 7) * 500 + (i % 3) * 200  # Volatile between 1000-4200ms
        perf_metric.add_value(value, timestamp)
    test_metrics["quality.validation.duration_ms"] = perf_metric
    
    # Rule performance metrics
    rule_perf_metric = TimeSeriesMetric("quality.validation.rule_duration_ms", MetricType.TIMER, {"rule_id": "rule_1"})
    for i in range(24):
        timestamp = base_time + (i * 3600)
        value = 500 + (i * 50)  # Degrading performance
        rule_perf_metric.add_value(value, timestamp)
    test_metrics["quality.validation.rule_duration_ms{rule_id=rule_1}"] = rule_perf_metric
    
    collector.get_all_metrics.return_value = test_metrics
    collector.get_metrics_by_pattern.side_effect = lambda pattern: {
        k: v for k, v in test_metrics.items() if pattern in k or pattern == "*"
    }
    
    return collector


@pytest.fixture
def analytics(mock_metrics_collector, mock_logger):
    """Create QualityAnalytics instance for testing."""
    return QualityAnalytics(
        metrics_collector=mock_metrics_collector,
        logger=mock_logger,
        trend_analysis_window_hours=24,
        trend_confidence_threshold=0.5,
        anomaly_threshold_std_devs=2.0
    )


class TestQualityAnalytics:
    """Test suite for QualityAnalytics."""
    
    def test_initialization(self, mock_metrics_collector, mock_logger):
        """Test proper initialization of analytics engine."""
        analytics = QualityAnalytics(
            metrics_collector=mock_metrics_collector,
            logger=mock_logger,
            trend_analysis_window_hours=48,
            trend_confidence_threshold=0.8,
            anomaly_threshold_std_devs=3.0
        )
        
        assert analytics.metrics_collector == mock_metrics_collector
        assert analytics.logger == mock_logger
        assert analytics.trend_analysis_window_hours == 48
        assert analytics.trend_confidence_threshold == 0.8
        assert analytics.anomaly_threshold_std_devs == 3.0
        assert len(analytics._insight_generators) > 0  # Default generators registered
    
    def test_trend_analysis_calculation(self, analytics):
        """Test individual metric trend analysis."""
        # Get test metrics
        test_metrics = analytics.metrics_collector.get_all_metrics()
        quality_metric = test_metrics["quality.overall.score"]
        
        # Analyze trend
        trend = analytics._analyze_metric_trend(quality_metric, 24)
        
        assert trend is not None
        assert isinstance(trend, TrendAnalysis)
        assert trend.metric_name == "quality.overall.score"
        assert trend.direction in [TrendDirection.IMPROVING, TrendDirection.DECLINING, TrendDirection.STABLE]
        assert 0.0 <= trend.confidence <= 1.0
        assert trend.data_points > 0
        assert trend.time_window_hours == 24
        
        # For improving quality score, should detect improvement
        assert trend.direction == TrendDirection.IMPROVING
        assert trend.percentage_change > 0  # Should be positive change
    
    def test_linear_regression_calculation(self, analytics):
        """Test linear regression calculation."""
        # Test with perfect linear relationship
        x_values = [1, 2, 3, 4, 5]
        y_values = [2, 4, 6, 8, 10]  # y = 2x
        
        slope, r_squared = analytics._calculate_linear_regression(x_values, y_values)
        
        assert abs(slope - 2.0) < 0.001  # Should be close to 2.0
        assert abs(r_squared - 1.0) < 0.001  # Perfect correlation
        
        # Test with no relationship
        y_random = [1, 5, 3, 7, 4]
        slope_random, r_squared_random = analytics._calculate_linear_regression(x_values, y_random)
        
        assert r_squared_random < 0.5  # Should be low correlation
    
    def test_trend_direction_determination(self, analytics):
        """Test trend direction determination logic."""
        # Test improving trend
        direction = analytics._determine_trend_direction(slope=0.5, r_squared=0.8, values=[10, 15, 20, 25, 30])
        assert direction == TrendDirection.IMPROVING
        
        # Test declining trend
        direction = analytics._determine_trend_direction(slope=-0.3, r_squared=0.7, values=[30, 25, 20, 15, 10])
        assert direction == TrendDirection.DECLINING
        
        # Test stable trend
        direction = analytics._determine_trend_direction(slope=0.005, r_squared=0.9, values=[20, 20.1, 19.9, 20.05, 19.95])
        assert direction == TrendDirection.STABLE
        
        # Test volatile trend (low correlation)
        direction = analytics._determine_trend_direction(slope=0.1, r_squared=0.2, values=[10, 25, 5, 30, 8])
        assert direction == TrendDirection.VOLATILE
    
    def test_comprehensive_trend_analysis(self, analytics):
        """Test comprehensive trend analysis across all metrics."""
        result = analytics.analyze_quality_trends(time_window_hours=24)
        
        assert isinstance(result, AnalyticsResult)
        assert result.time_window_hours == 24
        assert len(result.trend_analyses) > 0
        assert len(result.quality_insights) > 0
        assert result.quality_score is not None
        assert 0.0 <= result.quality_score <= 100.0
        assert len(result.recommendations) > 0
        
        # Verify specific trends were detected
        quality_trends = [t for t in result.trend_analyses if "quality" in t.metric_name.lower()]
        assert len(quality_trends) > 0
        
        violation_trends = [t for t in result.trend_analyses if "violation" in t.metric_name.lower()]
        assert len(violation_trends) > 0
    
    def test_insight_generation(self, analytics):
        """Test quality insight generation."""
        result = analytics.analyze_quality_trends()
        insights = result.quality_insights
        
        assert len(insights) > 0
        
        # Verify insight structure
        for insight in insights:
            assert isinstance(insight, QualityInsight)
            assert insight.insight_type in [e for e in InsightType]
            assert insight.severity in ["info", "warning", "error", "critical"]
            assert len(insight.title) > 0
            assert len(insight.description) > 0
            assert 0.0 <= insight.confidence <= 1.0
        
        # Check for specific insight types
        trend_insights = [i for i in insights if i.insight_type == InsightType.TREND]
        threshold_insights = [i for i in insights if i.insight_type == InsightType.THRESHOLD]
        
        # Should have trend insights for our test data
        assert len(trend_insights) > 0
    
    def test_trend_insight_generation(self, analytics):
        """Test trend-specific insight generation."""
        test_metrics = analytics.metrics_collector.get_all_metrics()
        
        # Create mock trend analyses
        trends = [
            TrendAnalysis(
                metric_name="quality.overall.score",
                direction=TrendDirection.IMPROVING,
                confidence=0.9,
                slope=1.5,
                r_squared=0.85,
                time_window_hours=24,
                data_points=24,
                start_value=60,
                end_value=95,
                average_value=77.5,
                volatility=0.15
            ),
            TrendAnalysis(
                metric_name="quality.validation.total_violations",
                direction=TrendDirection.DECLINING,
                confidence=0.8,
                slope=-2.0,
                r_squared=0.9,
                time_window_hours=24,
                data_points=24,
                start_value=50,
                end_value=2,
                average_value=26,
                volatility=0.3
            )
        ]
        
        insights = analytics._generate_trend_insights(test_metrics, trends, 24)
        
        assert len(insights) >= 2
        
        # Check for improving quality score insight
        quality_insights = [i for i in insights if "quality" in i.title.lower() and "improving" in i.description.lower()]
        assert len(quality_insights) > 0
        
        # Check for decreasing violations insight
        violation_insights = [i for i in insights if "violation" in i.title.lower() and ("decreasing" in i.description.lower() or "declining" in i.description.lower())]
        assert len(violation_insights) > 0
    
    def test_threshold_insight_generation(self, analytics):
        """Test threshold-based insight generation."""
        test_metrics = analytics.metrics_collector.get_all_metrics()
        
        # Add low quality score metric
        low_quality_metric = TimeSeriesMetric("quality.test.score", MetricType.GAUGE)
        low_quality_metric.add_value(45)  # Below fair threshold (60)
        test_metrics["quality.test.score"] = low_quality_metric
        
        # Add low success rate metric
        low_success_metric = TimeSeriesMetric("quality.validation.success_rate", MetricType.GAUGE)
        low_success_metric.add_value(0.7)  # 70% - below critical threshold
        test_metrics["quality.validation.success_rate"] = low_success_metric
        
        insights = analytics._generate_threshold_insights(test_metrics, [], 24)
        
        assert len(insights) > 0
        
        # Check for quality score threshold insights
        quality_threshold_insights = [i for i in insights if i.insight_type == InsightType.THRESHOLD and "quality" in i.title.lower()]
        assert len(quality_threshold_insights) > 0
        
        # Check for success rate threshold insights  
        success_threshold_insights = [i for i in insights if "success" in i.title.lower()]
        assert len(success_threshold_insights) > 0
    
    def test_performance_insight_generation(self, analytics):
        """Test performance-related insight generation."""
        test_metrics = analytics.metrics_collector.get_all_metrics()
        
        # Create performance degradation trend
        trends = [
            TrendAnalysis(
                metric_name="quality.validation.rule_duration_ms",
                direction=TrendDirection.DECLINING,  # Getting slower (declining performance)
                confidence=0.85,
                slope=50,  # Increasing duration
                r_squared=0.8,
                time_window_hours=24,
                data_points=24,
                start_value=500,
                end_value=1500,
                average_value=1000,
                volatility=0.2,
                metadata={"percentage_change": 200}  # 200% slower
            )
        ]
        
        insights = analytics._generate_performance_insights(test_metrics, trends, 24)
        
        assert len(insights) >= 1
        
        perf_insights = [i for i in insights if i.insight_type == InsightType.PERFORMANCE]
        assert len(perf_insights) > 0
        
        # Should detect performance degradation
        degradation_insights = [i for i in perf_insights if "degradation" in i.title.lower()]
        assert len(degradation_insights) > 0
    
    def test_anomaly_insight_generation(self, analytics):
        """Test anomaly detection insight generation."""
        test_metrics = analytics.metrics_collector.get_all_metrics()
        
        # Create highly volatile trend
        trends = [
            TrendAnalysis(
                metric_name="quality.test.volatile_metric",
                direction=TrendDirection.VOLATILE,
                confidence=0.3,
                slope=0.1,
                r_squared=0.2,
                time_window_hours=24,
                data_points=24,
                start_value=100,
                end_value=105,
                average_value=102.5,
                volatility=0.8  # High volatility
            )
        ]
        
        insights = analytics._generate_anomaly_insights(test_metrics, trends, 24)
        
        assert len(insights) >= 1
        
        anomaly_insights = [i for i in insights if i.insight_type == InsightType.ANOMALY]
        assert len(anomaly_insights) > 0
        
        # Should detect high volatility
        volatility_insights = [i for i in anomaly_insights if "volatility" in i.description.lower()]
        assert len(volatility_insights) > 0
    
    def test_pattern_insight_generation(self, analytics):
        """Test pattern-based insight generation."""
        test_metrics = analytics.metrics_collector.get_all_metrics()
        
        # Create multiple declining trends
        trends = [
            TrendAnalysis(
                metric_name="quality.metric1",
                direction=TrendDirection.DECLINING,
                confidence=0.8,
                slope=-1,
                r_squared=0.7,
                time_window_hours=24,
                data_points=24,
                start_value=100,
                end_value=80,
                average_value=90,
                volatility=0.1
            ),
            TrendAnalysis(
                metric_name="quality.metric2", 
                direction=TrendDirection.DECLINING,
                confidence=0.75,
                slope=-0.8,
                r_squared=0.65,
                time_window_hours=24,
                data_points=24,
                start_value=90,
                end_value=75,
                average_value=82.5,
                volatility=0.12
            ),
            TrendAnalysis(
                metric_name="quality.metric3",
                direction=TrendDirection.DECLINING,
                confidence=0.7,
                slope=-1.2,
                r_squared=0.8,
                time_window_hours=24,
                data_points=24,
                start_value=110,
                end_value=85,
                average_value=97.5,
                volatility=0.15
            )
        ]
        
        insights = analytics._generate_pattern_insights(test_metrics, trends, 24)
        
        # Should detect pattern of multiple declining metrics
        pattern_insights = [i for i in insights if i.insight_type == InsightType.PATTERN]
        assert len(pattern_insights) > 0
        
        multiple_declining = [i for i in pattern_insights if "multiple" in i.title.lower() and "declining" in i.title.lower()]
        assert len(multiple_declining) > 0
    
    def test_quality_score_calculation(self, analytics):
        """Test overall quality score calculation."""
        test_metrics = analytics.metrics_collector.get_all_metrics()
        trends = []
        
        # Test with various metric combinations
        quality_score = analytics._calculate_overall_quality_score(test_metrics, trends, 24)
        
        assert quality_score is not None
        assert 0.0 <= quality_score <= 100.0
        
        # Test with no relevant metrics
        empty_metrics = {}
        empty_score = analytics._calculate_overall_quality_score(empty_metrics, trends, 24)
        assert empty_score is None
    
    def test_recommendation_generation(self, analytics):
        """Test top-level recommendation generation."""
        # Create test data with various quality issues
        trends = [
            TrendAnalysis(
                metric_name="quality.declining1",
                direction=TrendDirection.DECLINING,
                confidence=0.8,
                slope=-1,
                r_squared=0.7,
                time_window_hours=24,
                data_points=24,
                start_value=100,
                end_value=80,
                average_value=90,
                volatility=0.1
            ),
            TrendAnalysis(
                metric_name="quality.declining2",
                direction=TrendDirection.DECLINING,
                confidence=0.75,
                slope=-0.8,
                r_squared=0.6,
                time_window_hours=24,
                data_points=24,
                start_value=90,
                end_value=70,
                average_value=80,
                volatility=0.1
            )
        ]
        
        insights = [
            QualityInsight(
                insight_type=InsightType.THRESHOLD,
                severity="critical",
                title="Critical Quality Issue",
                description="Quality below critical threshold"
            ),
            QualityInsight(
                insight_type=InsightType.TREND,
                severity="warning", 
                title="Quality Declining",
                description="Quality trend is declining"
            )
        ]
        
        quality_score = 45.0  # Low quality score
        
        recommendations = analytics._generate_recommendations(trends, insights, quality_score)
        
        assert len(recommendations) > 0
        
        # Should recommend critical review for low quality score
        critical_recs = [r for r in recommendations if "critical" in r.lower()]
        assert len(critical_recs) > 0
        
        # Should recommend system review for multiple declining trends
        system_recs = [r for r in recommendations if "system" in r.lower()]
        assert len(system_recs) > 0
        
        # Should recommend immediate attention for critical issues
        immediate_recs = [r for r in recommendations if "immediate" in r.lower()]
        assert len(immediate_recs) > 0
    
    def test_analytics_result_export(self, analytics):
        """Test analytics result export functionality."""
        result = analytics.analyze_quality_trends()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            analytics.export_analysis(result, temp_path, format="json")
            
            # Verify export file exists
            assert temp_path.exists()
            assert temp_path.stat().st_size > 0
            
            # Verify export content
            with open(temp_path, 'r') as f:
                exported_data = json.load(f)
            
            assert 'analysis_timestamp' in exported_data
            assert 'time_window_hours' in exported_data
            assert 'quality_score' in exported_data
            assert 'summary_statistics' in exported_data
            assert 'recommendations' in exported_data
            assert 'trend_analyses' in exported_data
            assert 'quality_insights' in exported_data
            
            # Verify trend analyses structure
            assert len(exported_data['trend_analyses']) > 0
            trend = exported_data['trend_analyses'][0]
            assert 'metric_name' in trend
            assert 'direction' in trend
            assert 'confidence' in trend
            assert 'change_description' in trend
            
            # Verify insights structure
            assert len(exported_data['quality_insights']) > 0
            insight = exported_data['quality_insights'][0]
            assert 'insight_type' in insight
            assert 'severity' in insight
            assert 'title' in insight
            assert 'description' in insight
            
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_cached_analysis(self, analytics):
        """Test analysis result caching."""
        # First analysis
        result1 = analytics.analyze_quality_trends(time_window_hours=24)
        
        # Should be cached
        cached_result = analytics.get_cached_analysis(window_hours=24)
        assert cached_result is not None
        assert cached_result.analysis_timestamp == result1.analysis_timestamp
        
        # Different window should not be cached
        cached_different = analytics.get_cached_analysis(window_hours=12)
        assert cached_different is None
    
    def test_custom_insight_generator(self, analytics):
        """Test adding custom insight generators."""
        custom_insights = []
        
        def custom_generator(metrics, trends, window_hours):
            """Custom insight generator for testing."""
            custom_insights.append(QualityInsight(
                insight_type=InsightType.RECOMMENDATION,
                severity="info",
                title="Custom Insight",
                description="This is a custom generated insight",
                recommendations=["Custom recommendation"]
            ))
            return custom_insights
        
        # Add custom generator
        analytics.add_insight_generator(custom_generator)
        
        # Run analysis
        result = analytics.analyze_quality_trends()
        
        # Verify custom insight was generated
        custom_insight_found = any(
            insight.title == "Custom Insight" for insight in result.quality_insights
        )
        assert custom_insight_found
    
    def test_error_handling(self, analytics, mock_logger):
        """Test error handling in analytics."""
        # Test with invalid export format
        result = analytics.analyze_quality_trends()
        
        with tempfile.NamedTemporaryFile(suffix='.txt') as temp_file:
            with pytest.raises(ValueError):
                analytics.export_analysis(result, Path(temp_file.name), format="invalid")
        
        # Test analysis with broken insight generator
        def broken_generator(metrics, trends, window_hours):
            raise Exception("Broken generator")
        
        analytics.add_insight_generator(broken_generator)
        
        # Should still complete analysis despite broken generator
        result = analytics.analyze_quality_trends()
        assert isinstance(result, AnalyticsResult)
        
        # Should have logged the error
        mock_logger.error.assert_called()
    
    def test_insufficient_data_handling(self, analytics):
        """Test handling of insufficient data scenarios."""
        # Create metrics collector with insufficient data
        empty_collector = Mock(spec=QualityMetricsCollector)
        empty_collector.get_all_metrics.return_value = {}
        empty_collector.get_metrics_by_pattern.return_value = {}
        
        empty_analytics = QualityAnalytics(
            metrics_collector=empty_collector,
            logger=Mock()
        )
        
        # Should handle empty metrics gracefully
        result = empty_analytics.analyze_quality_trends()
        
        assert isinstance(result, AnalyticsResult)
        assert len(result.trend_analyses) == 0
        assert result.quality_score is None
        assert len(result.quality_insights) == 0


class TestTrendAnalysis:
    """Test suite for TrendAnalysis class."""
    
    def test_trend_analysis_properties(self):
        """Test TrendAnalysis properties and calculations."""
        trend = TrendAnalysis(
            metric_name="test.metric",
            direction=TrendDirection.IMPROVING,
            confidence=0.85,
            slope=2.5,
            r_squared=0.9,
            time_window_hours=24,
            data_points=24,
            start_value=60,
            end_value=95,
            average_value=77.5,
            volatility=0.15
        )
        
        # Test percentage change calculation
        assert abs(trend.percentage_change - 58.33) < 0.1  # (95-60)/60 * 100
        
        # Test change description
        description = trend.change_description
        assert "improving" in description.lower()
        assert "58.3%" in description or "58.3" in description
        
        # Test with zero start value
        trend_zero = TrendAnalysis(
            metric_name="test.zero",
            direction=TrendDirection.STABLE,
            confidence=0.5,
            slope=0,
            r_squared=0.1,
            time_window_hours=1,
            data_points=3,
            start_value=0,
            end_value=10,
            average_value=5,
            volatility=0.5
        )
        
        assert trend_zero.percentage_change == 0.0  # Should handle division by zero
    
    def test_trend_direction_descriptions(self):
        """Test trend direction descriptions."""
        # Test each direction
        directions = [
            (TrendDirection.IMPROVING, "improving"),
            (TrendDirection.DECLINING, "declining"),
            (TrendDirection.STABLE, "stable"),
            (TrendDirection.VOLATILE, "volatile"),
            (TrendDirection.INSUFFICIENT_DATA, "insufficient data")
        ]
        
        for direction, expected_word in directions:
            trend = TrendAnalysis(
                metric_name="test",
                direction=direction,
                confidence=0.5,
                slope=1.0,
                r_squared=0.5,
                time_window_hours=1,
                data_points=10,
                start_value=50,
                end_value=75,
                average_value=62.5,
                volatility=0.2 if direction != TrendDirection.VOLATILE else 0.8
            )
            
            description = trend.change_description
            assert expected_word in description.lower()


class TestQualityInsight:
    """Test suite for QualityInsight class."""
    
    def test_quality_insight_creation(self):
        """Test QualityInsight creation and properties."""
        timestamp = time.time()
        
        insight = QualityInsight(
            insight_type=InsightType.TREND,
            severity="warning",
            title="Test Insight",
            description="This is a test insight",
            metric_name="test.metric",
            current_value=42.5,
            threshold_value=50.0,
            confidence=0.85,
            timestamp=timestamp,
            recommendations=["Fix this", "Do that"],
            related_metrics=["metric1", "metric2"],
            metadata={"source": "test"}
        )
        
        assert insight.insight_type == InsightType.TREND
        assert insight.severity == "warning"
        assert insight.title == "Test Insight"
        assert insight.description == "This is a test insight"
        assert insight.metric_name == "test.metric"
        assert insight.current_value == 42.5
        assert insight.threshold_value == 50.0
        assert insight.confidence == 0.85
        assert insight.timestamp == timestamp
        assert len(insight.recommendations) == 2
        assert len(insight.related_metrics) == 2
        assert insight.metadata["source"] == "test"
        
        # Test datetime property
        dt = insight.datetime
        assert dt.timestamp() == timestamp


class TestAnalyticsResult:
    """Test suite for AnalyticsResult class."""
    
    def test_analytics_result_creation(self):
        """Test AnalyticsResult creation and properties."""
        timestamp = time.time()
        
        trends = [
            TrendAnalysis(
                metric_name="test1",
                direction=TrendDirection.IMPROVING,
                confidence=0.8,
                slope=1.0,
                r_squared=0.7,
                time_window_hours=24,
                data_points=24,
                start_value=50,
                end_value=75,
                average_value=62.5,
                volatility=0.1
            )
        ]
        
        insights = [
            QualityInsight(
                insight_type=InsightType.TREND,
                severity="info",
                title="Test Insight",
                description="Test description"
            )
        ]
        
        result = AnalyticsResult(
            analysis_timestamp=timestamp,
            time_window_hours=24,
            trend_analyses=trends,
            quality_insights=insights,
            summary_statistics={"test": "stat"},
            quality_score=85.5,
            recommendations=["Recommendation 1", "Recommendation 2"]
        )
        
        assert result.analysis_timestamp == timestamp
        assert result.time_window_hours == 24
        assert len(result.trend_analyses) == 1
        assert len(result.quality_insights) == 1
        assert result.summary_statistics["test"] == "stat"
        assert result.quality_score == 85.5
        assert len(result.recommendations) == 2
        
        # Test datetime property
        dt = result.datetime
        assert dt.timestamp() == timestamp


if __name__ == "__main__":
    pytest.main([__file__])