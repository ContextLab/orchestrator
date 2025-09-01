"""
Comprehensive tests for quality dashboard system.

Tests all aspects of the QualityDashboard including:
- Dashboard configuration and widget management
- Data visualization and chart generation
- Web server functionality and API endpoints
- Dashboard export/import capabilities
- Real-time data updates and caching
"""

import pytest
import time
import tempfile
import json
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any

# Import the system under test
from src.orchestrator.quality.reporting.dashboard import (
    QualityDashboard,
    DashboardWidget,
    WidgetType,
    DashboardConfig
)
from src.orchestrator.quality.reporting.metrics import (
    QualityMetricsCollector,
    TimeSeriesMetric,
    MetricType
)
from src.orchestrator.quality.reporting.analytics import (
    QualityAnalytics,
    AnalyticsResult,
    QualityInsight,
    TrendAnalysis,
    TrendDirection,
    InsightType
)
from src.orchestrator.quality.reporting.alerts import (
    QualityAlertSystem,
    AlertNotification,
    AlertSeverity,
    AlertType
)


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
    
    # Quality score metrics
    quality_metric = TimeSeriesMetric("quality.overall.score", MetricType.GAUGE)
    base_time = time.time() - (24 * 3600)
    for i in range(24):
        timestamp = base_time + (i * 3600)
        value = 60 + (i * 1.5)  # Improving score
        quality_metric.add_value(value, timestamp)
    test_metrics["quality.overall.score"] = quality_metric
    
    # Violations metrics
    violations_metric = TimeSeriesMetric("quality.validation.total_violations", MetricType.GAUGE)
    for i in range(24):
        timestamp = base_time + (i * 3600)
        value = max(0, 20 - i)  # Decreasing violations
        violations_metric.add_value(value, timestamp)
    test_metrics["quality.validation.total_violations"] = violations_metric
    
    # Success rate metrics
    success_metric = TimeSeriesMetric("quality.validation.success_rate", MetricType.GAUGE)
    for i in range(24):
        timestamp = base_time + (i * 3600)
        value = 95 + (i % 3) - 1  # Stable around 95%
        success_metric.add_value(value, timestamp)
    test_metrics["quality.validation.success_rate"] = success_metric
    
    # Rule performance metrics
    for rule_id in ["rule_1", "rule_2", "rule_3"]:
        rule_metric = TimeSeriesMetric(
            "quality.validation.rule_duration_ms", 
            MetricType.TIMER, 
            {"rule_id": rule_id}
        )
        for i in range(24):
            timestamp = base_time + (i * 3600)
            value = 100 + (hash(rule_id) % 500) + (i % 10) * 20
            rule_metric.add_value(value, timestamp)
        test_metrics[f"quality.validation.rule_duration_ms{{rule_id={rule_id}}}"] = rule_metric
    
    collector.get_all_metrics.return_value = test_metrics
    collector.get_metrics_by_pattern.side_effect = lambda pattern: {
        k: v for k, v in test_metrics.items() 
        if pattern == "*" or any(p in k for p in pattern.split("*"))
    }
    
    return collector


@pytest.fixture
def mock_analytics():
    """Create mock analytics engine with test results."""
    analytics = Mock(spec=QualityAnalytics)
    
    # Create sample analytics result
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
            direction=TrendDirection.IMPROVING,
            confidence=0.8,
            slope=-0.8,
            r_squared=0.75,
            time_window_hours=24,
            data_points=24,
            start_value=20,
            end_value=5,
            average_value=12.5,
            volatility=0.2
        )
    ]
    
    insights = [
        QualityInsight(
            insight_type=InsightType.TREND,
            severity="info",
            title="Quality Score Improving",
            description="Quality score has improved by 58% over the last 24 hours",
            confidence=0.9,
            recommendations=["Continue current practices", "Monitor for sustained improvement"]
        ),
        QualityInsight(
            insight_type=InsightType.THRESHOLD,
            severity="warning",
            title="Success Rate Below Target",
            description="Validation success rate is below the target threshold",
            confidence=0.8,
            recommendations=["Review validation rules", "Check for data quality issues"]
        )
    ]
    
    result = AnalyticsResult(
        analysis_timestamp=time.time(),
        time_window_hours=24,
        trend_analyses=trends,
        quality_insights=insights,
        summary_statistics={"total_metrics": 10, "trends_detected": 2},
        quality_score=82.5,
        recommendations=["Maintain current quality practices", "Address warning-level issues"]
    )
    
    analytics.analyze_quality_trends.return_value = result
    return analytics


@pytest.fixture
def mock_alert_system():
    """Create mock alert system with test alerts."""
    alert_system = Mock(spec=QualityAlertSystem)
    
    # Create sample active alerts
    alerts = [
        AlertNotification(
            alert_id="alert_1",
            rule_id="quality_score_warning",
            timestamp=time.time() - 1800,  # 30 minutes ago
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.THRESHOLD,
            title="Quality Score Below Warning Threshold",
            message="Quality score has dropped below warning level",
            metric_name="quality.overall.score",
            current_value=68.0,
            threshold_value=75.0
        ),
        AlertNotification(
            alert_id="alert_2",
            rule_id="success_rate_critical",
            timestamp=time.time() - 3600,  # 1 hour ago
            severity=AlertSeverity.CRITICAL,
            alert_type=AlertType.THRESHOLD,
            title="Critical Success Rate",
            message="Success rate has fallen to critical levels",
            metric_name="quality.validation.success_rate",
            current_value=65.0,
            threshold_value=75.0
        )
    ]
    
    alert_system.get_active_alerts.return_value = alerts
    alert_system.get_alert_statistics.return_value = {
        "total_alerts": 15,
        "active_alerts": 2,
        "alerts_by_severity": {"critical": 1, "warning": 1, "error": 0, "info": 0},
        "alerts_by_type": {"threshold": 2, "trend": 0, "anomaly": 0}
    }
    
    return alert_system


@pytest.fixture
def dashboard(mock_metrics_collector, mock_analytics, mock_alert_system, mock_logger):
    """Create QualityDashboard instance for testing."""
    return QualityDashboard(
        metrics_collector=mock_metrics_collector,
        analytics=mock_analytics,
        alert_system=mock_alert_system,
        logger=mock_logger,
        port=8888,  # Use different port for testing
        host="localhost"
    )


class TestQualityDashboard:
    """Test suite for QualityDashboard."""
    
    def test_initialization(self, mock_metrics_collector, mock_analytics, mock_alert_system, mock_logger):
        """Test proper initialization of dashboard."""
        dashboard = QualityDashboard(
            metrics_collector=mock_metrics_collector,
            analytics=mock_analytics,
            alert_system=mock_alert_system,
            logger=mock_logger,
            port=8080,
            host="0.0.0.0"
        )
        
        assert dashboard.metrics_collector == mock_metrics_collector
        assert dashboard.analytics == mock_analytics
        assert dashboard.alert_system == mock_alert_system
        assert dashboard.logger == mock_logger
        assert dashboard.port == 8080
        assert dashboard.host == "0.0.0.0"
        
        # Should create default dashboard
        assert len(dashboard._dashboards) == 1
        assert "default" in dashboard._dashboards
        
        # Verify default dashboard structure
        default_dashboard = dashboard._dashboards["default"]
        assert default_dashboard.title == "Quality Control Dashboard"
        assert len(default_dashboard.widgets) > 0
    
    def test_dashboard_management(self, dashboard):
        """Test dashboard configuration management."""
        # Create custom dashboard
        custom_widgets = [
            DashboardWidget(
                widget_id="custom_gauge",
                title="Custom Gauge",
                widget_type=WidgetType.GAUGE,
                metric_patterns=["custom.metric.*"],
                position=(0, 0),
                size=(1, 2)
            )
        ]
        
        custom_dashboard = DashboardConfig(
            dashboard_id="custom",
            title="Custom Dashboard",
            description="Custom test dashboard",
            widgets=custom_widgets,
            layout_columns=8,
            auto_refresh=False,
            refresh_interval_seconds=60
        )
        
        # Add dashboard
        dashboard.add_dashboard(custom_dashboard)
        assert len(dashboard._dashboards) == 2
        assert "custom" in dashboard._dashboards
        
        # Get dashboard config
        retrieved_config = dashboard.get_dashboard_config("custom")
        assert retrieved_config is not None
        assert retrieved_config.title == "Custom Dashboard"
        assert len(retrieved_config.widgets) == 1
        
        # List dashboards
        dashboard_list = dashboard.list_dashboards()
        assert len(dashboard_list) == 2
        
        custom_info = next(d for d in dashboard_list if d['dashboard_id'] == 'custom')
        assert custom_info['title'] == "Custom Dashboard"
        assert custom_info['widget_count'] == 1
        assert custom_info['auto_refresh'] is False
        
        # Remove dashboard
        removed = dashboard.remove_dashboard("custom")
        assert removed is True
        assert len(dashboard._dashboards) == 1
        assert "custom" not in dashboard._dashboards
        
        # Test removing non-existent dashboard
        removed = dashboard.remove_dashboard("non_existent")
        assert removed is False
    
    def test_widget_data_generation_gauge(self, dashboard):
        """Test gauge widget data generation."""
        gauge_widget = DashboardWidget(
            widget_id="test_gauge",
            title="Test Gauge",
            widget_type=WidgetType.GAUGE,
            metric_patterns=["quality.overall.score"],
            config={
                "min_value": 0,
                "max_value": 100,
                "thresholds": [
                    {"value": 60, "color": "red"},
                    {"value": 80, "color": "yellow"},
                    {"value": 90, "color": "green"}
                ]
            }
        )
        
        data = dashboard._generate_widget_data(gauge_widget)
        
        assert data["type"] == "gauge"
        assert "value" in data
        assert data["min_value"] == 0
        assert data["max_value"] == 100
        assert len(data["thresholds"]) == 3
        assert "timestamp" in data
        
        # Value should be from the test metric (latest value around 95)
        assert 90 <= data["value"] <= 100
    
    def test_widget_data_generation_line_chart(self, dashboard):
        """Test line chart widget data generation."""
        chart_widget = DashboardWidget(
            widget_id="test_chart",
            title="Test Line Chart",
            widget_type=WidgetType.LINE_CHART,
            metric_patterns=["quality.validation.success_rate"],
            time_window_hours=12,
            config={"smooth": True, "show_points": False}
        )
        
        data = dashboard._generate_widget_data(chart_widget)
        
        assert data["type"] == "line"
        assert "datasets" in data
        assert len(data["datasets"]) >= 1
        
        dataset = data["datasets"][0]
        assert "label" in dataset
        assert "data" in dataset
        assert len(dataset["data"]) > 0
        
        # Verify data point format
        data_point = dataset["data"][0]
        assert "x" in data_point  # Timestamp in milliseconds
        assert "y" in data_point  # Value
        
        assert "options" in data
        assert data["options"]["smooth"] is True
    
    def test_widget_data_generation_bar_chart(self, dashboard):
        """Test bar chart widget data generation."""
        bar_widget = DashboardWidget(
            widget_id="test_bar",
            title="Test Bar Chart",
            widget_type=WidgetType.BAR_CHART,
            metric_patterns=["quality.validation.rule_duration_ms"],
            config={"orientation": "horizontal", "limit": 5}
        )
        
        data = dashboard._generate_widget_data(bar_widget)
        
        assert data["type"] == "bar"
        assert "data" in data
        assert "labels" in data["data"]
        assert "datasets" in data["data"]
        
        # Should have limited results
        assert len(data["data"]["labels"]) <= 5
        assert len(data["data"]["datasets"][0]["data"]) <= 5
        
        assert "options" in data
        assert data["options"]["indexAxis"] == "y"  # Horizontal orientation
    
    def test_widget_data_generation_alert_list(self, dashboard):
        """Test alert list widget data generation."""
        alert_widget = DashboardWidget(
            widget_id="test_alerts",
            title="Test Alerts",
            widget_type=WidgetType.ALERT_LIST,
            config={"max_alerts": 5}
        )
        
        data = dashboard._generate_widget_data(alert_widget)
        
        assert data["type"] == "alert_list"
        assert "alerts" in data
        assert "total_active" in data
        
        # Should have alerts from mock
        assert len(data["alerts"]) == 2  # From mock_alert_system
        alert = data["alerts"][0]
        assert "id" in alert
        assert "title" in alert
        assert "severity" in alert
        assert "timestamp" in alert
        assert "age_minutes" in alert
    
    def test_widget_data_generation_insight_panel(self, dashboard):
        """Test insight panel widget data generation."""
        insight_widget = DashboardWidget(
            widget_id="test_insights",
            title="Test Insights",
            widget_type=WidgetType.INSIGHT_PANEL,
            config={"max_insights": 3}
        )
        
        data = dashboard._generate_widget_data(insight_widget)
        
        assert data["type"] == "insight_panel"
        assert "insights" in data
        assert "quality_score" in data
        assert "recommendations" in data
        
        # Should have insights from mock analytics
        assert len(data["insights"]) >= 1
        insight = data["insights"][0]
        assert "type" in insight
        assert "severity" in insight
        assert "title" in insight
        assert "description" in insight
        assert "recommendations" in insight
        
        # Quality score should be from mock analytics (82.5)
        assert data["quality_score"] == 82.5
    
    def test_widget_data_generation_counter(self, dashboard):
        """Test counter widget data generation."""
        counter_widget = DashboardWidget(
            widget_id="test_counter",
            title="Test Counter",
            widget_type=WidgetType.COUNTER,
            metric_patterns=["quality.validation.total_violations"],
            config={"format": "number"}
        )
        
        data = dashboard._generate_widget_data(counter_widget)
        
        assert data["type"] == "counter"
        assert "value" in data
        assert data["format"] == "number"
        assert "timestamp" in data
        
        # Value should be sum of matching metrics
        assert isinstance(data["value"], (int, float))
    
    def test_widget_data_generation_trend_indicator(self, dashboard):
        """Test trend indicator widget data generation."""
        trend_widget = DashboardWidget(
            widget_id="test_trends",
            title="Test Trends", 
            widget_type=WidgetType.TREND_INDICATOR,
            metric_patterns=["quality.*"],
            time_window_hours=24
        )
        
        data = dashboard._generate_widget_data(trend_widget)
        
        assert data["type"] == "trend_indicator"
        assert "trends" in data
        
        # Should have trends from mock analytics
        assert len(data["trends"]) >= 1
        trend = data["trends"][0]
        assert "metric_name" in trend
        assert "direction" in trend
        assert "change_percent" in trend
        assert "confidence" in trend
        assert "description" in trend
    
    def test_widget_data_generation_table(self, dashboard):
        """Test table widget data generation."""
        table_widget = DashboardWidget(
            widget_id="test_table",
            title="Test Table",
            widget_type=WidgetType.TABLE,
            metric_patterns=["quality.validation.*"],
            time_window_hours=24
        )
        
        data = dashboard._generate_widget_data(table_widget)
        
        assert data["type"] == "table"
        assert "columns" in data
        assert "rows" in data
        
        # Should have standard columns
        expected_columns = ["metric_name", "current_value", "mean", "min", "max", "change"]
        assert data["columns"] == expected_columns
        
        # Should have rows with data
        if len(data["rows"]) > 0:
            row = data["rows"][0]
            assert "metric_name" in row
            assert "current_value" in row
            assert isinstance(row["current_value"], (int, float))
    
    def test_widget_data_caching(self, dashboard):
        """Test widget data caching mechanism."""
        gauge_widget = DashboardWidget(
            widget_id="cached_gauge",
            title="Cached Gauge",
            widget_type=WidgetType.GAUGE,
            metric_patterns=["quality.overall.score"]
        )
        
        # Add widget to default dashboard
        dashboard._dashboards["default"].widgets.append(gauge_widget)
        
        # First call should generate data
        data1 = dashboard.get_widget_data("default", "cached_gauge")
        assert data1 is not None
        
        # Second call should use cache
        data2 = dashboard.get_widget_data("default", "cached_gauge")
        assert data2 is not None
        assert data1["timestamp"] == data2["timestamp"]  # Should be same cached data
        
        # Clear cache by waiting for TTL (not practical for tests) or manually
        dashboard._data_cache.clear()
        
        # Third call should generate new data
        data3 = dashboard.get_widget_data("default", "cached_gauge")
        assert data3 is not None
        # Timestamp might be different due to cache refresh
    
    def test_complete_dashboard_data_retrieval(self, dashboard):
        """Test retrieving complete dashboard data."""
        dashboard_data = dashboard.get_dashboard_data("default")
        
        assert dashboard_data is not None
        assert dashboard_data["dashboard_id"] == "default"
        assert dashboard_data["title"] == "Quality Control Dashboard"
        assert "widgets" in dashboard_data
        assert "widget_data" in dashboard_data
        assert "timestamp" in dashboard_data
        
        # Should have widget data for enabled widgets
        assert len(dashboard_data["widget_data"]) > 0
        
        # Verify widget structure
        widgets = dashboard_data["widgets"]
        assert len(widgets) > 0
        widget = widgets[0]
        assert "widget_id" in widget
        assert "title" in widget
        assert "widget_type" in widget
        
        # Test non-existent dashboard
        non_existent = dashboard.get_dashboard_data("non_existent")
        assert non_existent is None
    
    def test_metric_color_generation(self, dashboard):
        """Test consistent metric color generation."""
        # Test color generation
        color1 = dashboard._get_metric_color("test.metric")
        color2 = dashboard._get_metric_color("test.metric")  # Should be identical
        color3 = dashboard._get_metric_color("different.metric")  # Should be different
        
        # Colors should be consistent for same metric
        assert color1 == color2
        assert color1 != color3
        
        # Should be valid RGB format
        assert color1.startswith("rgb(")
        assert color1.endswith(")")
        
        # Test with alpha
        rgba_color = dashboard._get_metric_color("test.metric", alpha=0.5)
        assert rgba_color.startswith("rgba(")
        assert "0.5" in rgba_color
    
    def test_pattern_matching(self, dashboard):
        """Test glob pattern matching utility."""
        assert dashboard._pattern_matches("quality.overall.score", "quality.*") is True
        assert dashboard._pattern_matches("quality.overall.score", "quality.overall.*") is True
        assert dashboard._pattern_matches("quality.overall.score", "performance.*") is False
        assert dashboard._pattern_matches("test.metric", "*") is True
        assert dashboard._pattern_matches("test.metric", "test.metric") is True
    
    def test_dashboard_export_import(self, dashboard):
        """Test dashboard configuration export and import."""
        # Export default dashboard
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            success = dashboard.export_dashboard_config("default", temp_path)
            assert success is True
            assert temp_path.exists()
            
            # Verify exported content
            with open(temp_path, 'r') as f:
                exported_config = json.load(f)
            
            assert exported_config["dashboard_id"] == "default"
            assert exported_config["title"] == "Quality Control Dashboard"
            assert "widgets" in exported_config
            assert len(exported_config["widgets"]) > 0
            
            # Test import
            exported_config["dashboard_id"] = "imported"
            exported_config["title"] = "Imported Dashboard"
            
            with open(temp_path, 'w') as f:
                json.dump(exported_config, f)
            
            success = dashboard.import_dashboard_config(temp_path)
            assert success is True
            assert "imported" in dashboard._dashboards
            
            imported_dashboard = dashboard.get_dashboard_config("imported")
            assert imported_dashboard.title == "Imported Dashboard"
            
        finally:
            temp_path.unlink()
        
        # Test export non-existent dashboard
        with tempfile.NamedTemporaryFile(suffix='.json') as temp_file:
            success = dashboard.export_dashboard_config("non_existent", Path(temp_file.name))
            assert success is False
    
    def test_unsupported_widget_type(self, dashboard):
        """Test handling of unsupported widget types."""
        # Create widget with invalid type (simulate future widget type)
        invalid_widget = DashboardWidget(
            widget_id="invalid_widget",
            title="Invalid Widget",
            widget_type="future_widget_type",  # Not in WidgetType enum
            metric_patterns=["test.*"]
        )
        
        # Should handle gracefully and return error
        with patch.object(invalid_widget, 'widget_type', "future_widget_type"):
            data = dashboard._generate_widget_data(invalid_widget)
            assert "error" in data
            assert "unsupported" in data["error"].lower()
    
    def test_error_handling_in_data_generation(self, dashboard, mock_logger):
        """Test error handling in widget data generation."""
        # Create widget that will cause an error
        error_widget = DashboardWidget(
            widget_id="error_widget",
            title="Error Widget",
            widget_type=WidgetType.GAUGE,
            metric_patterns=["error.metric.*"]
        )
        
        # Make metrics collector raise an exception
        dashboard.metrics_collector.get_metrics_by_pattern.side_effect = Exception("Test error")
        
        # Should handle error gracefully
        data = dashboard.get_widget_data("default", "error_widget")
        assert data is None  # Should return None for errors
        
        # Should have logged the error
        mock_logger.error.assert_called()
    
    def test_disabled_widget_handling(self, dashboard):
        """Test handling of disabled widgets."""
        # Create disabled widget
        disabled_widget = DashboardWidget(
            widget_id="disabled_widget",
            title="Disabled Widget",
            widget_type=WidgetType.COUNTER,
            metric_patterns=["test.*"],
            enabled=False  # Disabled
        )
        
        # Add to dashboard
        dashboard._dashboards["default"].widgets.append(disabled_widget)
        
        # Should not generate data for disabled widget
        data = dashboard.get_widget_data("default", "disabled_widget")
        assert data is None
        
        # Should not include in dashboard data
        dashboard_data = dashboard.get_dashboard_data("default")
        assert "disabled_widget" not in dashboard_data["widget_data"]


class TestDashboardWidget:
    """Test suite for DashboardWidget class."""
    
    def test_widget_creation(self):
        """Test DashboardWidget creation and properties."""
        widget = DashboardWidget(
            widget_id="test_widget",
            title="Test Widget Title",
            widget_type=WidgetType.LINE_CHART,
            metric_patterns=["quality.*", "performance.*"],
            time_window_hours=48,
            refresh_interval_seconds=15,
            position=(2, 3),
            size=(2, 4),
            config={"theme": "dark", "smooth": True},
            enabled=True
        )
        
        assert widget.widget_id == "test_widget"
        assert widget.title == "Test Widget Title"
        assert widget.widget_type == WidgetType.LINE_CHART
        assert len(widget.metric_patterns) == 2
        assert "quality.*" in widget.metric_patterns
        assert "performance.*" in widget.metric_patterns
        assert widget.time_window_hours == 48
        assert widget.refresh_interval_seconds == 15
        assert widget.position == (2, 3)
        assert widget.size == (2, 4)
        assert widget.config["theme"] == "dark"
        assert widget.config["smooth"] is True
        assert widget.enabled is True


class TestDashboardConfig:
    """Test suite for DashboardConfig class."""
    
    def test_config_creation(self):
        """Test DashboardConfig creation and properties."""
        widgets = [
            DashboardWidget("widget1", "Widget 1", WidgetType.GAUGE),
            DashboardWidget("widget2", "Widget 2", WidgetType.LINE_CHART)
        ]
        
        config = DashboardConfig(
            dashboard_id="test_dashboard",
            title="Test Dashboard",
            description="A test dashboard configuration",
            widgets=widgets,
            layout_columns=8,
            auto_refresh=False,
            refresh_interval_seconds=45,
            theme="dark",
            metadata={"author": "test_user", "version": "1.0"}
        )
        
        assert config.dashboard_id == "test_dashboard"
        assert config.title == "Test Dashboard"
        assert config.description == "A test dashboard configuration"
        assert len(config.widgets) == 2
        assert config.widgets[0].widget_id == "widget1"
        assert config.widgets[1].widget_id == "widget2"
        assert config.layout_columns == 8
        assert config.auto_refresh is False
        assert config.refresh_interval_seconds == 45
        assert config.theme == "dark"
        assert config.metadata["author"] == "test_user"
        assert config.metadata["version"] == "1.0"


class TestWebServerIntegration:
    """Test suite for web server functionality."""
    
    def test_server_lifecycle(self, dashboard):
        """Test web server start and stop."""
        # Server should not be running initially
        assert dashboard._is_running is False
        assert dashboard._server is None
        
        # Start server
        dashboard.start_server()
        
        # Give server time to start
        time.sleep(0.1)
        
        try:
            assert dashboard._is_running is True
            assert dashboard._server is not None
            assert dashboard._server_thread is not None
            
            # Test starting already running server
            dashboard.start_server()  # Should handle gracefully
            
        finally:
            # Stop server
            dashboard.stop_server()
            
            # Give server time to stop
            time.sleep(0.1)
            
            assert dashboard._is_running is False
    
    @patch('src.orchestrator.quality.reporting.dashboard.HTTPServer')
    def test_server_creation_error(self, mock_http_server, dashboard, mock_logger):
        """Test server creation error handling."""
        # Make HTTPServer raise an exception
        mock_http_server.side_effect = Exception("Port already in use")
        
        # Should handle error gracefully
        dashboard.start_server()
        
        # Should have logged error
        mock_logger.error.assert_called()
        assert dashboard._is_running is False
    
    def test_request_handler_creation(self, dashboard):
        """Test HTTP request handler creation."""
        handler_class = dashboard._create_request_handler()
        
        # Should create a class
        assert handler_class is not None
        assert callable(handler_class)
        
        # Should be a proper HTTP handler class
        assert hasattr(handler_class, 'do_GET')
        assert hasattr(handler_class, 'do_POST')


if __name__ == "__main__":
    pytest.main([__file__])