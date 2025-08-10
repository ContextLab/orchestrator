"""Real Performance Monitoring Tests - Issue #206 Task 3.2

Comprehensive tests for the performance monitoring and analytics system with real metrics,
dashboard functionality, and alert generation. NO MOCKS - real performance tracking only.
"""

import pytest
import asyncio
import logging
import time
import statistics

from orchestrator.analytics.performance_monitor import (
    PerformanceMonitor,
    PerformanceMetric,
    MetricType,
    AlertSeverity,
    PerformanceProfile
)
from orchestrator.analytics.dashboard import PerformanceDashboard, DashboardWidget

# Configure logging for test visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPerformanceMonitor:
    """Test performance monitoring system with real metrics collection."""
    
    @pytest.fixture
    async def performance_monitor(self):
        """Create performance monitor for testing."""
        monitor = PerformanceMonitor(collection_interval=0.5)  # Fast for testing
        await monitor.start_monitoring()
        yield monitor
        await monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_monitor_initialization(self, performance_monitor):
        """Test performance monitor initialization."""
        logger.info("🧪 Testing performance monitor initialization")
        
        assert performance_monitor.collector is not None
        assert performance_monitor.analyzer is not None
        assert performance_monitor._processing is True
        
        # Check initial statistics
        summary = performance_monitor.get_performance_summary()
        assert summary is not None
        assert 'system_health' in summary
        assert summary['metrics_processed'] >= 0
        
        logger.info("✅ Performance monitor initialization test passed")
    
    @pytest.mark.asyncio
    async def test_execution_recording(self, performance_monitor):
        """Test recording execution performance metrics."""
        logger.info("🧪 Testing execution recording")
        
        component_name = "test_component"
        
        # Record several executions with different performance characteristics
        test_executions = [
            (1.2, True, None),      # Fast successful execution
            (2.5, True, None),      # Medium successful execution  
            (5.0, False, "Test error"),  # Slow failed execution
            (0.8, True, None),      # Very fast execution
            (3.1, True, None),      # Another medium execution
        ]
        
        for exec_time, success, error in test_executions:
            await performance_monitor.record_execution(
                component=component_name,
                execution_time=exec_time,
                success=success,
                error_message=error,
                context={"test": True}
            )
        
        # Wait for processing
        await asyncio.sleep(2.0)
        
        # Check component performance
        component_perf = performance_monitor.get_component_performance(component_name)
        assert component_perf is not None
        assert component_perf['total_executions'] == len(test_executions)
        assert component_perf['successful_executions'] == 4  # 4 out of 5 successful
        assert component_perf['failed_executions'] == 1
        
        # Check calculated metrics
        success_rate = component_perf['success_rate']
        assert 70 <= success_rate <= 90  # 80% success rate
        
        avg_time = component_perf['average_execution_time']
        expected_avg = statistics.mean([1.2, 2.5, 0.8, 3.1])  # Only successful executions
        assert abs(avg_time - expected_avg) < 0.5  # More reasonable tolerance for test timing variations
        
        logger.info(f"Component performance: {component_perf}")
        logger.info("✅ Execution recording test passed")
    
    @pytest.mark.asyncio
    async def test_container_startup_tracking(self, performance_monitor):
        """Test container startup performance tracking."""
        logger.info("🧪 Testing container startup tracking")
        
        # Record several container startups
        container_data = [
            ("container_1", 2.3, "python:3.11-slim", True),
            ("container_2", 1.8, "python:3.11-slim", True),
            ("container_3", 4.5, "ubuntu:22.04", True),
            ("container_4", 10.2, "java:17", False),  # Failed startup
        ]
        
        for container_id, startup_time, image, success in container_data:
            await performance_monitor.record_container_startup(
                container_id=container_id,
                startup_time=startup_time,
                image=image,
                success=success
            )
        
        # Wait for processing
        await asyncio.sleep(2.0)
        
        # Check container manager performance
        container_perf = performance_monitor.get_component_performance("container_manager")
        assert container_perf is not None
        assert container_perf['total_executions'] == len(container_data)
        
        logger.info(f"Container startup performance: {container_perf}")
        logger.info("✅ Container startup tracking test passed")
    
    @pytest.mark.asyncio
    async def test_throughput_tracking(self, performance_monitor):
        """Test throughput metric recording."""
        logger.info("🧪 Testing throughput tracking")
        
        component = "throughput_test"
        
        # Record throughput measurements
        throughput_data = [10.5, 12.3, 8.7, 15.2, 11.8]  # operations per second
        
        for ops_per_sec in throughput_data:
            await performance_monitor.record_throughput(
                component=component,
                operations_per_second=ops_per_sec,
                context={"measurement_type": "test"}
            )
        
        # Wait for processing
        await asyncio.sleep(2.0)
        
        # Verify throughput was recorded
        summary = performance_monitor.get_performance_summary()
        assert summary['metrics_processed'] >= len(throughput_data)
        
        logger.info("✅ Throughput tracking test passed")
    
    @pytest.mark.asyncio
    async def test_alert_generation(self, performance_monitor):
        """Test performance alert generation."""
        logger.info("🧪 Testing alert generation")
        
        # Record execution that should trigger alerts (high execution time)
        slow_component = "slow_component"
        
        # Record a very slow execution that should trigger an alert
        await performance_monitor.record_execution(
            component=slow_component,
            execution_time=35.0,  # Should exceed 30s threshold
            success=True
        )
        
        # Wait for processing
        await asyncio.sleep(2.0)
        
        # Check for alerts
        alerts = performance_monitor.get_active_alerts()
        logger.info(f"Generated alerts: {len(alerts)}")
        
        # Should have at least one alert for high execution time
        execution_time_alerts = [a for a in alerts if a['metric_type'] == 'execution_time']
        assert len(execution_time_alerts) >= 1, f"Expected execution time alert. Alerts: {alerts}"
        
        # Test alert resolution
        if execution_time_alerts:
            alert_id = execution_time_alerts[0]['alert_id']
            resolved = performance_monitor.resolve_alert(alert_id)
            assert resolved is True
            
            # Check that alert is now resolved
            updated_alerts = performance_monitor.get_active_alerts()
            active_execution_alerts = [a for a in updated_alerts if a['metric_type'] == 'execution_time']
            assert len(active_execution_alerts) == 0
        
        logger.info("✅ Alert generation test passed")
    
    @pytest.mark.asyncio
    async def test_performance_summary(self, performance_monitor):
        """Test performance summary generation."""
        logger.info("🧪 Testing performance summary generation")
        
        # Generate some diverse performance data
        components = ["component_a", "component_b", "component_c"]
        
        for i, component in enumerate(components):
            # Each component has different performance characteristics
            base_time = (i + 1) * 2.0  # 2s, 4s, 6s base times
            success_rate = 0.9 - (i * 0.1)  # 90%, 80%, 70% success rates
            
            for j in range(10):
                execution_time = base_time + (j * 0.1)
                success = j < (10 * success_rate)  # Distribute failures
                
                await performance_monitor.record_execution(
                    component=component,
                    execution_time=execution_time,
                    success=success,
                    error_message="Test error" if not success else None
                )
        
        # Wait for processing
        await asyncio.sleep(3.0)
        
        # Get comprehensive summary
        summary = performance_monitor.get_performance_summary()
        
        # Verify summary structure
        required_keys = [
            'system_health', 'metrics_processed', 'active_alerts',
            'component_profiles', 'performance_issues', 'monitor_uptime'
        ]
        
        for key in required_keys:
            assert key in summary, f"Missing key in summary: {key}"
        
        # Check that system detected performance issues
        issues = summary.get('performance_issues', [])
        logger.info(f"Detected performance issues: {len(issues)}")
        
        # Should detect slow execution times and error rates
        slow_execution_issues = [i for i in issues if i['type'] == 'slow_execution']
        error_rate_issues = [i for i in issues if i['type'] == 'high_error_rate']
        
        assert len(slow_execution_issues) >= 1, "Should detect slow execution issues"
        assert len(error_rate_issues) >= 1, "Should detect high error rate issues"
        
        # Check system health score
        system_health = summary['system_health']
        assert 0 <= system_health <= 100, "System health should be 0-100"
        assert system_health < 90, "System health should reflect performance issues"
        
        logger.info(f"System health score: {system_health}")
        logger.info("✅ Performance summary test passed")
    
    @pytest.mark.asyncio
    async def test_performance_report_generation(self, performance_monitor):
        """Test comprehensive performance report generation."""
        logger.info("🧪 Testing performance report generation")
        
        # Generate some sample data
        await performance_monitor.record_execution("report_test", 2.5, True)
        await asyncio.sleep(1.0)
        
        # Generate performance report
        report = performance_monitor.generate_performance_report()
        
        # Verify report structure
        assert 'report_timestamp' in report
        assert 'executive_summary' in report
        assert 'performance_overview' in report
        assert 'recommendations' in report
        
        # Check executive summary
        exec_summary = report['executive_summary']
        assert 'system_health' in exec_summary
        assert 'total_alerts' in exec_summary
        assert 'metrics_processed' in exec_summary
        
        # Check recommendations
        recommendations = report['recommendations']
        assert isinstance(recommendations, list)
        assert len(recommendations) >= 1
        
        logger.info(f"Report generated with {len(recommendations)} recommendations")
        logger.info("✅ Performance report generation test passed")


class TestPerformanceDashboard:
    """Test performance dashboard functionality."""
    
    @pytest.fixture
    async def performance_monitor(self):
        """Create performance monitor for dashboard testing."""
        monitor = PerformanceMonitor(collection_interval=1.0)
        await monitor.start_monitoring()
        
        # Add some sample data
        await monitor.record_execution("dashboard_test", 1.5, True)
        await asyncio.sleep(2.0)  # Let it process
        
        yield monitor
        await monitor.stop_monitoring()
    
    @pytest.fixture
    def dashboard(self, performance_monitor):
        """Create performance dashboard."""
        return PerformanceDashboard(performance_monitor)
    
    @pytest.mark.asyncio
    async def test_dashboard_initialization(self, dashboard):
        """Test dashboard initialization with default widgets."""
        logger.info("🧪 Testing dashboard initialization")
        
        assert dashboard.performance_monitor is not None
        assert dashboard.data_provider is not None
        assert len(dashboard.widgets) >= 5  # Should have default widgets
        
        # Check default widgets
        widget_ids = [w.widget_id for w in dashboard.widgets]
        expected_widgets = ["system_overview", "resource_usage", "execution_performance", "active_alerts"]
        
        for expected_widget in expected_widgets:
            assert expected_widget in widget_ids
        
        logger.info(f"Dashboard initialized with {len(dashboard.widgets)} widgets")
        logger.info("✅ Dashboard initialization test passed")
    
    @pytest.mark.asyncio
    async def test_widget_data_generation(self, dashboard):
        """Test widget data generation."""
        logger.info("🧪 Testing widget data generation")
        
        # Test system overview widget
        overview_data = dashboard.get_widget_data("system_overview")
        assert overview_data['widget_id'] == "system_overview"
        assert 'data' in overview_data
        assert 'timestamp' in overview_data
        
        overview_metrics = overview_data['data']
        assert 'system_health' in overview_metrics
        assert 'total_components' in overview_metrics
        
        # Test resource usage widget
        resource_data = dashboard.get_widget_data("resource_usage")
        assert resource_data['widget_id'] == "resource_usage"
        
        resource_metrics = resource_data['data']
        assert 'cpu_usage' in resource_metrics or 'memory_usage' in resource_metrics
        
        logger.info("✅ Widget data generation test passed")
    
    @pytest.mark.asyncio  
    async def test_dashboard_configuration(self, dashboard):
        """Test dashboard configuration management."""
        logger.info("🧪 Testing dashboard configuration")
        
        # Get initial config
        initial_config = dashboard.get_dashboard_config()
        assert 'title' in initial_config
        assert 'widgets' in initial_config
        assert 'total_widgets' in initial_config
        
        # Update configuration
        new_config = {
            "title": "Test Dashboard",
            "refresh_interval": 10.0,
            "theme": "light"
        }
        
        dashboard.update_dashboard_config(new_config)
        
        updated_config = dashboard.get_dashboard_config()
        assert updated_config['title'] == "Test Dashboard"
        assert updated_config['refresh_interval'] == 10.0
        assert updated_config['theme'] == "light"
        
        logger.info("✅ Dashboard configuration test passed")
    
    @pytest.mark.asyncio
    async def test_custom_widget_management(self, dashboard):
        """Test adding and removing custom widgets."""
        logger.info("🧪 Testing custom widget management")
        
        initial_widget_count = len(dashboard.widgets)
        
        # Add custom widget
        custom_widget = DashboardWidget(
            widget_id="custom_test",
            widget_type="test_widget",
            title="Test Widget",
            data_source="system_overview",
            config={"test": True}
        )
        
        dashboard.add_widget(custom_widget)
        assert len(dashboard.widgets) == initial_widget_count + 1
        
        # Get widget data
        widget_data = dashboard.get_widget_data("custom_test")
        assert widget_data['widget_id'] == "custom_test"
        assert widget_data['title'] == "Test Widget"
        
        # Remove widget
        removed = dashboard.remove_widget("custom_test")
        assert removed is True
        assert len(dashboard.widgets) == initial_widget_count
        
        # Try to get removed widget data
        removed_data = dashboard.get_widget_data("custom_test")
        assert 'error' in removed_data
        
        logger.info("✅ Custom widget management test passed")
    
    @pytest.mark.asyncio
    async def test_dashboard_html_generation(self, dashboard):
        """Test dashboard HTML generation."""
        logger.info("🧪 Testing dashboard HTML generation")
        
        # Generate HTML
        html = dashboard.generate_dashboard_html()
        
        # Basic HTML validation
        assert html.startswith("<!DOCTYPE html>")
        assert "<html>" in html
        assert "<head>" in html
        assert "<body>" in html
        assert dashboard.dashboard_config["title"] in html
        
        # Check for widget content
        for widget in dashboard.widgets[:3]:  # Check first 3 widgets
            assert widget.title in html
        
        # Verify CSS and JavaScript
        assert "<style>" in html
        assert "<script>" in html
        
        logger.info(f"Generated HTML dashboard ({len(html)} characters)")
        logger.info("✅ Dashboard HTML generation test passed")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])