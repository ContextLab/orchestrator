"""
Comprehensive test suite for monitoring dashboard functionality.

Tests the web interface, API endpoints, real-time updates, and integration
with the wrapper monitoring system.
"""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from flask import Flask

from src.orchestrator.analytics.performance_monitor import PerformanceMonitor
from src.orchestrator.core.wrapper_monitoring import (
    WrapperMonitoring, 
    WrapperHealthStatus, 
    WrapperOperationMetrics,
    WrapperOperationStatus
)
from src.orchestrator.web.monitoring_dashboard import MonitoringDashboard, create_monitoring_dashboard


class TestMonitoringDashboard:
    """Test the MonitoringDashboard class functionality."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create a mock performance monitor."""
        return Mock(spec=PerformanceMonitor)
    
    @pytest.fixture
    def wrapper_monitoring(self):
        """Create a mock wrapper monitoring system."""
        monitoring = Mock(spec=WrapperMonitoring)
        
        # Configure default return values
        monitoring.get_system_health.return_value = {
            'overall_health_score': 0.95,
            'system_status': 'Healthy',
            'active_wrappers': 3,
            'healthy_wrappers': 3,
            'unhealthy_wrappers': 0,
            'total_operations': 150,
            'overall_success_rate': 0.98,
            'error_count': 3,
            'operations_per_minute': 12
        }
        
        monitoring.get_active_wrappers.return_value = ['routellm', 'poml', 'external_api']
        
        monitoring.get_wrapper_health.return_value = WrapperHealthStatus(
            wrapper_name='test_wrapper',
            health_score=0.9,
            success_rate=0.95,
            error_rate=0.02,
            fallback_rate=0.03,
            total_operations=100,
            last_activity=datetime.utcnow()
        )
        
        monitoring.export_metrics.return_value = [
            {
                'operation_id': 'op1',
                'wrapper_name': 'test_wrapper',
                'success': True,
                'duration_ms': 150.5,
                'start_time': datetime.utcnow().isoformat()
            }
        ]
        
        return monitoring
    
    @pytest.fixture
    def dashboard(self, wrapper_monitoring, performance_monitor):
        """Create a monitoring dashboard instance."""
        return MonitoringDashboard(wrapper_monitoring, performance_monitor)
    
    def test_dashboard_initialization(self, wrapper_monitoring, performance_monitor):
        """Test dashboard initialization."""
        dashboard = MonitoringDashboard(wrapper_monitoring, performance_monitor)
        
        assert dashboard.monitoring == wrapper_monitoring
        assert dashboard.performance_monitor == performance_monitor
        assert isinstance(dashboard.app, Flask)
        assert dashboard._update_interval == 5
        assert dashboard._last_metrics_update is not None
    
    def test_create_monitoring_dashboard(self, wrapper_monitoring, performance_monitor):
        """Test dashboard factory function."""
        dashboard = create_monitoring_dashboard(wrapper_monitoring, performance_monitor)
        
        assert isinstance(dashboard, MonitoringDashboard)
        assert dashboard.monitoring == wrapper_monitoring
        assert dashboard.performance_monitor == performance_monitor


class TestDashboardEndpoints:
    """Test dashboard API endpoints."""
    
    @pytest.fixture
    def app(self, wrapper_monitoring, performance_monitor):
        """Create Flask test app with dashboard."""
        dashboard = MonitoringDashboard(wrapper_monitoring, performance_monitor)
        dashboard.app.config['TESTING'] = True
        return dashboard.app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return app.test_client()
    
    def test_health_check_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'version' in data
    
    def test_dashboard_home(self, client):
        """Test the main dashboard page."""
        response = client.get('/')
        
        assert response.status_code == 200
        assert b'System Dashboard' in response.data
        assert b'monitoring dashboard' in response.data.lower()
    
    def test_system_health_api(self, client, wrapper_monitoring):
        """Test the system health API endpoint."""
        response = client.get('/api/system/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'data' in data
        assert 'timestamp' in data
        
        # Verify the monitoring system was called
        wrapper_monitoring.get_system_health.assert_called_once()
    
    def test_system_health_api_error(self, client, wrapper_monitoring):
        """Test system health API error handling."""
        # Configure mock to raise exception
        wrapper_monitoring.get_system_health.side_effect = Exception("Database error")
        
        response = client.get('/api/system/health')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'error' in data
        assert 'Database error' in data['error']
    
    def test_wrappers_health_api(self, client, wrapper_monitoring):
        """Test the wrappers health API endpoint."""
        # Configure mock return data
        health_data = {
            'routellm': {
                'health_score': 0.95,
                'status': 'healthy',
                'success_rate': 0.98,
                'error_rate': 0.01,
                'fallback_rate': 0.01,
                'total_operations': 200,
                'last_activity': datetime.utcnow().isoformat()
            },
            'poml': {
                'health_score': 0.87,
                'status': 'warning',
                'success_rate': 0.90,
                'error_rate': 0.05,
                'fallback_rate': 0.05,
                'total_operations': 150,
                'last_activity': datetime.utcnow().isoformat()
            }
        }
        
        # Mock the health status objects
        routellm_health = WrapperHealthStatus('routellm', health_score=0.95, success_rate=0.98)
        poml_health = WrapperHealthStatus('poml', health_score=0.87, success_rate=0.90)
        
        wrapper_monitoring.get_active_wrappers.return_value = ['routellm', 'poml']
        wrapper_monitoring.get_wrapper_health.side_effect = [routellm_health, poml_health]
        
        response = client.get('/api/wrappers/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'data' in data
        
        # Verify calls were made
        wrapper_monitoring.get_active_wrappers.assert_called_once()
        assert wrapper_monitoring.get_wrapper_health.call_count == 2
    
    def test_wrapper_metrics_api(self, client, wrapper_monitoring):
        """Test the wrapper metrics API endpoint."""
        # Configure mock data
        metrics_data = [
            {
                'timestamp': '2023-01-01T10:00:00',
                'total_operations': 50,
                'success_rate': 0.96,
                'avg_response_time': 125.3
            },
            {
                'timestamp': '2023-01-01T11:00:00',
                'total_operations': 45,
                'success_rate': 0.98,
                'avg_response_time': 110.7
            }
        ]
        
        wrapper_monitoring.get_wrapper_metrics.return_value = metrics_data
        
        response = client.get('/api/wrappers/test_wrapper/metrics?hours=24')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert data['wrapper_name'] == 'test_wrapper'
        assert data['time_range_hours'] == 24
        assert len(data['data']) == 2
        
        # Verify the correct parameters were passed
        wrapper_monitoring.get_wrapper_metrics.assert_called_once()
        args = wrapper_monitoring.get_wrapper_metrics.call_args[0]
        assert args[0] == 'test_wrapper'
        assert isinstance(args[1], datetime)  # since parameter
    
    def test_wrapper_metrics_api_error(self, client, wrapper_monitoring):
        """Test wrapper metrics API error handling."""
        wrapper_monitoring.get_wrapper_metrics.side_effect = Exception("Metrics error")
        
        response = client.get('/api/wrappers/test_wrapper/metrics')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'Metrics error' in data['error']
    
    def test_export_metrics_api(self, client, wrapper_monitoring):
        """Test the metrics export API endpoint."""
        response = client.get('/api/metrics/export')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert data['export_format'] == 'json'
        assert 'data' in data
        
        # Verify export was called
        wrapper_monitoring.export_metrics.assert_called_once()
    
    def test_export_metrics_api_unsupported_format(self, client):
        """Test export API with unsupported format."""
        response = client.get('/api/metrics/export?format=xml')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'Unsupported export format' in data['error']
    
    def test_performance_chart_api(self, client, wrapper_monitoring):
        """Test the performance chart API endpoint."""
        with patch.object(MonitoringDashboard, '_generate_performance_chart') as mock_chart:
            mock_chart.return_value = {
                'data': [{'x': [1, 2, 3], 'y': [90, 95, 92], 'type': 'scatter'}],
                'layout': {'title': 'Performance Chart'}
            }
            
            response = client.get('/api/charts/performance?wrapper=test&hours=24')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'data' in data
            assert 'layout' in data
            
            # Verify chart generation was called with correct parameters
            mock_chart.assert_called_once_with('test', 24)
    
    def test_performance_chart_api_error(self, client):
        """Test performance chart API error handling."""
        with patch.object(MonitoringDashboard, '_generate_performance_chart') as mock_chart:
            mock_chart.side_effect = Exception("Chart error")
            
            response = client.get('/api/charts/performance')
            
            assert response.status_code == 500
            data = json.loads(response.data)
            assert 'error' in data
            assert 'Chart error' in data['error']
    
    def test_health_chart_api(self, client):
        """Test the health chart API endpoint."""
        with patch.object(MonitoringDashboard, '_generate_health_chart') as mock_chart:
            mock_chart.return_value = {
                'data': [{'x': ['wrapper1', 'wrapper2'], 'y': [95, 87], 'type': 'bar'}],
                'layout': {'title': 'Health Chart'}
            }
            
            response = client.get('/api/charts/health')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'data' in data
            assert 'layout' in data
            
            mock_chart.assert_called_once()


class TestDashboardChartGeneration:
    """Test chart generation functionality."""
    
    @pytest.fixture
    def dashboard(self, wrapper_monitoring, performance_monitor):
        """Create dashboard instance."""
        return MonitoringDashboard(wrapper_monitoring, performance_monitor)
    
    def test_generate_performance_chart_single_wrapper(self, dashboard, wrapper_monitoring):
        """Test performance chart generation for single wrapper."""
        # Mock metrics data
        metrics_data = [
            {
                'timestamp': '2023-01-01T10:00:00',
                'success_rate': 0.96,
                'avg_response_time': 125.3
            },
            {
                'timestamp': '2023-01-01T11:00:00',
                'success_rate': 0.98,
                'avg_response_time': 110.7
            }
        ]
        
        wrapper_monitoring.get_wrapper_metrics.return_value = metrics_data
        
        chart_config = dashboard._generate_performance_chart('test_wrapper', 24)
        
        assert 'data' in chart_config
        assert 'layout' in chart_config
        assert chart_config['layout']['title']['text'] == 'Performance Metrics - test_wrapper'
        
        # Should have two traces for single wrapper (success rate and response time)
        assert len(chart_config['data']) == 2
    
    def test_generate_performance_chart_all_wrappers(self, dashboard, wrapper_monitoring):
        """Test performance chart generation for all wrappers."""
        wrapper_monitoring.get_active_wrappers.return_value = ['wrapper1', 'wrapper2']
        wrapper_monitoring.get_wrapper_metrics.return_value = [
            {'timestamp': '2023-01-01T10:00:00', 'success_rate': 0.95}
        ]
        
        chart_config = dashboard._generate_performance_chart('all', 24)
        
        assert 'data' in chart_config
        assert 'layout' in chart_config
        assert chart_config['layout']['title']['text'] == 'Performance Metrics - all'
        
        # Should have traces for each wrapper
        wrapper_monitoring.get_active_wrappers.assert_called_once()
    
    def test_generate_health_chart(self, dashboard, wrapper_monitoring):
        """Test health status chart generation."""
        # Mock health data
        health1 = WrapperHealthStatus('wrapper1', health_score=0.95)
        health2 = WrapperHealthStatus('wrapper2', health_score=0.75)
        health3 = WrapperHealthStatus('wrapper3', health_score=0.45)
        
        wrapper_monitoring.get_active_wrappers.return_value = ['wrapper1', 'wrapper2', 'wrapper3']
        wrapper_monitoring.get_wrapper_health.side_effect = [health1, health2, health3]
        
        chart_config = dashboard._generate_health_chart()
        
        assert 'data' in chart_config
        assert 'layout' in chart_config
        assert chart_config['layout']['title']['text'] == 'Wrapper Health Scores'
        
        # Should be a bar chart with one trace
        assert len(chart_config['data']) == 1
        bar_data = chart_config['data'][0]
        assert bar_data['type'] == 'bar'
        
        # Check data values
        assert bar_data['x'] == ['wrapper1', 'wrapper2', 'wrapper3']
        assert bar_data['y'] == [95.0, 75.0, 45.0]  # Converted to percentages
        
        # Check colors (green, yellow, red based on health scores)
        assert bar_data['marker']['color'] == ['green', 'yellow', 'red']


class TestRealTimeUpdates:
    """Test real-time update functionality."""
    
    @pytest.fixture
    def dashboard(self, wrapper_monitoring, performance_monitor):
        """Create dashboard with SocketIO."""
        return MonitoringDashboard(wrapper_monitoring, performance_monitor)
    
    def test_broadcast_metrics_update(self, dashboard, wrapper_monitoring):
        """Test metrics broadcasting."""
        # Mock system health data
        health_data = {
            'overall_health_score': 0.92,
            'active_wrappers': 3,
            'total_operations': 250
        }
        wrapper_monitoring.get_system_health.return_value = health_data
        
        # Mock socketio emit
        with patch.object(dashboard.socketio, 'emit') as mock_emit:
            dashboard.broadcast_metrics_update()
            
            # Verify emit was called with correct data
            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0] == 'metrics_broadcast'
            assert 'system_health' in call_args[0][1]
            assert 'timestamp' in call_args[0][1]
    
    def test_broadcast_metrics_update_error(self, dashboard, wrapper_monitoring):
        """Test metrics broadcasting error handling."""
        wrapper_monitoring.get_system_health.side_effect = Exception("Connection error")
        
        # Should not raise exception
        dashboard.broadcast_metrics_update()
        
        # Should still update last metrics time even on error
        assert dashboard._last_metrics_update is not None


class TestIntegrationScenarios:
    """Test integration scenarios with real monitoring data."""
    
    def test_end_to_end_monitoring_flow(self):
        """Test complete monitoring flow from operation to dashboard."""
        # Create real monitoring instance (not mock)
        performance_monitor = PerformanceMonitor()
        wrapper_monitoring = WrapperMonitoring(performance_monitor)
        
        # Simulate some wrapper operations
        op_id1 = wrapper_monitoring.start_operation("op1", "test_wrapper", "api_call")
        wrapper_monitoring.record_success(op_id1, custom_metrics={'response_size': 1024})
        wrapper_monitoring.end_operation(op_id1)
        
        op_id2 = wrapper_monitoring.start_operation("op2", "test_wrapper", "api_call")
        wrapper_monitoring.record_error(op_id2, "Network timeout")
        wrapper_monitoring.end_operation(op_id2)
        
        # Create dashboard
        dashboard = MonitoringDashboard(wrapper_monitoring, performance_monitor)
        
        # Test system health
        with dashboard.app.test_client() as client:
            response = client.get('/api/system/health')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert data['success'] is True
            health_data = data['data']
            
            # Should show 1 active wrapper
            assert health_data['active_wrappers'] == 1
            assert health_data['total_operations'] == 2
            
            # Test wrapper health
            response = client.get('/api/wrappers/health')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert 'test_wrapper' in data['data']
            wrapper_health = data['data']['test_wrapper']
            assert wrapper_health['total_operations'] == 2
            assert wrapper_health['success_rate'] == 0.5  # 1 success, 1 error
    
    def test_dashboard_with_no_data(self):
        """Test dashboard behavior with no monitoring data."""
        performance_monitor = PerformanceMonitor()
        wrapper_monitoring = WrapperMonitoring(performance_monitor)
        dashboard = MonitoringDashboard(wrapper_monitoring, performance_monitor)
        
        with dashboard.app.test_client() as client:
            # System health should work with no data
            response = client.get('/api/system/health')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            health_data = data['data']
            assert health_data['active_wrappers'] == 0
            assert health_data['total_operations'] == 0
            
            # Wrapper health should return empty
            response = client.get('/api/wrappers/health')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert data['data'] == {}


# Fixtures used across tests
@pytest.fixture
def wrapper_monitoring():
    """Create a mock wrapper monitoring system."""
    monitoring = Mock(spec=WrapperMonitoring)
    
    # Configure default return values
    monitoring.get_system_health.return_value = {
        'overall_health_score': 0.95,
        'system_status': 'Healthy',
        'active_wrappers': 3,
        'healthy_wrappers': 3,
        'unhealthy_wrappers': 0,
        'total_operations': 150,
        'overall_success_rate': 0.98,
        'error_count': 3,
        'operations_per_minute': 12
    }
    
    monitoring.get_active_wrappers.return_value = ['routellm', 'poml', 'external_api']
    
    monitoring.get_wrapper_health.return_value = WrapperHealthStatus(
        wrapper_name='test_wrapper',
        health_score=0.9,
        success_rate=0.95,
        error_rate=0.02,
        fallback_rate=0.03,
        total_operations=100,
        last_activity=datetime.utcnow()
    )
    
    monitoring.export_metrics.return_value = []
    
    return monitoring


@pytest.fixture
def performance_monitor():
    """Create a mock performance monitor."""
    return Mock(spec=PerformanceMonitor)


if __name__ == "__main__":
    pytest.main([__file__])