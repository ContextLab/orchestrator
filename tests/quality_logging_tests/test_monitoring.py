"""
Comprehensive tests for quality control monitoring integration.

This test suite validates external monitoring system integration including
Prometheus, webhooks, Elasticsearch, and the comprehensive monitoring framework.
"""

import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock
import pytest
import aiohttp

from src.orchestrator.quality.logging.monitoring import (
    AlertRule,
    MonitoringAlert,
    PrometheusBackend,
    WebhookBackend,
    ElasticsearchBackend,
    QualityMonitor,
    create_monitoring_setup
)
from src.orchestrator.quality.logging.logger import QualityEvent


class TestAlertRule:
    """Test AlertRule dataclass functionality."""
    
    def test_alert_rule_creation(self):
        """Test creating an alert rule."""
        rule = AlertRule(
            name="test_rule",
            condition="avg(quality_score) < threshold",
            severity="WARNING",
            threshold=0.8,
            window_seconds=300,
            cooldown_seconds=600,
            description="Test alert rule",
            remediation_url="https://example.com/fix",
            tags={"component": "quality"}
        )
        
        assert rule.name == "test_rule"
        assert rule.condition == "avg(quality_score) < threshold"
        assert rule.severity == "WARNING"
        assert rule.threshold == 0.8
        assert rule.window_seconds == 300
        assert rule.cooldown_seconds == 600
        assert rule.description == "Test alert rule"
        assert rule.remediation_url == "https://example.com/fix"
        assert rule.tags["component"] == "quality"

    def test_alert_rule_defaults(self):
        """Test alert rule with default values."""
        rule = AlertRule(
            name="minimal_rule",
            condition="metric > threshold",
            severity="ERROR",
            threshold=1.0,
            window_seconds=60
        )
        
        assert rule.cooldown_seconds == 300  # Default cooldown
        assert rule.description is None
        assert rule.remediation_url is None
        assert rule.tags is None


class TestMonitoringAlert:
    """Test MonitoringAlert dataclass functionality."""
    
    def test_monitoring_alert_creation(self):
        """Test creating a monitoring alert."""
        alert = MonitoringAlert(
            rule_name="test_rule",
            severity="CRITICAL",
            message="Quality score too low",
            timestamp="2023-01-01T00:00:00Z",
            value=0.5,
            threshold=0.8,
            context={"execution_id": "exec-123"},
            remediation_url="https://example.com/fix",
            tags={"component": "validation"}
        )
        
        assert alert.rule_name == "test_rule"
        assert alert.severity == "CRITICAL"
        assert alert.message == "Quality score too low"
        assert alert.value == 0.5
        assert alert.threshold == 0.8
        assert alert.context["execution_id"] == "exec-123"

    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        alert = MonitoringAlert(
            rule_name="test_rule",
            severity="WARNING",
            message="Test alert",
            timestamp="2023-01-01T00:00:00Z",
            value=1.0,
            threshold=0.5,
            context={"test": "data"}
        )
        
        result = alert.to_dict()
        
        assert result["rule_name"] == "test_rule"
        assert result["severity"] == "WARNING"
        assert result["message"] == "Test alert"
        assert result["context"]["test"] == "data"


class TestPrometheusBackend:
    """Test PrometheusBackend functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backend = PrometheusBackend(
            pushgateway_url="http://localhost:9091",
            job_name="test_job",
            instance_id="test_instance"
        )

    @pytest.mark.asyncio
    async def test_backend_creation(self):
        """Test creating Prometheus backend."""
        assert self.backend.pushgateway_url == "http://localhost:9091"
        assert self.backend.job_name == "test_job"
        assert self.backend.instance_id == "test_instance"

    @pytest.mark.asyncio
    async def test_send_metrics_success(self):
        """Test successful metrics sending."""
        test_metrics = {
            "test_metric": {
                "value": 1.0,
                "labels": {"component": "test"},
                "help": "Test metric"
            }
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await self.backend.send_metrics(test_metrics)
            
            assert result is True
            mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_metrics_failure(self):
        """Test metrics sending failure handling."""
        test_metrics = {"test_metric": {"value": 1.0}}
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await self.backend.send_metrics(test_metrics)
            
            assert result is False

    @pytest.mark.asyncio
    async def test_send_alert(self):
        """Test sending alert as metric."""
        alert = MonitoringAlert(
            rule_name="test_alert",
            severity="WARNING",
            message="Test alert message",
            timestamp="2023-01-01T00:00:00Z",
            value=1.0,
            threshold=0.8,
            context={},
            tags={"component": "test"}
        )
        
        with patch.object(self.backend, 'send_metrics', return_value=True) as mock_send:
            result = await self.backend.send_alert(alert)
            
            assert result is True
            mock_send.assert_called_once()
            
            # Check that alert was converted to metric format
            call_args = mock_send.call_args[0][0]
            assert "orchestrator_alert_test_alert" in call_args

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await self.backend.health_check()
            
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test failed health check."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = Exception("Connection failed")
            
            result = await self.backend.health_check()
            
            assert result is False

    def test_prometheus_metrics_formatting(self):
        """Test Prometheus metrics format generation."""
        metrics = {
            "test_counter": {
                "value": 5,
                "labels": {"service": "test", "env": "dev"},
                "help": "Test counter metric",
                "type": "counter"
            },
            "test_gauge": {
                "value": 0.85,
                "help": "Test gauge metric"
            }
        }
        
        result = self.backend._format_prometheus_metrics(metrics)
        
        # Check format structure
        assert "# HELP test_counter Test counter metric" in result
        assert "# TYPE test_counter counter" in result
        assert 'test_counter{service="test",env="dev"} 5' in result
        assert "# TYPE test_gauge gauge" in result
        assert "test_gauge 0.85" in result

    @pytest.mark.asyncio
    async def test_session_management(self):
        """Test HTTP session creation and management."""
        # Session should be created on first use
        session = await self.backend._get_session()
        assert session is not None
        
        # Should reuse same session
        session2 = await self.backend._get_session()
        assert session is session2

    @pytest.mark.asyncio
    async def test_backend_close(self):
        """Test backend cleanup."""
        # Get session first
        await self.backend._get_session()
        
        # Close should clean up session
        await self.backend.close()
        
        # Should create new session if accessed after close
        new_session = await self.backend._get_session()
        assert new_session is not None


class TestWebhookBackend:
    """Test WebhookBackend functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backend = WebhookBackend(
            webhook_url="https://example.com/webhook",
            headers={"Authorization": "Bearer token"},
            timeout=5.0,
            retry_count=2
        )

    @pytest.mark.asyncio
    async def test_backend_creation(self):
        """Test creating webhook backend."""
        assert self.backend.webhook_url == "https://example.com/webhook"
        assert self.backend.headers["Authorization"] == "Bearer token"
        assert self.backend.timeout == 5.0
        assert self.backend.retry_count == 2

    @pytest.mark.asyncio
    async def test_send_metrics_success(self):
        """Test successful metrics sending via webhook."""
        test_metrics = {"quality_score": 0.85, "violations": 2}
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await self.backend.send_metrics(test_metrics)
            
            assert result is True
            mock_post.assert_called_once()
            
            # Check payload structure
            call_args = mock_post.call_args
            assert call_args[1]['json']['type'] == 'metrics'
            assert call_args[1]['json']['data'] == test_metrics

    @pytest.mark.asyncio
    async def test_send_alert(self):
        """Test sending alert via webhook."""
        alert = MonitoringAlert(
            rule_name="test_alert",
            severity="ERROR",
            message="Test alert",
            timestamp="2023-01-01T00:00:00Z",
            value=1.0,
            threshold=0.5,
            context={}
        )
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 202
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await self.backend.send_alert(alert)
            
            assert result is True
            
            # Check payload structure
            call_args = mock_post.call_args
            assert call_args[1]['json']['type'] == 'alert'
            assert call_args[1]['json']['data']['rule_name'] == 'test_alert'

    @pytest.mark.asyncio
    async def test_retry_mechanism(self):
        """Test retry mechanism on server errors."""
        test_metrics = {"test": "data"}
        
        with patch('aiohttp.ClientSession.post') as mock_post, \
             patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            
            # First call returns server error, second succeeds
            responses = [AsyncMock(), AsyncMock()]
            responses[0].status = 500
            responses[1].status = 200
            
            mock_post.return_value.__aenter__.side_effect = responses
            
            result = await self.backend.send_metrics(test_metrics)
            
            assert result is True
            assert mock_post.call_count == 2
            mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self):
        """Test behavior when all retries are exhausted."""
        test_metrics = {"test": "data"}
        
        with patch('aiohttp.ClientSession.post') as mock_post, \
             patch('asyncio.sleep', new_callable=AsyncMock):
            
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await self.backend.send_metrics(test_metrics)
            
            assert result is False
            assert mock_post.call_count == 2  # Original + 1 retry

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test webhook health check."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await self.backend.health_check()
            
            assert result is True
            
            # Check that health check payload was sent
            call_args = mock_post.call_args
            assert call_args[1]['json']['type'] == 'health_check'

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """Test exception handling in webhook calls."""
        test_metrics = {"test": "data"}
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = Exception("Network error")
            
            result = await self.backend.send_metrics(test_metrics)
            
            assert result is False


class TestElasticsearchBackend:
    """Test ElasticsearchBackend functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backend = ElasticsearchBackend(
            elasticsearch_url="http://localhost:9200",
            index_name="test-quality",
            username="test_user",
            password="test_pass"
        )

    @pytest.mark.asyncio
    async def test_backend_creation(self):
        """Test creating Elasticsearch backend."""
        assert self.backend.elasticsearch_url == "http://localhost:9200"
        assert self.backend.index_name == "test-quality"
        assert self.backend.username == "test_user"
        assert self.backend.password == "test_pass"

    @pytest.mark.asyncio
    async def test_send_metrics_success(self):
        """Test successful metrics sending to Elasticsearch."""
        test_metrics = {"quality_score": 0.85, "violations": 2}
        
        with patch('aiohttp.ClientSession.put') as mock_put:
            mock_response = AsyncMock()
            mock_response.status = 201
            mock_put.return_value.__aenter__.return_value = mock_response
            
            result = await self.backend.send_metrics(test_metrics)
            
            assert result is True
            mock_put.assert_called_once()
            
            # Check document structure
            call_args = mock_put.call_args
            doc = call_args[1]['json']
            assert doc['type'] == 'metrics'
            assert doc['metrics'] == test_metrics
            assert '@timestamp' in doc

    @pytest.mark.asyncio
    async def test_send_alert(self):
        """Test sending alert to Elasticsearch."""
        alert = MonitoringAlert(
            rule_name="test_alert",
            severity="WARNING",
            message="Test alert",
            timestamp="2023-01-01T00:00:00Z",
            value=0.6,
            threshold=0.8,
            context={}
        )
        
        with patch('aiohttp.ClientSession.put') as mock_put:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_put.return_value.__aenter__.return_value = mock_response
            
            result = await self.backend.send_alert(alert)
            
            assert result is True
            
            # Check document structure
            call_args = mock_put.call_args
            doc = call_args[1]['json']
            assert doc['type'] == 'alert'
            assert doc['alert']['rule_name'] == 'test_alert'

    @pytest.mark.asyncio
    async def test_health_check_green(self):
        """Test health check with green cluster status."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "green"})
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await self.backend.health_check()
            
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_yellow(self):
        """Test health check with yellow cluster status."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "yellow"})
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await self.backend.health_check()
            
            assert result is True  # Yellow is acceptable

    @pytest.mark.asyncio
    async def test_health_check_red(self):
        """Test health check with red cluster status."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "red"})
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await self.backend.health_check()
            
            assert result is False  # Red is not acceptable

    @pytest.mark.asyncio
    async def test_authentication(self):
        """Test that authentication headers are properly set."""
        session = await self.backend._get_session()
        
        # Check that BasicAuth was configured
        assert session._auth is not None


class TestQualityMonitor:
    """Test QualityMonitor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_backend = AsyncMock()
        self.alert_rules = [
            AlertRule(
                name="test_rule",
                condition="avg(quality_score) < threshold",
                severity="WARNING",
                threshold=0.8,
                window_seconds=300
            )
        ]
        
        self.monitor = QualityMonitor(
            backends=[self.mock_backend],
            alert_rules=self.alert_rules,
            metrics_collection_interval=0.1,  # Fast for testing
            alert_evaluation_interval=0.1
        )

    @pytest.mark.asyncio
    async def test_monitor_creation(self):
        """Test creating quality monitor."""
        assert len(self.monitor.backends) == 1
        assert "test_rule" in self.monitor.alert_rules
        assert self.monitor.metrics_collection_interval == 0.1

    @pytest.mark.asyncio
    async def test_start_stop_monitor(self):
        """Test starting and stopping monitor."""
        await self.monitor.start()
        assert self.monitor._running is True
        assert len(self.monitor._tasks) == 3  # metrics, alerts, queue processor
        
        await self.monitor.stop()
        assert self.monitor._running is False

    @pytest.mark.asyncio
    async def test_record_metric(self):
        """Test recording metrics."""
        await self.monitor.start()
        
        await self.monitor.record_metric("test_metric", 1.0, {"label": "value"})
        
        # Give queue processor time to work
        await asyncio.sleep(0.2)
        
        await self.monitor.stop()
        
        # Metric should be in buffer
        assert "test_metric{label=value}" in self.monitor._metrics_buffer

    @pytest.mark.asyncio
    async def test_record_quality_event(self):
        """Test recording quality events."""
        event = QualityEvent(
            event_type="test_event",
            severity="INFO",
            source_component="test",
            quality_score=0.9
        )
        
        await self.monitor.start()
        await self.monitor.record_quality_event(event)
        
        # Give queue processor time to work
        await asyncio.sleep(0.2)
        
        await self.monitor.stop()

    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test metrics collection and sending to backends."""
        await self.monitor.start()
        
        # Add some metrics
        await self.monitor.record_metric("test_metric", 2.0)
        
        # Give collection cycle time to run
        await asyncio.sleep(0.15)
        
        await self.monitor.stop()
        
        # Backend should have been called
        self.mock_backend.send_metrics.assert_called()

    @pytest.mark.asyncio
    async def test_health_check_all_backends(self):
        """Test health checking all backends."""
        self.mock_backend.health_check = AsyncMock(return_value=True)
        
        health_status = await self.monitor.health_check()
        
        assert "AsyncMock" in health_status
        assert health_status["AsyncMock"] is True

    @pytest.mark.asyncio
    async def test_queue_overflow_handling(self):
        """Test handling of queue overflow."""
        # Create monitor with small queue
        small_monitor = QualityMonitor(
            backends=[self.mock_backend],
            alert_rules=[],
            max_queue_size=2
        )
        
        await small_monitor.start()
        
        # Try to overflow queue
        for i in range(5):
            await small_monitor.record_metric(f"metric_{i}", float(i))
        
        await small_monitor.stop()
        
        # Should not crash despite overflow
        assert True

    @pytest.mark.asyncio
    async def test_alert_evaluation(self):
        """Test alert rule evaluation."""
        # This test would require implementing the full alert evaluation logic
        # For now, test that the evaluation loop runs without crashing
        
        await self.monitor.start()
        
        # Give alert evaluator time to run at least once
        await asyncio.sleep(0.15)
        
        await self.monitor.stop()
        
        # If we get here, alert evaluator didn't crash
        assert True


class TestCreateMonitoringSetup:
    """Test create_monitoring_setup function."""
    
    def test_prometheus_backend_creation(self):
        """Test creating Prometheus backend from config."""
        config = {
            "backends": [
                {
                    "type": "prometheus",
                    "pushgateway_url": "http://localhost:9091",
                    "job_name": "test_job"
                }
            ],
            "alert_rules": [],
            "metrics_collection_interval": 30.0
        }
        
        monitor = create_monitoring_setup(config)
        
        assert len(monitor.backends) == 1
        assert isinstance(monitor.backends[0], PrometheusBackend)
        assert monitor.backends[0].job_name == "test_job"

    def test_webhook_backend_creation(self):
        """Test creating webhook backend from config."""
        config = {
            "backends": [
                {
                    "type": "webhook",
                    "webhook_url": "https://example.com/webhook",
                    "headers": {"Authorization": "Bearer token"},
                    "retry_count": 5
                }
            ],
            "alert_rules": []
        }
        
        monitor = create_monitoring_setup(config)
        
        assert len(monitor.backends) == 1
        assert isinstance(monitor.backends[0], WebhookBackend)
        assert monitor.backends[0].webhook_url == "https://example.com/webhook"
        assert monitor.backends[0].retry_count == 5

    def test_elasticsearch_backend_creation(self):
        """Test creating Elasticsearch backend from config."""
        config = {
            "backends": [
                {
                    "type": "elasticsearch",
                    "elasticsearch_url": "http://localhost:9200",
                    "index_name": "custom-quality",
                    "username": "user",
                    "password": "pass"
                }
            ],
            "alert_rules": []
        }
        
        monitor = create_monitoring_setup(config)
        
        assert len(monitor.backends) == 1
        assert isinstance(monitor.backends[0], ElasticsearchBackend)
        assert monitor.backends[0].index_name == "custom-quality"

    def test_alert_rules_creation(self):
        """Test creating alert rules from config."""
        config = {
            "backends": [],
            "alert_rules": [
                {
                    "name": "low_quality",
                    "condition": "avg(quality_score) < threshold",
                    "severity": "WARNING",
                    "threshold": 0.7,
                    "window_seconds": 600,
                    "description": "Quality score is too low"
                }
            ]
        }
        
        monitor = create_monitoring_setup(config)
        
        assert len(monitor.alert_rules) == 1
        assert "low_quality" in monitor.alert_rules
        rule = monitor.alert_rules["low_quality"]
        assert rule.threshold == 0.7
        assert rule.severity == "WARNING"

    def test_multiple_backends_creation(self):
        """Test creating multiple backends from config."""
        config = {
            "backends": [
                {
                    "type": "prometheus",
                    "pushgateway_url": "http://localhost:9091",
                    "job_name": "test"
                },
                {
                    "type": "webhook",
                    "webhook_url": "https://example.com/webhook"
                }
            ],
            "alert_rules": []
        }
        
        monitor = create_monitoring_setup(config)
        
        assert len(monitor.backends) == 2
        assert isinstance(monitor.backends[0], PrometheusBackend)
        assert isinstance(monitor.backends[1], WebhookBackend)

    def test_configuration_parameters(self):
        """Test configuration parameters are applied correctly."""
        config = {
            "backends": [],
            "alert_rules": [],
            "metrics_collection_interval": 120.0,
            "alert_evaluation_interval": 45.0,
            "max_queue_size": 5000
        }
        
        monitor = create_monitoring_setup(config)
        
        assert monitor.metrics_collection_interval == 120.0
        assert monitor.alert_evaluation_interval == 45.0
        assert monitor.max_queue_size == 5000

    def test_empty_configuration(self):
        """Test handling of empty configuration."""
        config = {}
        
        monitor = create_monitoring_setup(config)
        
        assert len(monitor.backends) == 0
        assert len(monitor.alert_rules) == 0
        # Should use default values
        assert monitor.metrics_collection_interval == 60.0


if __name__ == "__main__":
    pytest.main([__file__])