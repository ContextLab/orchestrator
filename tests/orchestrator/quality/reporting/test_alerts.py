"""
Comprehensive tests for quality alerting system.

Tests all aspects of the QualityAlertSystem including:
- Alert rule management and evaluation
- Threshold monitoring and breach detection
- Multi-channel notification delivery
- Rate limiting and suppression
- Alert lifecycle management
"""

import pytest
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any

# Import the system under test
from src.orchestrator.quality.reporting.alerts import (
    QualityAlertSystem,
    AlertRule,
    AlertSeverity,
    AlertType,
    AlertCondition,
    AlertNotification,
    AlertChannel
)
from src.orchestrator.quality.reporting.metrics import (
    QualityMetricsCollector,
    QualityMetric,
    MetricType,
    TimeSeriesMetric
)


@pytest.fixture
def mock_logger():
    """Create mock logger for testing."""
    return Mock()


@pytest.fixture
def mock_metrics_collector():
    """Create mock metrics collector."""
    collector = Mock(spec=QualityMetricsCollector)
    
    # Create sample time series for testing
    quality_metric = TimeSeriesMetric("quality.overall.score", MetricType.GAUGE)
    quality_metric.add_value(45)  # Low quality score
    
    success_metric = TimeSeriesMetric("quality.validation.success_rate", MetricType.GAUGE)
    success_metric.add_value(0.65)  # Low success rate
    
    collector.get_metric_time_series.side_effect = lambda name, labels: {
        "quality.overall.score": quality_metric,
        "quality.validation.success_rate": success_metric
    }.get(name)
    
    return collector


@pytest.fixture
def sample_alert_rules():
    """Create sample alert rules for testing."""
    return [
        AlertRule(
            rule_id="quality_score_critical",
            name="Quality Score Critical",
            description="Alert when quality score drops below critical threshold",
            metric_pattern="quality.*.score",
            condition=AlertCondition.LESS_THAN,
            threshold_value=50.0,
            severity=AlertSeverity.CRITICAL,
            alert_type=AlertType.THRESHOLD,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            cooldown_seconds=300,
            max_alerts_per_hour=5
        ),
        AlertRule(
            rule_id="success_rate_warning",
            name="Low Success Rate Warning",
            description="Alert when validation success rate is low",
            metric_pattern="*.success_rate",
            condition=AlertCondition.LESS_THAN,
            threshold_value=0.8,
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.THRESHOLD,
            channels=[AlertChannel.LOG],
            cooldown_seconds=600
        ),
        AlertRule(
            rule_id="performance_degradation",
            name="Performance Degradation",
            description="Alert on performance degradation trends",
            metric_pattern="*.duration_ms",
            condition=AlertCondition.CHANGE_PERCENT,
            threshold_value=50.0,  # 50% increase
            severity=AlertSeverity.ERROR,
            alert_type=AlertType.TREND,
            channels=[AlertChannel.EMAIL]
        )
    ]


@pytest.fixture
def alert_system(mock_metrics_collector, mock_logger):
    """Create QualityAlertSystem instance for testing."""
    return QualityAlertSystem(
        metrics_collector=mock_metrics_collector,
        analytics=None,  # Not needed for basic tests
        logger=mock_logger
    )


class TestQualityAlertSystem:
    """Test suite for QualityAlertSystem."""
    
    def test_initialization(self, mock_metrics_collector, mock_logger):
        """Test proper initialization of alert system."""
        alert_system = QualityAlertSystem(
            metrics_collector=mock_metrics_collector,
            analytics=None,
            logger=mock_logger
        )
        
        assert alert_system.metrics_collector == mock_metrics_collector
        assert alert_system.logger == mock_logger
        assert len(alert_system._alert_rules) == 0
        assert len(alert_system._active_alerts) == 0
        assert alert_system._monitoring_enabled is True
        assert AlertChannel.EMAIL in alert_system._notification_handlers
        assert AlertChannel.WEBHOOK in alert_system._notification_handlers
        assert AlertChannel.LOG in alert_system._notification_handlers
    
    def test_alert_rule_management(self, alert_system, sample_alert_rules):
        """Test alert rule addition, removal, and management."""
        # Add rules
        for rule in sample_alert_rules:
            alert_system.add_alert_rule(rule)
        
        assert len(alert_system._alert_rules) == 3
        assert "quality_score_critical" in alert_system._alert_rules
        assert "success_rate_warning" in alert_system._alert_rules
        assert "performance_degradation" in alert_system._alert_rules
        
        # Test rule retrieval
        quality_rule = alert_system._alert_rules["quality_score_critical"]
        assert quality_rule.name == "Quality Score Critical"
        assert quality_rule.severity == AlertSeverity.CRITICAL
        
        # Test rule removal
        removed = alert_system.remove_alert_rule("success_rate_warning")
        assert removed is True
        assert len(alert_system._alert_rules) == 2
        assert "success_rate_warning" not in alert_system._alert_rules
        
        # Test removing non-existent rule
        removed = alert_system.remove_alert_rule("non_existent")
        assert removed is False
    
    def test_rule_enable_disable(self, alert_system, sample_alert_rules):
        """Test enabling and disabling alert rules."""
        # Add rule
        rule = sample_alert_rules[0]
        alert_system.add_alert_rule(rule)
        
        # Rule should be enabled by default
        assert alert_system._alert_rules[rule.rule_id].enabled is True
        
        # Disable rule
        disabled = alert_system.disable_rule(rule.rule_id)
        assert disabled is True
        assert alert_system._alert_rules[rule.rule_id].enabled is False
        
        # Enable rule
        enabled = alert_system.enable_rule(rule.rule_id)
        assert enabled is True
        assert alert_system._alert_rules[rule.rule_id].enabled is True
        
        # Test with non-existent rule
        assert alert_system.enable_rule("non_existent") is False
        assert alert_system.disable_rule("non_existent") is False
    
    def test_metric_pattern_matching(self, alert_system, sample_alert_rules):
        """Test metric pattern matching for alert rules."""
        # Add rules
        for rule in sample_alert_rules:
            alert_system.add_alert_rule(rule)
        
        # Test pattern matching
        matches = alert_system._get_matching_rules("quality.overall.score")
        rule_ids = [rule.rule_id for rule in matches]
        assert "quality_score_critical" in rule_ids
        
        matches = alert_system._get_matching_rules("quality.validation.success_rate")
        rule_ids = [rule.rule_id for rule in matches]
        assert "success_rate_warning" in rule_ids
        
        matches = alert_system._get_matching_rules("performance.duration_ms")
        rule_ids = [rule.rule_id for rule in matches]
        assert "performance_degradation" in rule_ids
        
        # Test no matches
        matches = alert_system._get_matching_rules("unmatched.metric")
        assert len(matches) == 0
    
    def test_alert_condition_evaluation(self, alert_system):
        """Test alert condition evaluation logic."""
        # Create test rule
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            description="Test",
            metric_pattern="test.*",
            condition=AlertCondition.GREATER_THAN,
            threshold_value=100,
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.THRESHOLD
        )
        
        # Test GREATER_THAN condition
        metric = QualityMetric("test.metric", 150, MetricType.GAUGE, time.time())
        assert alert_system._evaluate_rule_condition(rule, metric) is True
        
        metric = QualityMetric("test.metric", 50, MetricType.GAUGE, time.time())
        assert alert_system._evaluate_rule_condition(rule, metric) is False
        
        # Test LESS_THAN condition
        rule.condition = AlertCondition.LESS_THAN
        rule.threshold_value = 50
        
        metric = QualityMetric("test.metric", 30, MetricType.GAUGE, time.time())
        assert alert_system._evaluate_rule_condition(rule, metric) is True
        
        metric = QualityMetric("test.metric", 80, MetricType.GAUGE, time.time())
        assert alert_system._evaluate_rule_condition(rule, metric) is False
        
        # Test EQUAL condition
        rule.condition = AlertCondition.EQUAL
        rule.threshold_value = 42
        
        metric = QualityMetric("test.metric", 42, MetricType.GAUGE, time.time())
        assert alert_system._evaluate_rule_condition(rule, metric) is True
        
        metric = QualityMetric("test.metric", 43, MetricType.GAUGE, time.time())
        assert alert_system._evaluate_rule_condition(rule, metric) is False
    
    def test_percentage_change_condition(self, alert_system, mock_metrics_collector):
        """Test percentage change condition evaluation."""
        # Setup time series with historical data
        time_series = TimeSeriesMetric("test.metric", MetricType.GAUGE)
        time_series.add_value(100, time.time() - 60)  # Previous value
        time_series.add_value(150, time.time())       # Current value (50% increase)
        
        mock_metrics_collector.get_metric_time_series.return_value = time_series
        
        rule = AlertRule(
            rule_id="change_rule",
            name="Change Rule",
            description="Test percentage change",
            metric_pattern="test.*",
            condition=AlertCondition.CHANGE_PERCENT,
            threshold_value=25.0,  # 25% threshold
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.TREND
        )
        
        metric = QualityMetric("test.metric", 150, MetricType.GAUGE, time.time())
        
        # Should trigger because 50% > 25%
        assert alert_system._evaluate_rule_condition(rule, metric) is True
        
        # Test with smaller change
        time_series.values[-1] = (time.time(), 110)  # 10% increase
        assert alert_system._evaluate_rule_condition(rule, metric) is False
    
    def test_alert_triggering(self, alert_system, sample_alert_rules, mock_metrics_collector):
        """Test alert triggering mechanism."""
        # Add rule
        rule = sample_alert_rules[0]  # Critical quality score rule
        alert_system.add_alert_rule(rule)
        
        # Create metric that should trigger alert
        metric = QualityMetric(
            name="quality.overall.score",
            value=30,  # Below threshold of 50
            metric_type=MetricType.GAUGE,
            timestamp=time.time(),
            labels={"environment": "test"}
        )
        
        # Mock notification handlers to avoid actual sending
        alert_system._notification_handlers[AlertChannel.EMAIL] = Mock(return_value=True)
        alert_system._notification_handlers[AlertChannel.SLACK] = Mock(return_value=True)
        
        # Trigger alert
        alert_system._trigger_alert(rule, metric)
        
        # Verify alert was created
        assert len(alert_system._active_alerts) == 1
        alert_id = list(alert_system._active_alerts.keys())[0]
        alert = alert_system._active_alerts[alert_id]
        
        assert alert.rule_id == rule.rule_id
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.metric_name == "quality.overall.score"
        assert alert.current_value == 30
        assert alert.threshold_value == 50.0
        
        # Verify cooldown was set
        assert rule.rule_id in alert_system._cooldowns
        
        # Verify rate limiting was updated
        assert len(alert_system._alert_counts[rule.rule_id]) == 1
    
    def test_cooldown_mechanism(self, alert_system, sample_alert_rules):
        """Test alert cooldown mechanism."""
        rule = sample_alert_rules[0]
        rule.cooldown_seconds = 300  # 5 minutes
        alert_system.add_alert_rule(rule)
        
        # Set cooldown
        alert_system._cooldowns[rule.rule_id] = time.time() + 300
        
        # Should be in cooldown
        assert alert_system._is_in_cooldown(rule.rule_id) is True
        
        # Set expired cooldown
        alert_system._cooldowns[rule.rule_id] = time.time() - 10
        
        # Should not be in cooldown
        assert alert_system._is_in_cooldown(rule.rule_id) is False
    
    def test_rate_limiting(self, alert_system, sample_alert_rules):
        """Test alert rate limiting mechanism."""
        rule = sample_alert_rules[0]
        rule.max_alerts_per_hour = 3
        alert_system.add_alert_rule(rule)
        
        # Add alerts within the hour
        current_time = time.time()
        for i in range(3):
            alert_system._alert_counts[rule.rule_id].append(current_time - (i * 60))
        
        # Should be rate limited
        assert alert_system._is_rate_limited(rule.rule_id) is True
        
        # Add old alerts (over 1 hour ago)
        old_time = current_time - 3700  # Over 1 hour ago
        alert_system._alert_counts[rule.rule_id].clear()
        alert_system._alert_counts[rule.rule_id].append(old_time)
        
        # Should not be rate limited
        assert alert_system._is_rate_limited(rule.rule_id) is False
    
    def test_log_notification_handler(self, alert_system, mock_logger):
        """Test log notification handler."""
        alert = AlertNotification(
            alert_id="test_alert",
            rule_id="test_rule",
            timestamp=time.time(),
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.THRESHOLD,
            title="Test Alert",
            message="This is a test alert",
            metric_name="test.metric",
            current_value=42,
            threshold_value=50
        )
        
        # Test log handler
        success = alert_system._send_log_alert(alert)
        
        assert success is True
        assert AlertChannel.LOG in alert.channels_sent
        assert mock_logger.warning.called
    
    @patch('requests.post')
    def test_webhook_notification_handler(self, mock_post, alert_system):
        """Test webhook notification handler."""
        # Configure webhook
        alert_system.configure_webhook({
            "url": "https://example.com/webhook",
            "headers": {"Authorization": "Bearer token"}
        })
        
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        alert = AlertNotification(
            alert_id="test_alert",
            rule_id="test_rule",
            timestamp=time.time(),
            severity=AlertSeverity.ERROR,
            alert_type=AlertType.THRESHOLD,
            title="Test Alert",
            message="Test webhook alert"
        )
        
        # Test webhook handler
        success = alert_system._send_webhook_alert(alert)
        
        assert success is True
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json']['alert_id'] == "test_alert"
        assert call_args[1]['json']['severity'] == "error"
        assert call_args[1]['headers']['Authorization'] == "Bearer token"
    
    @patch('smtplib.SMTP')
    def test_email_notification_handler(self, mock_smtp, alert_system):
        """Test email notification handler."""
        # Configure email
        alert_system.configure_email({
            "smtp_server": "smtp.example.com",
            "smtp_port": 587,
            "from_address": "alerts@example.com",
            "to_addresses": ["admin@example.com"],
            "use_tls": True,
            "username": "user",
            "password": "pass"
        })
        
        # Mock SMTP
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        alert = AlertNotification(
            alert_id="test_alert",
            rule_id="test_rule",
            timestamp=time.time(),
            severity=AlertSeverity.CRITICAL,
            alert_type=AlertType.THRESHOLD,
            title="Critical Alert",
            message="This is a critical alert"
        )
        
        # Test email handler
        success = alert_system._send_email_alert(alert)
        
        assert success is True
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("user", "pass")
        mock_server.send_message.assert_called_once()
    
    @patch('requests.post')
    def test_slack_notification_handler(self, mock_post, alert_system):
        """Test Slack notification handler."""
        # Configure Slack
        alert_system.configure_slack({
            "webhook_url": "https://hooks.slack.com/webhook"
        })
        
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        alert = AlertNotification(
            alert_id="test_alert",
            rule_id="test_rule",
            timestamp=time.time(),
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.THRESHOLD,
            title="Warning Alert",
            message="This is a warning alert",
            metric_name="test.metric",
            current_value=75,
            threshold_value=80
        )
        
        # Test Slack handler
        success = alert_system._send_slack_alert(alert)
        
        assert success is True
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert 'attachments' in call_args[1]['json']
        attachment = call_args[1]['json']['attachments'][0]
        assert attachment['title'] == "Warning Alert"
        assert attachment['color'] == 'warning'
    
    def test_alert_acknowledgment(self, alert_system):
        """Test alert acknowledgment functionality."""
        # Create an active alert
        alert = AlertNotification(
            alert_id="test_alert",
            rule_id="test_rule",
            timestamp=time.time(),
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.THRESHOLD,
            title="Test Alert",
            message="Test message"
        )
        
        alert_system._active_alerts[alert.alert_id] = alert
        
        # Acknowledge alert
        success = alert_system.acknowledge_alert(alert.alert_id, "test_user")
        
        assert success is True
        assert alert.acknowledged is True
        assert alert.metadata['acknowledged_by'] == "test_user"
        assert 'acknowledged_at' in alert.metadata
        
        # Test acknowledging non-existent alert
        success = alert_system.acknowledge_alert("non_existent", "user")
        assert success is False
    
    def test_alert_resolution(self, alert_system):
        """Test alert resolution functionality."""
        # Create an active alert
        alert = AlertNotification(
            alert_id="test_alert",
            rule_id="test_rule",
            timestamp=time.time(),
            severity=AlertSeverity.ERROR,
            alert_type=AlertType.THRESHOLD,
            title="Test Alert",
            message="Test message"
        )
        
        alert_system._active_alerts[alert.alert_id] = alert
        
        # Resolve alert
        success = alert_system.resolve_alert(alert.alert_id, "test_user")
        
        assert success is True
        assert alert.resolved is True
        assert alert.metadata['resolved_by'] == "test_user"
        assert 'resolved_at' in alert.metadata
    
    def test_alert_suppression(self, alert_system, sample_alert_rules):
        """Test alert suppression functionality."""
        # Add rules
        for rule in sample_alert_rules[:2]:
            alert_system.add_alert_rule(rule)
        
        rule_ids = [rule.rule_id for rule in sample_alert_rules[:2]]
        
        # Suppress alerts
        alert_system.suppress_alerts(rule_ids, 3600)  # 1 hour
        
        # Verify rules are suppressed
        for rule_id in rule_ids:
            assert rule_id in alert_system._suppressed_alerts
            assert alert_system._is_in_cooldown(rule_id) is True
    
    def test_active_alerts_retrieval(self, alert_system):
        """Test active alerts retrieval."""
        # Create multiple alerts with different severities
        alerts = [
            AlertNotification(
                alert_id="critical_alert",
                rule_id="rule1",
                timestamp=time.time(),
                severity=AlertSeverity.CRITICAL,
                alert_type=AlertType.THRESHOLD,
                title="Critical Alert",
                message="Critical"
            ),
            AlertNotification(
                alert_id="warning_alert",
                rule_id="rule2",
                timestamp=time.time() - 60,
                severity=AlertSeverity.WARNING,
                alert_type=AlertType.THRESHOLD,
                title="Warning Alert",
                message="Warning"
            ),
            AlertNotification(
                alert_id="resolved_alert",
                rule_id="rule3",
                timestamp=time.time() - 120,
                severity=AlertSeverity.ERROR,
                alert_type=AlertType.THRESHOLD,
                title="Resolved Alert",
                message="Resolved",
                resolved=True
            )
        ]
        
        for alert in alerts:
            alert_system._active_alerts[alert.alert_id] = alert
        
        # Get all active alerts (should exclude resolved)
        active_alerts = alert_system.get_active_alerts()
        assert len(active_alerts) == 2  # Should exclude resolved alert
        
        # Get critical alerts only
        critical_alerts = alert_system.get_active_alerts(AlertSeverity.CRITICAL)
        assert len(critical_alerts) == 1
        assert critical_alerts[0].alert_id == "critical_alert"
        
        # Verify sorting (most recent first)
        assert active_alerts[0].timestamp >= active_alerts[1].timestamp
    
    def test_alert_history_retrieval(self, alert_system):
        """Test alert history retrieval."""
        # Create alerts with different timestamps
        current_time = time.time()
        alerts = [
            AlertNotification(
                alert_id="recent_alert",
                rule_id="rule1",
                timestamp=current_time - 1800,  # 30 minutes ago
                severity=AlertSeverity.WARNING,
                alert_type=AlertType.THRESHOLD,
                title="Recent Alert",
                message="Recent"
            ),
            AlertNotification(
                alert_id="old_alert",
                rule_id="rule2", 
                timestamp=current_time - 90000,  # 25 hours ago
                severity=AlertSeverity.ERROR,
                alert_type=AlertType.THRESHOLD,
                title="Old Alert",
                message="Old"
            )
        ]
        
        for alert in alerts:
            alert_system._alert_history.append(alert)
        
        # Get 24-hour history (should include recent alert only)
        history_24h = alert_system.get_alert_history(hours=24)
        assert len(history_24h) == 1
        assert history_24h[0].alert_id == "recent_alert"
        
        # Get 48-hour history (should include both)
        history_48h = alert_system.get_alert_history(hours=48)
        assert len(history_48h) == 2
        
        # Get warning alerts only
        warning_history = alert_system.get_alert_history(hours=48, severity_filter=AlertSeverity.WARNING)
        assert len(warning_history) == 1
        assert warning_history[0].alert_id == "recent_alert"
    
    def test_alert_statistics(self, alert_system):
        """Test alert statistics calculation."""
        # Create sample alert history
        current_time = time.time()
        alerts = [
            AlertNotification("alert1", "rule1", current_time - 1800, AlertSeverity.CRITICAL, AlertType.THRESHOLD, "Alert 1", "Message 1"),
            AlertNotification("alert2", "rule1", current_time - 3600, AlertSeverity.WARNING, AlertType.TREND, "Alert 2", "Message 2"),
            AlertNotification("alert3", "rule2", current_time - 7200, AlertSeverity.ERROR, AlertType.THRESHOLD, "Alert 3", "Message 3"),
            AlertNotification("alert4", "rule2", current_time - 10800, AlertSeverity.WARNING, AlertType.ANOMALY, "Alert 4", "Message 4", resolved=True)
        ]
        
        for alert in alerts:
            alert_system._alert_history.append(alert)
            if not alert.resolved:
                alert_system._active_alerts[alert.alert_id] = alert
        
        # Get statistics
        stats = alert_system.get_alert_statistics(hours=24)
        
        assert stats['total_alerts'] == 4
        assert stats['active_alerts'] == 3  # Excluding resolved
        assert stats['alerts_by_severity']['critical'] == 1
        assert stats['alerts_by_severity']['warning'] == 2
        assert stats['alerts_by_severity']['error'] == 1
        assert stats['alerts_by_type']['threshold'] == 2
        assert stats['alerts_by_type']['trend'] == 1
        assert stats['alerts_by_type']['anomaly'] == 1
        assert stats['alerts_by_rule']['rule1'] == 2
        assert stats['alerts_by_rule']['rule2'] == 2
    
    def test_configuration_loading(self, alert_system):
        """Test loading configuration from file."""
        config_data = {
            "alert_rules": [
                {
                    "rule_id": "test_rule",
                    "name": "Test Rule",
                    "description": "Test rule from config",
                    "metric_pattern": "test.*",
                    "condition": "lt",
                    "threshold_value": 50.0,
                    "severity": "warning",
                    "alert_type": "threshold",
                    "enabled": True,
                    "channels": ["log", "email"],
                    "cooldown_seconds": 300,
                    "max_alerts_per_hour": 5
                }
            ],
            "email_config": {
                "smtp_server": "smtp.test.com",
                "from_address": "test@example.com",
                "to_addresses": ["admin@example.com"]
            },
            "webhook_config": {
                "url": "https://webhook.example.com"
            }
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(config_data, temp_file)
            temp_path = Path(temp_file.name)
        
        try:
            # Load configuration
            alert_system._load_configuration(temp_path)
            
            # Verify rule was loaded
            assert "test_rule" in alert_system._alert_rules
            rule = alert_system._alert_rules["test_rule"]
            assert rule.name == "Test Rule"
            assert rule.condition == AlertCondition.LESS_THAN
            assert rule.threshold_value == 50.0
            assert rule.severity == AlertSeverity.WARNING
            
            # Verify email config was loaded
            assert alert_system._email_config["smtp_server"] == "smtp.test.com"
            
            # Verify webhook config was loaded
            assert alert_system._webhook_config["url"] == "https://webhook.example.com"
            
        finally:
            temp_path.unlink()
    
    def test_configuration_export(self, alert_system, sample_alert_rules):
        """Test exporting configuration to file."""
        # Add rules and configuration
        for rule in sample_alert_rules:
            alert_system.add_alert_rule(rule)
        
        alert_system.configure_email({"smtp_server": "test.com"})
        alert_system.configure_webhook({"url": "https://test.com"})
        
        # Export configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            alert_system.export_configuration(temp_path)
            
            # Verify export file exists and has content
            assert temp_path.exists()
            assert temp_path.stat().st_size > 0
            
            # Load and verify exported content
            with open(temp_path, 'r') as f:
                exported_config = json.load(f)
            
            assert 'alert_rules' in exported_config
            assert len(exported_config['alert_rules']) == 3
            assert 'email_config' in exported_config
            assert 'webhook_config' in exported_config
            
            # Verify rule structure
            rule = exported_config['alert_rules'][0]
            assert 'rule_id' in rule
            assert 'name' in rule
            assert 'condition' in rule
            assert 'threshold_value' in rule
            
        finally:
            temp_path.unlink()
    
    def test_error_handling(self, alert_system, mock_logger):
        """Test error handling in alert system."""
        # Test notification handler error
        def failing_handler(alert):
            raise Exception("Handler failed")
        
        alert_system._notification_handlers[AlertChannel.EMAIL] = failing_handler
        
        alert = AlertNotification(
            alert_id="test_alert",
            rule_id="test_rule",
            timestamp=time.time(),
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.THRESHOLD,
            title="Test Alert",
            message="Test"
        )
        
        # Should handle error gracefully
        alert_system._send_alert_notifications(alert, [AlertChannel.EMAIL])
        
        # Should have logged error
        mock_logger.error.assert_called()
        
        # Test invalid condition evaluation
        rule = AlertRule(
            rule_id="invalid_rule",
            name="Invalid Rule",
            description="Test",
            metric_pattern="test.*",
            condition="invalid_condition",  # Invalid condition
            threshold_value=50,
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.THRESHOLD
        )
        
        metric = QualityMetric("test.metric", 100, MetricType.GAUGE, time.time())
        
        # Should handle invalid condition gracefully
        result = alert_system._evaluate_rule_condition(rule, metric)
        assert result is False  # Should default to False for invalid conditions


class TestAlertRule:
    """Test suite for AlertRule class."""
    
    def test_alert_rule_creation(self):
        """Test AlertRule creation and properties."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Alert Rule",
            description="This is a test alert rule",
            metric_pattern="test.*.metric",
            condition=AlertCondition.GREATER_THAN,
            threshold_value=100.0,
            severity=AlertSeverity.ERROR,
            alert_type=AlertType.THRESHOLD,
            enabled=True,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            cooldown_seconds=600,
            max_alerts_per_hour=5,
            escalation_after_alerts=3,
            escalation_channels=[AlertChannel.SMS],
            metadata={"source": "test"},
            tags=["test", "example"]
        )
        
        assert rule.rule_id == "test_rule"
        assert rule.name == "Test Alert Rule"
        assert rule.description == "This is a test alert rule"
        assert rule.metric_pattern == "test.*.metric"
        assert rule.condition == AlertCondition.GREATER_THAN
        assert rule.threshold_value == 100.0
        assert rule.severity == AlertSeverity.ERROR
        assert rule.alert_type == AlertType.THRESHOLD
        assert rule.enabled is True
        assert AlertChannel.EMAIL in rule.channels
        assert AlertChannel.SLACK in rule.channels
        assert rule.cooldown_seconds == 600
        assert rule.max_alerts_per_hour == 5
        assert rule.escalation_after_alerts == 3
        assert AlertChannel.SMS in rule.escalation_channels
        assert rule.metadata["source"] == "test"
        assert "test" in rule.tags
        assert "example" in rule.tags


class TestAlertNotification:
    """Test suite for AlertNotification class."""
    
    def test_alert_notification_creation(self):
        """Test AlertNotification creation and properties."""
        timestamp = time.time()
        
        alert = AlertNotification(
            alert_id="test_alert_123",
            rule_id="test_rule",
            timestamp=timestamp,
            severity=AlertSeverity.CRITICAL,
            alert_type=AlertType.THRESHOLD,
            title="Critical Quality Alert",
            message="Quality has dropped below critical threshold",
            metric_name="quality.overall.score",
            current_value=25.0,
            threshold_value=50.0,
            execution_context={"pipeline_id": "test_pipeline"},
            metadata={"source": "test"}
        )
        
        assert alert.alert_id == "test_alert_123"
        assert alert.rule_id == "test_rule"
        assert alert.timestamp == timestamp
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.alert_type == AlertType.THRESHOLD
        assert alert.title == "Critical Quality Alert"
        assert alert.message == "Quality has dropped below critical threshold"
        assert alert.metric_name == "quality.overall.score"
        assert alert.current_value == 25.0
        assert alert.threshold_value == 50.0
        assert alert.execution_context["pipeline_id"] == "test_pipeline"
        assert alert.metadata["source"] == "test"
        
        # Test initial state
        assert len(alert.channels_sent) == 0
        assert alert.delivery_attempts == 0
        assert alert.delivered_successfully is False
        assert alert.escalated is False
        assert alert.suppressed is False
        assert alert.acknowledged is False
        assert alert.resolved is False
        
        # Test datetime property
        dt = alert.datetime
        assert dt.timestamp() == timestamp
        
        # Test age calculation
        age = alert.age_seconds
        assert age >= 0


if __name__ == "__main__":
    pytest.main([__file__])