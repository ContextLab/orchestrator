"""
Quality Control Alerting System

This module provides a comprehensive alerting system for quality threshold breaches,
trend anomalies, and critical quality issues. It integrates with the metrics collection
and analytics systems to provide real-time notifications and escalation procedures.

Key Features:
- Configurable alerting rules with threshold conditions
- Multiple notification channels (email, webhook, slack, etc.)
- Alert escalation and suppression mechanisms
- Integration with quality analytics for trend-based alerts
- Alert history and audit trail
- Rate limiting and alert fatigue prevention
"""

from __future__ import annotations

import asyncio
import json
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Set
import smtplib
import requests
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

from .metrics import QualityMetricsCollector, TimeSeriesMetric, QualityMetric, MetricType
from .analytics import QualityAnalytics, QualityInsight, TrendAnalysis, InsightType
from ..logging.logger import StructuredLogger, get_logger, LogCategory
from ...execution.state import ExecutionContext


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of quality alerts."""
    THRESHOLD = "threshold"
    TREND = "trend"
    ANOMALY = "anomaly"
    PERFORMANCE = "performance"
    SYSTEM = "system"
    COMPLIANCE = "compliance"


class AlertCondition(Enum):
    """Alert condition operators."""
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    EQUAL = "eq"
    NOT_EQUAL = "ne"
    GREATER_EQUAL = "ge"
    LESS_EQUAL = "le"
    CHANGE_PERCENT = "change_pct"
    TREND_DIRECTION = "trend_dir"


class AlertChannel(Enum):
    """Alert delivery channels."""
    EMAIL = "email"
    WEBHOOK = "webhook" 
    SLACK = "slack"
    TEAMS = "teams"
    LOG = "log"
    SMS = "sms"


@dataclass
class AlertRule:
    """Configurable alert rule definition."""
    rule_id: str
    name: str
    description: str
    metric_pattern: str  # Glob pattern to match metrics
    condition: AlertCondition
    threshold_value: Union[float, int, str]
    severity: AlertSeverity
    alert_type: AlertType
    enabled: bool = True
    channels: List[AlertChannel] = field(default_factory=list)
    
    # Alert control
    cooldown_seconds: int = 300  # 5 minutes default cooldown
    max_alerts_per_hour: int = 10
    escalation_after_alerts: Optional[int] = None
    escalation_channels: List[AlertChannel] = field(default_factory=list)
    
    # Additional configuration
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class AlertNotification:
    """Individual alert notification."""
    alert_id: str
    rule_id: str
    timestamp: float
    severity: AlertSeverity
    alert_type: AlertType
    title: str
    message: str
    metric_name: Optional[str] = None
    current_value: Optional[Union[float, int]] = None
    threshold_value: Optional[Union[float, int]] = None
    execution_context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Tracking
    channels_sent: Set[AlertChannel] = field(default_factory=set)
    delivery_attempts: int = 0
    delivered_successfully: bool = False
    escalated: bool = False
    suppressed: bool = False
    acknowledged: bool = False
    resolved: bool = False
    
    @property
    def datetime(self) -> datetime:
        """Get alert timestamp as datetime."""
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)
    
    @property
    def age_seconds(self) -> float:
        """Get alert age in seconds."""
        return time.time() - self.timestamp


class QualityAlertSystem:
    """
    Comprehensive quality alerting system with threshold monitoring,
    trend analysis, and multi-channel notifications.
    """
    
    def __init__(
        self,
        metrics_collector: QualityMetricsCollector,
        analytics: Optional[QualityAnalytics] = None,
        logger: Optional[StructuredLogger] = None,
        config_path: Optional[Path] = None
    ):
        """
        Initialize quality alert system.
        
        Args:
            metrics_collector: Metrics collector for monitoring
            analytics: Optional analytics engine for trend alerts  
            logger: Optional structured logger
            config_path: Optional path to alert configuration
        """
        self.metrics_collector = metrics_collector
        self.analytics = analytics
        self.logger = logger or get_logger("quality_alerts")
        
        # Alert management
        self._alert_rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, AlertNotification] = {}
        self._alert_history: deque[AlertNotification] = deque(maxlen=10000)
        self._alert_lock = threading.RLock()
        
        # Rate limiting and suppression
        self._alert_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._cooldowns: Dict[str, float] = {}
        self._suppressed_alerts: Set[str] = set()
        
        # Notification channels
        self._notification_handlers: Dict[AlertChannel, Callable] = {
            AlertChannel.EMAIL: self._send_email_alert,
            AlertChannel.WEBHOOK: self._send_webhook_alert,
            AlertChannel.SLACK: self._send_slack_alert,
            AlertChannel.LOG: self._send_log_alert,
            AlertChannel.SMS: self._send_sms_alert,
            AlertChannel.TEAMS: self._send_teams_alert
        }
        
        # Configuration
        self._email_config = {}
        self._webhook_config = {}
        self._slack_config = {}
        self._sms_config = {}
        
        # Monitoring state
        self._monitoring_enabled = True
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Load configuration
        if config_path:
            self._load_configuration(config_path)
        
        # Register with metrics collector hooks
        self.metrics_collector.add_collection_hook('on_metric_collected', self._check_metric_alerts)
        
        self.logger.info("Initialized QualityAlertSystem", category=LogCategory.MONITORING)
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add or update an alert rule."""
        with self._alert_lock:
            self._alert_rules[rule.rule_id] = rule
        
        self.logger.info(f"Added alert rule: {rule.name} ({rule.rule_id})", category=LogCategory.MONITORING)
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        with self._alert_lock:
            removed_rule = self._alert_rules.pop(rule_id, None)
        
        if removed_rule:
            self.logger.info(f"Removed alert rule: {rule_id}", category=LogCategory.MONITORING)
            return True
        return False
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable an alert rule."""
        with self._alert_lock:
            if rule_id in self._alert_rules:
                self._alert_rules[rule_id].enabled = True
                self.logger.info(f"Enabled alert rule: {rule_id}", category=LogCategory.MONITORING)
                return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable an alert rule."""
        with self._alert_lock:
            if rule_id in self._alert_rules:
                self._alert_rules[rule_id].enabled = False
                self.logger.info(f"Disabled alert rule: {rule_id}", category=LogCategory.MONITORING)
                return True
        return False
    
    def _check_metric_alerts(self, metric: QualityMetric, **kwargs) -> None:
        """Check if metric triggers any alert rules."""
        if not self._monitoring_enabled:
            return
        
        with self._alert_lock:
            matching_rules = self._get_matching_rules(metric.name)
        
        for rule in matching_rules:
            if not rule.enabled:
                continue
            
            # Check if rule is in cooldown
            if self._is_in_cooldown(rule.rule_id):
                continue
            
            # Check if rate limited
            if self._is_rate_limited(rule.rule_id):
                continue
            
            # Evaluate rule condition
            if self._evaluate_rule_condition(rule, metric):
                self._trigger_alert(rule, metric)
    
    def _get_matching_rules(self, metric_name: str) -> List[AlertRule]:
        """Get alert rules matching the metric name."""
        import fnmatch
        
        matching_rules = []
        for rule in self._alert_rules.values():
            if fnmatch.fnmatch(metric_name, rule.metric_pattern):
                matching_rules.append(rule)
        
        return matching_rules
    
    def _evaluate_rule_condition(self, rule: AlertRule, metric: QualityMetric) -> bool:
        """Evaluate if metric satisfies alert rule condition."""
        try:
            threshold = rule.threshold_value
            value = metric.value
            
            if rule.condition == AlertCondition.GREATER_THAN:
                return value > threshold
            elif rule.condition == AlertCondition.LESS_THAN:
                return value < threshold
            elif rule.condition == AlertCondition.EQUAL:
                return value == threshold
            elif rule.condition == AlertCondition.NOT_EQUAL:
                return value != threshold
            elif rule.condition == AlertCondition.GREATER_EQUAL:
                return value >= threshold
            elif rule.condition == AlertCondition.LESS_EQUAL:
                return value <= threshold
            elif rule.condition == AlertCondition.CHANGE_PERCENT:
                # Need historical data for percentage change
                time_series = self.metrics_collector.get_metric_time_series(metric.name, metric.labels)
                if time_series and len(time_series.values) >= 2:
                    previous_value = time_series.values[-2][1]
                    if previous_value != 0:
                        change_percent = ((value - previous_value) / abs(previous_value)) * 100
                        return abs(change_percent) > threshold
            elif rule.condition == AlertCondition.TREND_DIRECTION:
                # Use analytics for trend analysis
                if self.analytics:
                    time_series = self.metrics_collector.get_metric_time_series(metric.name, metric.labels)
                    if time_series:
                        trend_analysis = self.analytics._analyze_metric_trend(time_series, 1)  # 1 hour window
                        if trend_analysis:
                            return trend_analysis.direction.value == threshold
            
        except Exception as e:
            self.logger.error(f"Error evaluating rule condition: {e}", category=LogCategory.MONITORING, exception=e)
        
        return False
    
    def _is_in_cooldown(self, rule_id: str) -> bool:
        """Check if rule is in cooldown period."""
        cooldown_until = self._cooldowns.get(rule_id, 0)
        return time.time() < cooldown_until
    
    def _is_rate_limited(self, rule_id: str) -> bool:
        """Check if rule is rate limited."""
        rule = self._alert_rules.get(rule_id)
        if not rule:
            return False
        
        alert_times = self._alert_counts[rule_id]
        current_time = time.time()
        
        # Clean old alerts (older than 1 hour)
        while alert_times and current_time - alert_times[0] > 3600:
            alert_times.popleft()
        
        return len(alert_times) >= rule.max_alerts_per_hour
    
    def _trigger_alert(self, rule: AlertRule, metric: QualityMetric) -> None:
        """Trigger an alert for the given rule and metric."""
        alert_id = f"{rule.rule_id}_{metric.name}_{int(time.time())}"
        
        # Create alert notification
        alert = AlertNotification(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            timestamp=time.time(),
            severity=rule.severity,
            alert_type=rule.alert_type,
            title=self._format_alert_title(rule, metric),
            message=self._format_alert_message(rule, metric),
            metric_name=metric.name,
            current_value=metric.value,
            threshold_value=rule.threshold_value,
            metadata={
                'rule_name': rule.name,
                'rule_description': rule.description,
                'metric_labels': metric.labels,
                'metric_metadata': metric.metadata
            }
        )
        
        # Store alert
        with self._alert_lock:
            self._active_alerts[alert_id] = alert
            self._alert_history.append(alert)
        
        # Update rate limiting
        self._alert_counts[rule.rule_id].append(time.time())
        self._cooldowns[rule.rule_id] = time.time() + rule.cooldown_seconds
        
        # Send notifications
        self._send_alert_notifications(alert, rule.channels)
        
        self.logger.warning(
            f"Quality alert triggered: {alert.title}",
            category=LogCategory.MONITORING,
            metadata={
                'alert_id': alert_id,
                'rule_id': rule.rule_id,
                'metric_name': metric.name,
                'current_value': metric.value,
                'threshold_value': rule.threshold_value
            }
        )
    
    def _format_alert_title(self, rule: AlertRule, metric: QualityMetric) -> str:
        """Format alert title."""
        return f"Quality Alert: {rule.name}"
    
    def _format_alert_message(self, rule: AlertRule, metric: QualityMetric) -> str:
        """Format alert message."""
        condition_text = self._format_condition_text(rule.condition, rule.threshold_value)
        
        message = f"""
Quality Alert Triggered

Rule: {rule.name}
Description: {rule.description}
Severity: {rule.severity.value.upper()}

Metric: {metric.name}
Current Value: {metric.value}
Condition: {condition_text}
Threshold: {rule.threshold_value}

Timestamp: {datetime.fromtimestamp(metric.timestamp, tz=timezone.utc).isoformat()}
"""
        
        if metric.labels:
            message += f"\nMetric Labels: {json.dumps(metric.labels, indent=2)}"
        
        return message.strip()
    
    def _format_condition_text(self, condition: AlertCondition, threshold: Union[float, int, str]) -> str:
        """Format condition for human reading."""
        condition_map = {
            AlertCondition.GREATER_THAN: f"greater than {threshold}",
            AlertCondition.LESS_THAN: f"less than {threshold}",
            AlertCondition.EQUAL: f"equal to {threshold}",
            AlertCondition.NOT_EQUAL: f"not equal to {threshold}",
            AlertCondition.GREATER_EQUAL: f"greater than or equal to {threshold}",
            AlertCondition.LESS_EQUAL: f"less than or equal to {threshold}",
            AlertCondition.CHANGE_PERCENT: f"changed by more than {threshold}%",
            AlertCondition.TREND_DIRECTION: f"trend direction is {threshold}"
        }
        
        return condition_map.get(condition, f"{condition.value} {threshold}")
    
    def _send_alert_notifications(self, alert: AlertNotification, channels: List[AlertChannel]) -> None:
        """Send alert through specified notification channels."""
        for channel in channels:
            try:
                handler = self._notification_handlers.get(channel)
                if handler:
                    success = handler(alert)
                    if success:
                        alert.channels_sent.add(channel)
                        alert.delivered_successfully = True
                    alert.delivery_attempts += 1
                else:
                    self.logger.warning(f"No handler configured for channel: {channel}")
                    
            except Exception as e:
                self.logger.error(f"Failed to send alert via {channel}: {e}", category=LogCategory.MONITORING, exception=e)
    
    def _send_email_alert(self, alert: AlertNotification) -> bool:
        """Send alert via email."""
        if not self._email_config:
            self.logger.warning("Email configuration not available")
            return False
        
        try:
            msg = MimeMultipart()
            msg['From'] = self._email_config['from_address']
            msg['To'] = ', '.join(self._email_config['to_addresses'])
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            msg.attach(MimeText(alert.message, 'plain'))
            
            with smtplib.SMTP(self._email_config['smtp_server'], self._email_config['smtp_port']) as server:
                if self._email_config.get('use_tls'):
                    server.starttls()
                if self._email_config.get('username'):
                    server.login(self._email_config['username'], self._email_config['password'])
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Email alert failed: {e}", exception=e)
            return False
    
    def _send_webhook_alert(self, alert: AlertNotification) -> bool:
        """Send alert via webhook."""
        if not self._webhook_config:
            return False
        
        try:
            payload = {
                'alert_id': alert.alert_id,
                'severity': alert.severity.value,
                'type': alert.alert_type.value,
                'title': alert.title,
                'message': alert.message,
                'timestamp': alert.timestamp,
                'metric_name': alert.metric_name,
                'current_value': alert.current_value,
                'threshold_value': alert.threshold_value,
                'metadata': alert.metadata
            }
            
            response = requests.post(
                self._webhook_config['url'],
                json=payload,
                headers=self._webhook_config.get('headers', {}),
                timeout=10
            )
            
            response.raise_for_status()
            return True
            
        except Exception as e:
            self.logger.error(f"Webhook alert failed: {e}", exception=e)
            return False
    
    def _send_slack_alert(self, alert: AlertNotification) -> bool:
        """Send alert via Slack."""
        if not self._slack_config:
            return False
        
        try:
            color_map = {
                AlertSeverity.INFO: 'good',
                AlertSeverity.WARNING: 'warning', 
                AlertSeverity.ERROR: 'danger',
                AlertSeverity.CRITICAL: 'danger'
            }
            
            payload = {
                'attachments': [{
                    'color': color_map.get(alert.severity, 'warning'),
                    'title': alert.title,
                    'text': alert.message,
                    'fields': [
                        {'title': 'Metric', 'value': alert.metric_name, 'short': True},
                        {'title': 'Current Value', 'value': str(alert.current_value), 'short': True},
                        {'title': 'Threshold', 'value': str(alert.threshold_value), 'short': True},
                        {'title': 'Severity', 'value': alert.severity.value.upper(), 'short': True}
                    ],
                    'ts': int(alert.timestamp)
                }]
            }
            
            response = requests.post(
                self._slack_config['webhook_url'],
                json=payload,
                timeout=10
            )
            
            response.raise_for_status()
            return True
            
        except Exception as e:
            self.logger.error(f"Slack alert failed: {e}", exception=e)
            return False
    
    def _send_teams_alert(self, alert: AlertNotification) -> bool:
        """Send alert via Microsoft Teams."""
        if not self._webhook_config:  # Teams uses webhook format
            return False
        
        try:
            color_map = {
                AlertSeverity.INFO: 'Good',
                AlertSeverity.WARNING: 'Warning',
                AlertSeverity.ERROR: 'Attention',
                AlertSeverity.CRITICAL: 'Attention'
            }
            
            payload = {
                '@type': 'MessageCard',
                '@context': 'http://schema.org/extensions',
                'themeColor': color_map.get(alert.severity, 'Warning'),
                'summary': alert.title,
                'sections': [{
                    'activityTitle': alert.title,
                    'activitySubtitle': f"Severity: {alert.severity.value.upper()}",
                    'text': alert.message,
                    'facts': [
                        {'name': 'Metric', 'value': alert.metric_name},
                        {'name': 'Current Value', 'value': str(alert.current_value)},
                        {'name': 'Threshold', 'value': str(alert.threshold_value)}
                    ]
                }]
            }
            
            response = requests.post(
                self._webhook_config['teams_url'],
                json=payload,
                timeout=10
            )
            
            response.raise_for_status()
            return True
            
        except Exception as e:
            self.logger.error(f"Teams alert failed: {e}", exception=e)
            return False
    
    def _send_log_alert(self, alert: AlertNotification) -> bool:
        """Send alert to log output."""
        log_level_map = {
            AlertSeverity.INFO: self.logger.info,
            AlertSeverity.WARNING: self.logger.warning,
            AlertSeverity.ERROR: self.logger.error,
            AlertSeverity.CRITICAL: self.logger.critical
        }
        
        log_func = log_level_map.get(alert.severity, self.logger.warning)
        log_func(
            alert.message,
            category=LogCategory.MONITORING,
            metadata={
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type.value,
                'metric_name': alert.metric_name,
                'current_value': alert.current_value,
                'threshold_value': alert.threshold_value
            }
        )
        
        return True
    
    def _send_sms_alert(self, alert: AlertNotification) -> bool:
        """Send alert via SMS (placeholder - requires SMS provider integration)."""
        # This would require integration with SMS providers like Twilio, AWS SNS, etc.
        self.logger.warning("SMS alerting not implemented - requires SMS provider configuration")
        return False
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert."""
        with self._alert_lock:
            alert = self._active_alerts.get(alert_id)
            if alert:
                alert.acknowledged = True
                alert.metadata['acknowledged_by'] = acknowledged_by
                alert.metadata['acknowledged_at'] = time.time()
                self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an alert."""
        with self._alert_lock:
            alert = self._active_alerts.get(alert_id)
            if alert:
                alert.resolved = True
                alert.metadata['resolved_by'] = resolved_by
                alert.metadata['resolved_at'] = time.time()
                self.logger.info(f"Alert {alert_id} resolved by {resolved_by}")
                return True
        return False
    
    def suppress_alerts(self, rule_ids: List[str], duration_seconds: int) -> None:
        """Temporarily suppress alerts for specified rules."""
        suppress_until = time.time() + duration_seconds
        
        for rule_id in rule_ids:
            self._suppressed_alerts.add(rule_id)
            self._cooldowns[rule_id] = suppress_until
        
        self.logger.info(f"Suppressed alerts for rules {rule_ids} for {duration_seconds}s")
    
    def get_active_alerts(self, severity_filter: Optional[AlertSeverity] = None) -> List[AlertNotification]:
        """Get currently active alerts."""
        with self._alert_lock:
            alerts = [alert for alert in self._active_alerts.values() if not alert.resolved]
            
            if severity_filter:
                alerts = [alert for alert in alerts if alert.severity == severity_filter]
            
            return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_history(
        self,
        hours: int = 24,
        severity_filter: Optional[AlertSeverity] = None
    ) -> List[AlertNotification]:
        """Get alert history for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._alert_lock:
            alerts = [alert for alert in self._alert_history if alert.timestamp >= cutoff_time]
            
            if severity_filter:
                alerts = [alert for alert in alerts if alert.severity == severity_filter]
            
            return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert statistics for specified time period."""
        alerts = self.get_alert_history(hours)
        
        severity_counts = defaultdict(int)
        type_counts = defaultdict(int)
        rule_counts = defaultdict(int)
        
        for alert in alerts:
            severity_counts[alert.severity.value] += 1
            type_counts[alert.alert_type.value] += 1
            rule_counts[alert.rule_id] += 1
        
        return {
            'total_alerts': len(alerts),
            'active_alerts': len(self.get_active_alerts()),
            'alerts_by_severity': dict(severity_counts),
            'alerts_by_type': dict(type_counts),
            'alerts_by_rule': dict(rule_counts),
            'time_period_hours': hours,
            'suppressed_rules': len(self._suppressed_alerts)
        }
    
    def configure_email(self, config: Dict[str, Any]) -> None:
        """Configure email notifications."""
        self._email_config = config
        self.logger.info("Email notification configured")
    
    def configure_webhook(self, config: Dict[str, Any]) -> None:
        """Configure webhook notifications."""
        self._webhook_config = config
        self.logger.info("Webhook notification configured")
    
    def configure_slack(self, config: Dict[str, Any]) -> None:
        """Configure Slack notifications."""
        self._slack_config = config
        self.logger.info("Slack notification configured")
    
    def _load_configuration(self, config_path: Path) -> None:
        """Load alert configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load alert rules
            for rule_config in config.get('alert_rules', []):
                rule = AlertRule(
                    rule_id=rule_config['rule_id'],
                    name=rule_config['name'],
                    description=rule_config['description'],
                    metric_pattern=rule_config['metric_pattern'],
                    condition=AlertCondition(rule_config['condition']),
                    threshold_value=rule_config['threshold_value'],
                    severity=AlertSeverity(rule_config['severity']),
                    alert_type=AlertType(rule_config['alert_type']),
                    enabled=rule_config.get('enabled', True),
                    channels=[AlertChannel(ch) for ch in rule_config.get('channels', [])],
                    cooldown_seconds=rule_config.get('cooldown_seconds', 300),
                    max_alerts_per_hour=rule_config.get('max_alerts_per_hour', 10)
                )
                self.add_alert_rule(rule)
            
            # Load notification configurations
            if 'email_config' in config:
                self.configure_email(config['email_config'])
            if 'webhook_config' in config:
                self.configure_webhook(config['webhook_config'])
            if 'slack_config' in config:
                self.configure_slack(config['slack_config'])
            
            self.logger.info(f"Loaded alert configuration from {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load alert configuration: {e}", exception=e)
    
    def export_configuration(self, output_path: Path) -> None:
        """Export current alert configuration to file."""
        config = {
            'alert_rules': [
                {
                    'rule_id': rule.rule_id,
                    'name': rule.name,
                    'description': rule.description,
                    'metric_pattern': rule.metric_pattern,
                    'condition': rule.condition.value,
                    'threshold_value': rule.threshold_value,
                    'severity': rule.severity.value,
                    'alert_type': rule.alert_type.value,
                    'enabled': rule.enabled,
                    'channels': [ch.value for ch in rule.channels],
                    'cooldown_seconds': rule.cooldown_seconds,
                    'max_alerts_per_hour': rule.max_alerts_per_hour
                }
                for rule in self._alert_rules.values()
            ],
            'email_config': self._email_config,
            'webhook_config': self._webhook_config,
            'slack_config': self._slack_config
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self.logger.info(f"Exported alert configuration to {output_path}")