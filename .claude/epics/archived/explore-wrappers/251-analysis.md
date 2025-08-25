# Issue #251: Configuration & Monitoring - Comprehensive Analysis and Implementation Plan

**Task**: Implement comprehensive configuration management and performance monitoring systems that integrate with the unified wrapper architecture (#249) and provide operational excellence for the RouteLLM (#248) and POML (#250) integrations.

**Created**: 2025-08-25  
**Status**: Implementation Phase  
**Epic**: explore-wrappers  

## Executive Summary

This analysis outlines the implementation of advanced configuration management and performance monitoring systems that extend the unified wrapper architecture. Building on the foundation established in #249, we will create comprehensive operational tools including external tool configuration management, real-time monitoring dashboards, cost tracking integration, and alert systems that provide full operational visibility and control over wrapper integrations.

## Analysis of Current Infrastructure

### Existing Wrapper Architecture (#249) - Foundation Review

**Available Infrastructure:**
- **BaseWrapperConfig**: Abstract configuration class with validation, inheritance, and runtime updates
- **ConfigurationManager**: Centralized configuration management with environment overrides
- **WrapperMonitoring**: Comprehensive monitoring with operation tracking and health status
- **FeatureFlags**: Hierarchical flag system with dependency management
- **AlertSystem**: Rule-based alerting with severity levels and cooldowns

**Key Strengths:**
- Standardized configuration patterns across all wrappers
- Comprehensive monitoring with metrics collection and health tracking
- Feature flag integration with runtime updates
- Alert system with configurable rules and notifications
- Thread-safe operations with proper error handling

**Extension Opportunities:**
- External tool configuration management (API keys, endpoints, etc.)
- Performance monitoring dashboards with visual analytics
- Cost tracking and optimization recommendations  
- Admin interfaces for configuration management
- Integration with existing RouteLLM cost tracking and POML template monitoring

### RouteLLM Integration (#248) - Cost Tracking Patterns

**Available Cost Tracking:**
- **CostTracker**: Real-time cost monitoring and budget tracking
- **RoutingMetrics**: Decision tracking with cost impact analysis
- **OptimizationAnalytics**: Cost savings analysis and recommendations
- **BudgetMonitoring**: Budget utilization and alerting

**Integration Points:**
- Cost metrics collection during routing decisions
- Real-time cost impact tracking for different model selections
- Budget threshold monitoring with configurable alerts
- Cost optimization recommendations based on historical data

### POML Integration (#250) - Template Monitoring Patterns

**Available Template Monitoring:**
- **TemplateMetrics**: Processing time and success rate tracking
- **ValidationReporting**: Template validation results and error analysis
- **MigrationAnalytics**: Migration success rates and compatibility analysis
- **PerformanceTracking**: Template processing performance metrics

**Integration Points:**
- Template processing performance monitoring
- Validation result aggregation and reporting
- Migration success tracking and optimization recommendations
- Error pattern analysis for template processing

## Configuration Management Architecture

### 1. External Tool Configuration System

**ExternalToolConfig Class:**
```python
@dataclass
class ExternalToolConfig(BaseWrapperConfig):
    """Configuration for external tool integrations."""
    
    # API Configuration
    api_endpoint: str = ""
    api_key: str = ""
    api_version: str = "v1"
    
    # Authentication
    auth_type: str = "bearer"  # bearer, api_key, oauth
    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[str] = None
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = 60
    rate_limit_tokens_per_minute: int = 10000
    rate_limit_burst_size: int = 10
    
    # Connection Settings
    connection_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 30.0
    max_connections: int = 100
    max_keepalive_connections: int = 20
    
    # Retry Configuration
    retry_backoff_factor: float = 2.0
    retry_max_delay_seconds: float = 60.0
    retry_jitter: bool = True
    
    # Health Check Settings
    health_check_enabled: bool = True
    health_check_interval_seconds: int = 30
    health_check_endpoint: str = "/health"
    health_check_timeout_seconds: float = 5.0
    
    def get_config_fields(self) -> Dict[str, ConfigField]:
        """Get external tool specific configuration fields."""
        return {
            "api_endpoint": ConfigField(
                "api_endpoint", str, "", 
                "API endpoint URL", 
                required=True,
                validator=lambda x: x.startswith(('http://', 'https://'))
            ),
            "api_key": ConfigField(
                "api_key", str, "", 
                "API key for authentication",
                sensitive=True,
                environment_var="EXTERNAL_API_KEY"
            ),
            "rate_limit_requests_per_minute": ConfigField(
                "rate_limit_requests_per_minute", int, 60,
                "Maximum requests per minute",
                min_value=1, max_value=10000
            ),
            "connection_timeout_seconds": ConfigField(
                "connection_timeout_seconds", float, 10.0,
                "Connection timeout in seconds",
                min_value=1.0, max_value=300.0
            ),
            # ... additional field definitions
        }
```

**Credential Management System:**
```python
import keyring
from cryptography.fernet import Fernet
from dataclasses import dataclass
from typing import Dict, Optional, Any

@dataclass
class CredentialConfig:
    """Configuration for credential management."""
    
    storage_backend: str = "keyring"  # keyring, file, environment
    encryption_enabled: bool = True
    key_rotation_days: int = 90
    audit_logging: bool = True

class CredentialManager:
    """Secure credential management for external tools."""
    
    def __init__(self, config: CredentialConfig):
        self.config = config
        self._encryption_key = self._get_or_create_encryption_key()
        self._audit_log: List[Dict[str, Any]] = []
    
    def store_credential(
        self, 
        service: str, 
        key: str, 
        value: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store encrypted credential securely."""
        try:
            # Encrypt the credential if encryption is enabled
            if self.config.encryption_enabled:
                cipher_suite = Fernet(self._encryption_key)
                encrypted_value = cipher_suite.encrypt(value.encode()).decode()
            else:
                encrypted_value = value
            
            # Store using configured backend
            if self.config.storage_backend == "keyring":
                keyring.set_password(service, key, encrypted_value)
            elif self.config.storage_backend == "environment":
                os.environ[f"{service.upper()}_{key.upper()}"] = encrypted_value
            elif self.config.storage_backend == "file":
                self._store_to_secure_file(service, key, encrypted_value)
            
            # Audit logging
            if self.config.audit_logging:
                self._audit_log.append({
                    "action": "store",
                    "service": service,
                    "key": key,
                    "timestamp": datetime.utcnow(),
                    "metadata": metadata or {}
                })
            
            logger.info(f"Stored credential for {service}/{key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store credential for {service}/{key}: {e}")
            return False
    
    def retrieve_credential(
        self, 
        service: str, 
        key: str
    ) -> Optional[str]:
        """Retrieve and decrypt credential."""
        try:
            # Retrieve using configured backend
            if self.config.storage_backend == "keyring":
                encrypted_value = keyring.get_password(service, key)
            elif self.config.storage_backend == "environment":
                encrypted_value = os.environ.get(f"{service.upper()}_{key.upper()}")
            elif self.config.storage_backend == "file":
                encrypted_value = self._retrieve_from_secure_file(service, key)
            else:
                return None
            
            if not encrypted_value:
                return None
            
            # Decrypt if encryption is enabled
            if self.config.encryption_enabled:
                cipher_suite = Fernet(self._encryption_key)
                decrypted_value = cipher_suite.decrypt(encrypted_value.encode()).decode()
            else:
                decrypted_value = encrypted_value
            
            # Audit logging
            if self.config.audit_logging:
                self._audit_log.append({
                    "action": "retrieve",
                    "service": service,
                    "key": key,
                    "timestamp": datetime.utcnow()
                })
            
            return decrypted_value
            
        except Exception as e:
            logger.error(f"Failed to retrieve credential for {service}/{key}: {e}")
            return None
    
    def rotate_credentials(self) -> Dict[str, bool]:
        """Rotate credentials that are due for rotation."""
        results = {}
        
        # Implementation would check credential ages and rotate as needed
        # This would integrate with external tool APIs to generate new credentials
        
        return results
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for credential storage."""
        # Implementation would securely generate/store the encryption key
        return Fernet.generate_key()
```

### 2. Environment Configuration Management

**EnvironmentConfig System:**
```python
from enum import Enum
from typing import Dict, List, Any, Optional

class Environment(Enum):
    """Supported deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class EnvironmentOverride:
    """Environment-specific configuration override."""
    
    environment: Environment
    config_path: str  # dot notation: "wrapper.api.timeout_seconds"
    value: Any
    reason: str = ""
    expires_at: Optional[datetime] = None

class EnvironmentConfigManager:
    """Manages environment-specific configuration overrides."""
    
    def __init__(self, current_environment: Environment):
        self.current_environment = current_environment
        self._overrides: Dict[str, List[EnvironmentOverride]] = {}
        self._base_configs: Dict[str, BaseWrapperConfig] = {}
    
    def register_base_config(
        self, 
        wrapper_name: str, 
        config: BaseWrapperConfig
    ) -> None:
        """Register a base configuration for environment overrides."""
        self._base_configs[wrapper_name] = config
        
        # Load environment-specific overrides
        self._load_environment_overrides(wrapper_name)
    
    def add_override(
        self, 
        wrapper_name: str, 
        override: EnvironmentOverride
    ) -> None:
        """Add an environment-specific override."""
        if wrapper_name not in self._overrides:
            self._overrides[wrapper_name] = []
        
        self._overrides[wrapper_name].append(override)
        
        # Apply override immediately if it matches current environment
        if override.environment == self.current_environment:
            self._apply_override(wrapper_name, override)
    
    def get_effective_config(
        self, 
        wrapper_name: str
    ) -> Optional[BaseWrapperConfig]:
        """Get configuration with environment overrides applied."""
        if wrapper_name not in self._base_configs:
            return None
        
        # Start with base configuration
        config = copy.deepcopy(self._base_configs[wrapper_name])
        
        # Apply environment-specific overrides
        overrides = self._overrides.get(wrapper_name, [])
        for override in overrides:
            if override.environment == self.current_environment:
                if not override.expires_at or override.expires_at > datetime.utcnow():
                    self._apply_override_to_config(config, override)
        
        return config
    
    def _apply_override_to_config(
        self, 
        config: BaseWrapperConfig, 
        override: EnvironmentOverride
    ) -> None:
        """Apply a single override to a configuration object."""
        path_parts = override.config_path.split('.')
        obj = config
        
        # Navigate to the parent object
        for part in path_parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                logger.warning(f"Invalid override path: {override.config_path}")
                return
        
        # Set the final value
        final_key = path_parts[-1]
        if hasattr(obj, final_key):
            setattr(obj, final_key, override.value)
            logger.debug(f"Applied override: {override.config_path} = {override.value}")
        else:
            logger.warning(f"Invalid override key: {final_key}")
```

## Performance Monitoring Architecture

### 1. Web-based Monitoring Dashboards

**DashboardBuilder System:**
```python
from flask import Flask, render_template, jsonify, request
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.utils

@dataclass
class DashboardConfig:
    """Configuration for monitoring dashboards."""
    
    refresh_interval_seconds: int = 30
    max_data_points: int = 1000
    enable_real_time: bool = True
    themes: List[str] = field(default_factory=lambda: ["light", "dark"])
    default_theme: str = "light"

class MonitoringDashboard:
    """Web-based monitoring dashboard for wrapper performance."""
    
    def __init__(
        self, 
        monitoring: WrapperMonitoring,
        config_manager: ConfigurationManager,
        config: DashboardConfig
    ):
        self.monitoring = monitoring
        self.config_manager = config_manager
        self.config = config
        self.app = Flask(__name__)
        
        # Register routes
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Setup Flask routes for dashboard."""
        
        @self.app.route('/')
        def dashboard_home():
            """Main dashboard page."""
            return render_template('dashboard.html', config=self.config)
        
        @self.app.route('/api/system-health')
        def get_system_health():
            """Get overall system health metrics."""
            health_data = self.monitoring.get_system_health()
            return jsonify(health_data)
        
        @self.app.route('/api/wrapper-stats/<wrapper_name>')
        def get_wrapper_stats(wrapper_name: str):
            """Get detailed statistics for a specific wrapper."""
            stats = self.monitoring.get_wrapper_stats(wrapper_name)
            health = self.monitoring.get_wrapper_health(wrapper_name)
            
            return jsonify({
                "stats": stats,
                "health": asdict(health),
                "charts": self._generate_wrapper_charts(wrapper_name)
            })
        
        @self.app.route('/api/alerts')
        def get_active_alerts():
            """Get current active alerts."""
            severity_filter = request.args.get('severity')
            wrapper_filter = request.args.get('wrapper')
            
            alerts = self.monitoring.get_alerts(
                wrapper_name=wrapper_filter,
                severity=AlertSeverity(severity_filter) if severity_filter else None,
                include_resolved=False
            )
            
            return jsonify([asdict(alert) for alert in alerts])
        
        @self.app.route('/api/performance-metrics')
        def get_performance_metrics():
            """Get performance metrics for charting."""
            wrapper_name = request.args.get('wrapper')
            time_range = request.args.get('range', '24h')
            
            metrics = self._get_performance_metrics(wrapper_name, time_range)
            return jsonify(metrics)
        
        @self.app.route('/api/cost-analysis')
        def get_cost_analysis():
            """Get cost analysis data."""
            wrapper_name = request.args.get('wrapper')
            time_range = request.args.get('range', '7d')
            
            cost_data = self._get_cost_analysis(wrapper_name, time_range)
            return jsonify(cost_data)
    
    def _generate_wrapper_charts(self, wrapper_name: str) -> Dict[str, Any]:
        """Generate chart data for a specific wrapper."""
        metrics = self.monitoring.export_metrics(
            wrapper_name=wrapper_name,
            start_time=datetime.utcnow() - timedelta(hours=24)
        )
        
        if not metrics:
            return {}
        
        # Success rate over time
        success_chart = self._create_success_rate_chart(metrics)
        
        # Response time distribution
        response_time_chart = self._create_response_time_chart(metrics)
        
        # Error frequency chart
        error_chart = self._create_error_frequency_chart(metrics)
        
        # Fallback usage chart
        fallback_chart = self._create_fallback_usage_chart(metrics)
        
        return {
            "success_rate": success_chart,
            "response_time": response_time_chart,
            "error_frequency": error_chart,
            "fallback_usage": fallback_chart
        }
    
    def _create_success_rate_chart(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create success rate chart data."""
        # Group metrics by hour
        hourly_data = defaultdict(list)
        
        for metric in metrics:
            hour = datetime.fromisoformat(metric['start_time']).replace(minute=0, second=0, microsecond=0)
            hourly_data[hour].append(metric['success'])
        
        timestamps = sorted(hourly_data.keys())
        success_rates = [
            sum(hourly_data[ts]) / len(hourly_data[ts]) * 100 
            for ts in timestamps
        ]
        
        fig = go.Figure(data=go.Scatter(
            x=timestamps,
            y=success_rates,
            mode='lines+markers',
            name='Success Rate %',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title="Success Rate Over Time",
            xaxis_title="Time",
            yaxis_title="Success Rate (%)",
            yaxis=dict(range=[0, 100])
        )
        
        return plotly.utils.PlotlyJSONEncoder().encode(fig)
```

### 2. Alert System Integration

**Enhanced AlertManager:**
```python
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional
import smtplib
import requests
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

@dataclass
class NotificationChannel:
    """Configuration for alert notification channels."""
    
    name: str
    type: str  # email, slack, webhook, sms
    config: Dict[str, Any]
    enabled: bool = True
    severity_filter: Optional[List[AlertSeverity]] = None

class AlertManager:
    """Enhanced alert management with multiple notification channels."""
    
    def __init__(self, monitoring: WrapperMonitoring):
        self.monitoring = monitoring
        self._notification_channels: Dict[str, NotificationChannel] = {}
        self._alert_templates: Dict[str, str] = {}
        
        # Load default alert templates
        self._setup_default_templates()
    
    def add_notification_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel for alerts."""
        self._notification_channels[channel.name] = channel
        logger.info(f"Added notification channel: {channel.name} ({channel.type})")
    
    def send_alert(self, alert: Alert) -> Dict[str, bool]:
        """Send alert through configured notification channels."""
        results = {}
        
        for channel_name, channel in self._notification_channels.items():
            if not channel.enabled:
                continue
            
            # Check severity filter
            if channel.severity_filter and alert.severity not in channel.severity_filter:
                continue
            
            try:
                success = self._send_alert_to_channel(alert, channel)
                results[channel_name] = success
                
                if success:
                    logger.info(f"Alert sent via {channel_name}: {alert.id}")
                else:
                    logger.error(f"Failed to send alert via {channel_name}: {alert.id}")
                    
            except Exception as e:
                logger.error(f"Exception sending alert via {channel_name}: {e}")
                results[channel_name] = False
        
        return results
    
    def _send_alert_to_channel(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send alert to a specific notification channel."""
        if channel.type == "email":
            return self._send_email_alert(alert, channel)
        elif channel.type == "slack":
            return self._send_slack_alert(alert, channel)
        elif channel.type == "webhook":
            return self._send_webhook_alert(alert, channel)
        elif channel.type == "sms":
            return self._send_sms_alert(alert, channel)
        else:
            logger.error(f"Unknown notification channel type: {channel.type}")
            return False
    
    def _send_email_alert(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send alert via email."""
        try:
            smtp_server = channel.config.get('smtp_server', 'localhost')
            smtp_port = channel.config.get('smtp_port', 587)
            username = channel.config.get('username', '')
            password = channel.config.get('password', '')
            from_email = channel.config.get('from_email', username)
            to_emails = channel.config.get('to_emails', [])
            
            if not to_emails:
                logger.error("No recipient emails configured for email channel")
                return False
            
            # Create message
            msg = MimeMultipart()
            msg['From'] = from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] Wrapper Alert: {alert.wrapper_name}"
            
            # Format alert message
            template = self._alert_templates.get('email', self._alert_templates['default'])
            body = template.format(
                alert_id=alert.id,
                wrapper_name=alert.wrapper_name,
                severity=alert.severity.value.upper(),
                message=alert.message,
                timestamp=alert.timestamp.isoformat(),
                rule_name=alert.rule_name
            )
            
            msg.attach(MimeText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            if username and password:
                server.starttls()
                server.login(username, password)
            
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _send_slack_alert(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send alert via Slack webhook."""
        try:
            webhook_url = channel.config.get('webhook_url', '')
            channel_name = channel.config.get('channel', '#alerts')
            
            if not webhook_url:
                logger.error("No webhook URL configured for Slack channel")
                return False
            
            # Format Slack message
            color = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning", 
                AlertSeverity.ERROR: "danger",
                AlertSeverity.CRITICAL: "danger"
            }.get(alert.severity, "warning")
            
            payload = {
                "channel": channel_name,
                "username": "Wrapper Monitor",
                "text": f"Wrapper Alert: {alert.wrapper_name}",
                "attachments": [
                    {
                        "color": color,
                        "title": f"{alert.severity.value.upper()}: {alert.rule_name}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Wrapper",
                                "value": alert.wrapper_name,
                                "short": True
                            },
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.isoformat(),
                                "short": True
                            },
                            {
                                "title": "Alert ID",
                                "value": alert.id,
                                "short": True
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=payload)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
```

### 3. Cost Tracking Integration

**CostMonitoringIntegration:**
```python
from typing import Dict, List, Any, Optional
from decimal import Decimal
import json

@dataclass
class CostMetrics:
    """Cost metrics for wrapper operations."""
    
    operation_id: str
    wrapper_name: str
    timestamp: datetime
    
    # Cost breakdown
    api_cost: Decimal = Decimal('0.0')
    infrastructure_cost: Decimal = Decimal('0.0')
    total_cost: Decimal = Decimal('0.0')
    
    # Usage metrics
    tokens_used: int = 0
    requests_made: int = 0
    compute_time_ms: float = 0.0
    
    # Optimization metrics
    cost_savings: Decimal = Decimal('0.0')
    efficiency_score: float = 1.0
    
    # Context
    model_used: Optional[str] = None
    optimization_decision: Optional[str] = None

class CostMonitoringIntegration:
    """Integration with external cost tracking systems."""
    
    def __init__(
        self, 
        monitoring: WrapperMonitoring,
        routellm_cost_tracker: Optional[Any] = None
    ):
        self.monitoring = monitoring
        self.routellm_cost_tracker = routellm_cost_tracker
        self._cost_metrics: List[CostMetrics] = []
        self._cost_budgets: Dict[str, Dict[str, Any]] = {}
    
    def record_operation_cost(
        self,
        operation_id: str,
        wrapper_name: str,
        cost_data: Dict[str, Any]
    ) -> None:
        """Record cost metrics for a wrapper operation."""
        
        cost_metrics = CostMetrics(
            operation_id=operation_id,
            wrapper_name=wrapper_name,
            timestamp=datetime.utcnow(),
            api_cost=Decimal(str(cost_data.get('api_cost', '0.0'))),
            infrastructure_cost=Decimal(str(cost_data.get('infrastructure_cost', '0.0'))),
            total_cost=Decimal(str(cost_data.get('total_cost', '0.0'))),
            tokens_used=cost_data.get('tokens_used', 0),
            requests_made=cost_data.get('requests_made', 0),
            compute_time_ms=cost_data.get('compute_time_ms', 0.0),
            cost_savings=Decimal(str(cost_data.get('cost_savings', '0.0'))),
            efficiency_score=cost_data.get('efficiency_score', 1.0),
            model_used=cost_data.get('model_used'),
            optimization_decision=cost_data.get('optimization_decision')
        )
        
        self._cost_metrics.append(cost_metrics)
        
        # Update wrapper monitoring with cost information
        self.monitoring._active_operations[operation_id].cost_estimate = float(cost_metrics.total_cost)
        
        # Check budget thresholds
        self._check_budget_thresholds(wrapper_name, cost_metrics)
        
        logger.debug(f"Recorded cost metrics for operation {operation_id}: ${cost_metrics.total_cost}")
    
    def get_cost_analysis(
        self, 
        wrapper_name: Optional[str] = None,
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """Get cost analysis for wrapper operations."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
        
        # Filter metrics
        filtered_metrics = [
            m for m in self._cost_metrics 
            if m.timestamp >= cutoff_time and (not wrapper_name or m.wrapper_name == wrapper_name)
        ]
        
        if not filtered_metrics:
            return {"total_cost": 0.0, "operations": 0, "average_cost": 0.0}
        
        # Calculate aggregated metrics
        total_cost = sum(m.total_cost for m in filtered_metrics)
        total_savings = sum(m.cost_savings for m in filtered_metrics)
        total_operations = len(filtered_metrics)
        average_cost = total_cost / total_operations
        
        # Cost by wrapper
        cost_by_wrapper = defaultdict(Decimal)
        for metric in filtered_metrics:
            cost_by_wrapper[metric.wrapper_name] += metric.total_cost
        
        # Cost over time (hourly buckets)
        hourly_costs = defaultdict(Decimal)
        for metric in filtered_metrics:
            hour = metric.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_costs[hour] += metric.total_cost
        
        return {
            "total_cost": float(total_cost),
            "total_savings": float(total_savings),
            "operations": total_operations,
            "average_cost": float(average_cost),
            "cost_by_wrapper": {k: float(v) for k, v in cost_by_wrapper.items()},
            "hourly_costs": {k.isoformat(): float(v) for k, v in hourly_costs.items()},
            "efficiency_metrics": {
                "average_efficiency_score": sum(m.efficiency_score for m in filtered_metrics) / total_operations,
                "total_tokens_used": sum(m.tokens_used for m in filtered_metrics),
                "total_requests": sum(m.requests_made for m in filtered_metrics)
            }
        }
    
    def set_budget(
        self, 
        wrapper_name: str, 
        daily_budget: Decimal, 
        monthly_budget: Decimal,
        alert_thresholds: Dict[str, float] = None
    ) -> None:
        """Set cost budget for a wrapper with alert thresholds."""
        
        alert_thresholds = alert_thresholds or {"warning": 0.8, "critical": 0.95}
        
        self._cost_budgets[wrapper_name] = {
            "daily_budget": daily_budget,
            "monthly_budget": monthly_budget,
            "alert_thresholds": alert_thresholds,
            "last_reset": datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        }
        
        logger.info(f"Set budget for {wrapper_name}: Daily ${daily_budget}, Monthly ${monthly_budget}")
    
    def _check_budget_thresholds(self, wrapper_name: str, cost_metrics: CostMetrics) -> None:
        """Check if cost metrics exceed budget thresholds."""
        
        if wrapper_name not in self._cost_budgets:
            return
        
        budget_config = self._cost_budgets[wrapper_name]
        
        # Calculate current daily and monthly spending
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        daily_spending = sum(
            m.total_cost for m in self._cost_metrics 
            if m.wrapper_name == wrapper_name and m.timestamp >= today_start
        )
        
        monthly_spending = sum(
            m.total_cost for m in self._cost_metrics 
            if m.wrapper_name == wrapper_name and m.timestamp >= month_start
        )
        
        # Check thresholds
        for threshold_name, threshold_pct in budget_config["alert_thresholds"].items():
            daily_threshold = budget_config["daily_budget"] * Decimal(str(threshold_pct))
            monthly_threshold = budget_config["monthly_budget"] * Decimal(str(threshold_pct))
            
            if daily_spending >= daily_threshold:
                self._trigger_budget_alert(
                    wrapper_name, 
                    "daily", 
                    daily_spending, 
                    budget_config["daily_budget"],
                    threshold_name
                )
            
            if monthly_spending >= monthly_threshold:
                self._trigger_budget_alert(
                    wrapper_name, 
                    "monthly", 
                    monthly_spending, 
                    budget_config["monthly_budget"],
                    threshold_name
                )
    
    def _trigger_budget_alert(
        self, 
        wrapper_name: str, 
        period: str, 
        current_spending: Decimal, 
        budget: Decimal, 
        threshold_name: str
    ) -> None:
        """Trigger budget threshold alert."""
        
        percentage = (current_spending / budget) * 100
        severity = AlertSeverity.WARNING if threshold_name == "warning" else AlertSeverity.CRITICAL
        
        alert = Alert(
            wrapper_name=wrapper_name,
            rule_name=f"budget_{threshold_name}",
            severity=severity,
            message=f"Budget {threshold_name} threshold exceeded: {percentage:.1f}% of {period} budget (${current_spending}/${budget})"
        )
        
        # Add to monitoring system's alerts
        self.monitoring._alerts.append(alert)
        
        logger.warning(f"Budget alert triggered for {wrapper_name}: {alert.message}")
```

## Implementation Plan

### Phase 1: Core Configuration Extensions (Days 1-3)

**Day 1:**
1. Extend `wrapper_config.py` with external tool configuration classes
2. Implement credential management system with encryption
3. Add environment configuration management
4. Create configuration validation and migration tools

**Day 2:**
1. Enhance existing monitoring with cost tracking integration
2. Create performance analytics and trend analysis
3. Implement advanced alerting with multiple notification channels
4. Add budget monitoring and threshold alerts

**Day 3:**
1. Create web-based monitoring dashboard framework
2. Implement real-time metrics visualization
3. Add interactive charts and performance analytics
4. Create responsive dashboard UI with theme support

### Phase 2: Admin Interface Development (Days 4-6)

**Day 4:**
1. Create admin interface for configuration management
2. Implement configuration editor with validation
3. Add credential management interface
4. Create environment configuration tools

**Day 5:**
1. Build monitoring dashboard with multiple views
2. Implement alert management interface
3. Add cost analysis and budget management tools
4. Create system health overview dashboard

**Day 6:**
1. Integrate with existing RouteLLM cost tracking
2. Connect with POML template monitoring
3. Add wrapper-specific monitoring extensions
4. Implement unified metrics aggregation

### Phase 3: Testing and Documentation (Days 7-8)

**Day 7:**
1. Create comprehensive test suite for all components
2. Implement integration tests with existing wrappers
3. Add performance testing and benchmarking
4. Create security testing for credential management

**Day 8:**
1. Write comprehensive documentation
2. Create admin user guides and API documentation
3. Implement example configurations and use cases
4. Finalize deployment and migration guides

## File Structure

```
src/orchestrator/core/
├── wrapper_config.py            # Extended with external tool configs
├── wrapper_monitoring.py        # Enhanced with advanced analytics
├── credential_manager.py        # New - secure credential management
└── environment_config.py        # New - environment-specific overrides

src/orchestrator/web/
├── __init__.py                  # New directory
├── monitoring_dashboard.py      # New - web dashboard framework
├── dashboard_routes.py          # New - Flask routes for dashboards
├── static/                      # New - CSS/JS for dashboard
└── templates/                   # New - HTML templates

src/orchestrator/admin/
├── __init__.py                  # New directory
├── config_manager.py           # New - admin interface for configs
├── alert_manager.py            # New - alert management interface
└── cost_manager.py             # New - cost analysis and budgets

src/orchestrator/integrations/
├── cost_monitoring.py          # New - cost tracking integration
├── routellm_monitoring.py      # New - RouteLLM specific monitoring
└── poml_monitoring.py          # New - POML specific monitoring

tests/core/monitoring/
├── test_credential_manager.py   # New - credential management tests
├── test_environment_config.py  # New - environment config tests
├── test_cost_monitoring.py     # New - cost tracking tests
└── test_dashboard_integration.py # New - dashboard integration tests
```

## Success Criteria

1. **External Tool Configuration**: Centralized configuration for all external integrations
2. **Secure Credential Management**: Encrypted credential storage with rotation capabilities  
3. **Real-time Monitoring**: Web-based dashboards with live performance metrics
4. **Cost Tracking Integration**: Seamless integration with RouteLLM cost optimization
5. **Alert System**: Multi-channel alerting with configurable rules and thresholds
6. **Admin Interface**: Comprehensive admin tools for configuration and monitoring
7. **Budget Management**: Cost budgets with threshold monitoring and alerts
8. **Performance Analytics**: Trend analysis and optimization recommendations

## Risk Mitigation

### Security Considerations
1. **Credential Security**: Multi-layer encryption for API keys and sensitive data
2. **Access Control**: Role-based access for admin interfaces
3. **Audit Logging**: Comprehensive audit trails for all configuration changes
4. **Network Security**: HTTPS enforcement and secure communication channels

### Performance Considerations  
1. **Monitoring Overhead**: Efficient metrics collection with minimal performance impact
2. **Dashboard Performance**: Optimized queries and caching for real-time dashboards
3. **Memory Management**: Proper cleanup and retention policies for metrics data
4. **Scalability**: Design for horizontal scaling of monitoring infrastructure

### Operational Considerations
1. **Backward Compatibility**: Zero breaking changes to existing configurations
2. **Gradual Rollout**: Feature flags for safe deployment of new capabilities
3. **Migration Support**: Tools and guides for migrating existing configurations
4. **Disaster Recovery**: Backup and recovery procedures for configuration data

This comprehensive configuration and monitoring system will provide full operational visibility and control over all wrapper integrations, enabling efficient management, cost optimization, and proactive issue resolution.