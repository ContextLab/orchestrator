"""
Quality Control Reporting & Analytics System

This module provides comprehensive quality reporting, analytics, and alerting
capabilities for the orchestrator quality control system. It builds on the
validation data from Stream A and logging infrastructure from Stream B to
deliver actionable quality insights.

Key Components:
- Metrics Collection: Aggregates quality data from validation sessions
- Analytics Engine: Provides trend analysis and actionable insights  
- Alerting System: Monitors quality thresholds and sends notifications
- Quality Dashboard: Web interface for monitoring and insights visualization

Integration Points:
- Validation Engine: Consumes validation session results
- Structured Logger: Integrates with quality event logging
- Execution Context: Tracks quality across pipeline executions
"""

from .metrics import (
    QualityMetricsCollector,
    MetricType,
    QualityMetric,
    MetricsSnapshot,
    TimeSeriesMetric
)

from .analytics import (
    QualityAnalytics,
    TrendAnalysis,
    QualityInsight,
    InsightType,
    TrendDirection,
    AnalyticsResult
)

from .alerts import (
    QualityAlertSystem,
    AlertRule,
    AlertSeverity,
    AlertType,
    AlertCondition,
    AlertNotification,
    AlertChannel
)

from .dashboard import (
    QualityDashboard,
    DashboardWidget,
    WidgetType,
    DashboardConfig
)

__all__ = [
    # Metrics
    'QualityMetricsCollector',
    'MetricType', 
    'QualityMetric',
    'MetricsSnapshot',
    'TimeSeriesMetric',
    
    # Analytics
    'QualityAnalytics',
    'TrendAnalysis',
    'QualityInsight',
    'InsightType',
    'TrendDirection', 
    'AnalyticsResult',
    
    # Alerts
    'QualityAlertSystem',
    'AlertRule',
    'AlertSeverity',
    'AlertType',
    'AlertCondition',
    'AlertNotification',
    'AlertChannel',
    
    # Dashboard
    'QualityDashboard',
    'DashboardWidget',
    'WidgetType',
    'DashboardConfig'
]