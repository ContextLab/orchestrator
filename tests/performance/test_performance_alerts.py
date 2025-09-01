#!/usr/bin/env python3
"""
Performance monitoring and alerting system for CI/CD integration.

Implements performance regression detection, alerting, and continuous 
monitoring capabilities for the orchestrator across different platforms.
"""

import asyncio
import json
import logging
import statistics
import time
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pytest

logger = logging.getLogger(__name__)

# Add orchestrator to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""
    
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    unit: str
    comparison: str = "greater_than"  # greater_than, less_than, absolute_diff


@dataclass
class PerformanceAlert:
    """Performance alert details."""
    
    alert_id: str
    severity: str  # info, warning, critical
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    platform: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceTrend:
    """Performance trend analysis."""
    
    metric_name: str
    platform: str
    measurements: List[float]
    trend_direction: str  # improving, degrading, stable
    trend_slope: float
    confidence: float
    period_days: int


class PerformanceMonitor:
    """
    Monitors performance metrics and generates alerts for CI/CD integration.
    
    Tracks performance trends, detects regressions, and provides
    actionable alerts for continuous integration pipelines.
    """

    def __init__(self):
        self.results_dir = Path("tests/performance/results")
        self.alerts_dir = Path("tests/performance/alerts")
        self.history_dir = Path("tests/performance/history")
        
        # Ensure directories exist
        for directory in [self.results_dir, self.alerts_dir, self.history_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Performance thresholds (configurable)
        self.thresholds = [
            PerformanceThreshold("execution_time_ms", 5000, 10000, "ms", "greater_than"),
            PerformanceThreshold("memory_usage_mb", 500, 1000, "MB", "greater_than"),
            PerformanceThreshold("cpu_usage_percent", 80, 95, "%", "greater_than"),
            PerformanceThreshold("success_rate", 0.8, 0.6, "ratio", "less_than")
        ]
        
        self.alerts = []
        self.trends = {}
        
    def load_historical_data(self, days_back: int = 30) -> Dict[str, List[Dict[str, Any]]]:
        """Load historical performance data."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        historical_data = {}
        
        # Load from performance result files
        for result_file in self.results_dir.glob("*.json"):
            try:
                with open(result_file) as f:
                    data = json.load(f)
                
                # Check if data is recent enough
                if "test_summary" in data and "test_date" in data["test_summary"]:
                    test_date = datetime.fromisoformat(data["test_summary"]["test_date"].replace("Z", "+00:00"))
                    if test_date < cutoff_date:
                        continue
                
                # Extract platform and metrics
                platform = data.get("platform_info", {}).get("system", "unknown")
                
                if platform not in historical_data:
                    historical_data[platform] = []
                
                historical_data[platform].append(data)
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to load historical data from {result_file}: {e}")
                continue
        
        return historical_data
    
    def analyze_performance_trends(self, historical_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, PerformanceTrend]:
        """Analyze performance trends from historical data."""
        trends = {}
        
        for platform, data_points in historical_data.items():
            if len(data_points) < 3:  # Need at least 3 points for trend analysis
                continue
                
            # Sort by date
            data_points.sort(key=lambda x: x.get("test_summary", {}).get("test_date", ""))
            
            # Analyze different metrics
            metrics_to_analyze = [
                ("execution_time", "performance_statistics.execution_time.mean_ms"),
                ("memory_usage", "performance_statistics.memory_usage.mean_mb"),
                ("cpu_usage", "performance_statistics.cpu_usage.mean_percent")
            ]
            
            for metric_name, json_path in metrics_to_analyze:
                values = []
                
                for data_point in data_points:
                    try:
                        # Navigate nested dictionary
                        value = data_point
                        for key in json_path.split('.'):
                            value = value[key]
                        values.append(float(value))
                    except (KeyError, TypeError, ValueError):
                        continue
                
                if len(values) >= 3:
                    trend = self._calculate_trend(values)
                    trend_key = f"{platform}_{metric_name}"
                    
                    trends[trend_key] = PerformanceTrend(
                        metric_name=metric_name,
                        platform=platform,
                        measurements=values,
                        trend_direction=trend["direction"],
                        trend_slope=trend["slope"],
                        confidence=trend["confidence"],
                        period_days=30
                    )
        
        self.trends = trends
        return trends
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend statistics for a series of values."""
        if len(values) < 2:
            return {"direction": "stable", "slope": 0, "confidence": 0}
        
        # Simple linear regression
        n = len(values)
        x = list(range(n))
        
        # Calculate slope
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        # Determine direction and confidence
        if len(values) >= 3:
            recent_avg = statistics.mean(values[-3:])
            older_avg = statistics.mean(values[:-3]) if len(values) > 3 else values[0]
            
            relative_change = abs((recent_avg - older_avg) / max(older_avg, 1)) * 100
            
            if relative_change < 5:  # Less than 5% change
                direction = "stable"
                confidence = min(0.9, relative_change / 5)
            elif slope > 0:
                direction = "degrading"  # Assuming higher values are worse
                confidence = min(0.9, relative_change / 20)
            else:
                direction = "improving"
                confidence = min(0.9, relative_change / 20)
        else:
            direction = "stable"
            confidence = 0.5
        
        return {
            "direction": direction,
            "slope": slope,
            "confidence": confidence
        }
    
    def check_performance_thresholds(self, current_metrics: Dict[str, Any]) -> List[PerformanceAlert]:
        """Check current metrics against performance thresholds."""
        alerts = []
        platform = current_metrics.get("platform_info", {}).get("system", "unknown")
        
        for threshold in self.thresholds:
            current_value = self._extract_metric_value(current_metrics, threshold.metric_name)
            
            if current_value is None:
                continue
            
            alert_severity = None
            threshold_value = None
            
            if threshold.comparison == "greater_than":
                if current_value > threshold.critical_threshold:
                    alert_severity = "critical"
                    threshold_value = threshold.critical_threshold
                elif current_value > threshold.warning_threshold:
                    alert_severity = "warning"
                    threshold_value = threshold.warning_threshold
            elif threshold.comparison == "less_than":
                if current_value < threshold.critical_threshold:
                    alert_severity = "critical"
                    threshold_value = threshold.critical_threshold
                elif current_value < threshold.warning_threshold:
                    alert_severity = "warning"
                    threshold_value = threshold.warning_threshold
            
            if alert_severity:
                alert = PerformanceAlert(
                    alert_id=f"{platform}_{threshold.metric_name}_{int(time.time())}",
                    severity=alert_severity,
                    metric_name=threshold.metric_name,
                    current_value=current_value,
                    threshold_value=threshold_value,
                    message=f"{threshold.metric_name} {alert_severity}: {current_value:.2f}{threshold.unit} exceeds threshold of {threshold_value:.2f}{threshold.unit}",
                    platform=platform,
                    metadata={
                        "threshold_config": threshold.__dict__,
                        "measurement_context": "performance_monitoring"
                    }
                )
                alerts.append(alert)
        
        return alerts
    
    def _extract_metric_value(self, metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Extract metric value from performance data."""
        try:
            if metric_name == "execution_time_ms":
                return metrics["performance_statistics"]["execution_time"]["mean_ms"]
            elif metric_name == "memory_usage_mb":
                return metrics["performance_statistics"]["memory_usage"]["mean_mb"]
            elif metric_name == "cpu_usage_percent":
                return metrics["performance_statistics"]["cpu_usage"]["mean_percent"]
            elif metric_name == "success_rate":
                # Calculate success rate from detailed results
                detailed = metrics.get("detailed_results", [])
                if detailed:
                    # Assume success if execution time is reasonable
                    successful = sum(1 for result in detailed if result.get("execution_time_ms", 0) < 30000)
                    return successful / len(detailed)
        except (KeyError, TypeError, ZeroDivisionError):
            pass
        
        return None
    
    def check_trend_alerts(self) -> List[PerformanceAlert]:
        """Generate alerts based on performance trends."""
        alerts = []
        
        for trend_key, trend in self.trends.items():
            if trend.trend_direction == "degrading" and trend.confidence > 0.7:
                alert = PerformanceAlert(
                    alert_id=f"trend_{trend_key}_{int(time.time())}",
                    severity="warning" if trend.confidence < 0.9 else "critical",
                    metric_name=trend.metric_name,
                    current_value=trend.measurements[-1] if trend.measurements else 0,
                    threshold_value=trend.measurements[0] if trend.measurements else 0,
                    message=f"Performance degradation trend detected for {trend.metric_name} on {trend.platform} (confidence: {trend.confidence:.1%})",
                    platform=trend.platform,
                    metadata={
                        "trend_data": trend.__dict__,
                        "measurement_context": "trend_analysis"
                    }
                )
                alerts.append(alert)
        
        return alerts
    
    def generate_ci_cd_report(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate CI/CD integration report."""
        # Load historical data and analyze trends
        historical_data = self.load_historical_data()
        trends = self.analyze_performance_trends(historical_data)
        
        # Check thresholds and trends
        threshold_alerts = self.check_performance_thresholds(current_metrics)
        trend_alerts = self.check_trend_alerts()
        
        all_alerts = threshold_alerts + trend_alerts
        self.alerts.extend(all_alerts)
        
        # Generate report
        report = {
            "report_info": {
                "timestamp": datetime.utcnow().isoformat(),
                "platform": current_metrics.get("platform_info", {}).get("system", "unknown"),
                "report_type": "ci_cd_performance_check"
            },
            "current_metrics": {
                "execution_time_ms": self._extract_metric_value(current_metrics, "execution_time_ms"),
                "memory_usage_mb": self._extract_metric_value(current_metrics, "memory_usage_mb"),
                "cpu_usage_percent": self._extract_metric_value(current_metrics, "cpu_usage_percent"),
                "success_rate": self._extract_metric_value(current_metrics, "success_rate")
            },
            "alerts": [
                {
                    "id": alert.alert_id,
                    "severity": alert.severity,
                    "metric": alert.metric_name,
                    "current_value": alert.current_value,
                    "threshold_value": alert.threshold_value,
                    "message": alert.message,
                    "platform": alert.platform
                }
                for alert in all_alerts
            ],
            "trends": {
                trend_key: {
                    "metric": trend.metric_name,
                    "platform": trend.platform,
                    "direction": trend.trend_direction,
                    "confidence": trend.confidence,
                    "recent_values": trend.measurements[-5:] if len(trend.measurements) >= 5 else trend.measurements
                }
                for trend_key, trend in trends.items()
            },
            "summary": {
                "total_alerts": len(all_alerts),
                "critical_alerts": len([a for a in all_alerts if a.severity == "critical"]),
                "warning_alerts": len([a for a in all_alerts if a.severity == "warning"]),
                "overall_status": "FAIL" if any(a.severity == "critical" for a in all_alerts) else 
                                 "WARN" if any(a.severity == "warning" for a in all_alerts) else "PASS",
                "performance_trends_analyzed": len(trends)
            }
        }
        
        # Save report
        report_filename = f"ci_cd_performance_report_{int(time.time())}.json"
        report_path = self.alerts_dir / report_filename
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def save_performance_history(self, metrics: Dict[str, Any]):
        """Save current metrics to performance history."""
        history_filename = f"performance_history_{int(time.time())}.json"
        history_path = self.history_dir / history_filename
        
        history_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "platform": metrics.get("platform_info", {}).get("system", "unknown")
        }
        
        with open(history_path, "w") as f:
            json.dump(history_entry, f, indent=2, default=str)
    
    def print_alert_summary(self):
        """Print alert summary to console."""
        if not self.alerts:
            print("✅ No performance alerts detected")
            return
        
        print(f"\n{'='*60}")
        print("PERFORMANCE MONITORING ALERT SUMMARY")
        print(f"{'='*60}")
        
        critical_alerts = [a for a in self.alerts if a.severity == "critical"]
        warning_alerts = [a for a in self.alerts if a.severity == "warning"]
        
        if critical_alerts:
            print(f"\n🚨 CRITICAL ALERTS ({len(critical_alerts)}):")
            for alert in critical_alerts:
                print(f"  • {alert.message}")
        
        if warning_alerts:
            print(f"\n⚠️  WARNING ALERTS ({len(warning_alerts)}):")
            for alert in warning_alerts:
                print(f"  • {alert.message}")
        
        print(f"\nTotal Alerts: {len(self.alerts)}")
        print(f"Status: {'FAIL' if critical_alerts else 'WARN' if warning_alerts else 'PASS'}")


# pytest test functions

@pytest.fixture
def performance_monitor():
    """Create performance monitor instance."""
    return PerformanceMonitor()


def test_threshold_configuration(performance_monitor):
    """Test performance threshold configuration."""
    monitor = performance_monitor
    
    # Should have default thresholds
    assert len(monitor.thresholds) > 0
    
    # Each threshold should have required fields
    for threshold in monitor.thresholds:
        assert threshold.metric_name
        assert threshold.warning_threshold > 0
        assert threshold.critical_threshold > threshold.warning_threshold
        assert threshold.unit
        assert threshold.comparison in ["greater_than", "less_than"]


def test_metric_extraction(performance_monitor):
    """Test metric extraction from performance data."""
    monitor = performance_monitor
    
    # Mock performance data
    mock_data = {
        "performance_statistics": {
            "execution_time": {"mean_ms": 1500.0},
            "memory_usage": {"mean_mb": 250.0},
            "cpu_usage": {"mean_percent": 45.0}
        },
        "detailed_results": [
            {"execution_time_ms": 1400},
            {"execution_time_ms": 1600}
        ]
    }
    
    # Should extract metrics correctly
    exec_time = monitor._extract_metric_value(mock_data, "execution_time_ms")
    assert exec_time == 1500.0
    
    memory = monitor._extract_metric_value(mock_data, "memory_usage_mb")
    assert memory == 250.0
    
    success_rate = monitor._extract_metric_value(mock_data, "success_rate")
    assert success_rate == 1.0  # Both results under 30s threshold


def test_alert_generation(performance_monitor):
    """Test performance alert generation."""
    monitor = performance_monitor
    
    # Mock data that exceeds thresholds
    mock_data = {
        "platform_info": {"system": "TestOS"},
        "performance_statistics": {
            "execution_time": {"mean_ms": 12000.0},  # Exceeds critical threshold
            "memory_usage": {"mean_mb": 750.0},      # Exceeds warning threshold
            "cpu_usage": {"mean_percent": 30.0}      # Within limits
        }
    }
    
    alerts = monitor.check_performance_thresholds(mock_data)
    
    # Should generate alerts for exceeded thresholds
    assert len(alerts) >= 2
    
    # Should have critical alert for execution time
    critical_alerts = [a for a in alerts if a.severity == "critical"]
    assert any(a.metric_name == "execution_time_ms" for a in critical_alerts)
    
    # Should have warning alert for memory
    warning_alerts = [a for a in alerts if a.severity == "warning"]
    assert any(a.metric_name == "memory_usage_mb" for a in warning_alerts)


def test_trend_analysis(performance_monitor):
    """Test performance trend analysis."""
    monitor = performance_monitor
    
    # Test trend calculation
    improving_values = [100, 95, 90, 85, 80]  # Improving trend
    degrading_values = [80, 85, 90, 95, 100]  # Degrading trend
    stable_values = [90, 91, 89, 90, 91]      # Stable trend
    
    improving_trend = monitor._calculate_trend(improving_values)
    assert improving_trend["direction"] == "improving"
    
    degrading_trend = monitor._calculate_trend(degrading_values)
    assert degrading_trend["direction"] == "degrading"
    
    stable_trend = monitor._calculate_trend(stable_values)
    assert stable_trend["direction"] == "stable"


@pytest.mark.slow
def test_ci_cd_integration(performance_monitor):
    """Test CI/CD integration report generation."""
    monitor = performance_monitor
    
    # Mock current metrics
    current_metrics = {
        "platform_info": {"system": "Linux"},
        "performance_statistics": {
            "execution_time": {"mean_ms": 2000.0},
            "memory_usage": {"mean_mb": 300.0},
            "cpu_usage": {"mean_percent": 50.0}
        },
        "test_summary": {"test_date": datetime.utcnow().isoformat()},
        "detailed_results": [{"execution_time_ms": 2000}]
    }
    
    # Generate CI/CD report
    report = monitor.generate_ci_cd_report(current_metrics)
    
    # Should have required sections
    assert "report_info" in report
    assert "current_metrics" in report
    assert "alerts" in report
    assert "trends" in report
    assert "summary" in report
    
    # Summary should have status
    assert report["summary"]["overall_status"] in ["PASS", "WARN", "FAIL"]


if __name__ == "__main__":
    # Example usage for CI/CD integration
    monitor = PerformanceMonitor()
    
    # Mock performance data (in real usage, this would come from actual tests)
    mock_metrics = {
        "platform_info": {"system": "Linux"},
        "performance_statistics": {
            "execution_time": {"mean_ms": 3500.0},
            "memory_usage": {"mean_mb": 450.0},
            "cpu_usage": {"mean_percent": 65.0}
        },
        "test_summary": {"test_date": datetime.utcnow().isoformat()},
        "detailed_results": [
            {"execution_time_ms": 3400},
            {"execution_time_ms": 3600}
        ]
    }
    
    # Generate CI/CD report
    report = monitor.generate_ci_cd_report(mock_metrics)
    
    # Save to history
    monitor.save_performance_history(mock_metrics)
    
    # Print summary
    monitor.print_alert_summary()
    
    print(f"\nCI/CD Integration Status: {report['summary']['overall_status']}")
    print(f"Total Alerts: {report['summary']['total_alerts']}")
    
    # Exit with appropriate code for CI/CD
    exit_code = 0
    if report['summary']['overall_status'] == "FAIL":
        exit_code = 1
    elif report['summary']['overall_status'] == "WARN":
        exit_code = 2
    
    print(f"Exit code: {exit_code}")