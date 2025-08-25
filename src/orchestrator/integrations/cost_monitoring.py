"""
Cost monitoring integration for wrapper performance tracking.

This module provides comprehensive cost tracking and analysis integration
that works with existing monitoring systems and provides detailed cost
analytics for wrapper operations.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple, Union
from threading import Lock
import statistics

from ..core.wrapper_monitoring import WrapperMonitoring, Alert, AlertSeverity

logger = logging.getLogger(__name__)


@dataclass
class CostMetrics:
    """Comprehensive cost metrics for wrapper operations."""
    
    # Operation identification
    operation_id: str
    wrapper_name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Cost breakdown
    api_cost: Decimal = field(default_factory=lambda: Decimal('0.0'))
    infrastructure_cost: Decimal = field(default_factory=lambda: Decimal('0.0'))
    compute_cost: Decimal = field(default_factory=lambda: Decimal('0.0'))
    total_cost: Decimal = field(default_factory=lambda: Decimal('0.0'))
    
    # Usage metrics
    tokens_used: int = 0
    requests_made: int = 0
    compute_time_ms: float = 0.0
    bandwidth_bytes: int = 0
    
    # Optimization metrics
    cost_savings: Decimal = field(default_factory=lambda: Decimal('0.0'))
    efficiency_score: float = 1.0
    optimization_applied: bool = False
    
    # Model and routing information
    model_used: Optional[str] = None
    routing_decision: Optional[str] = None
    fallback_used: bool = False
    
    # Context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    region: Optional[str] = None
    
    def __post_init__(self):
        """Ensure decimal precision for cost fields."""
        if not isinstance(self.api_cost, Decimal):
            self.api_cost = Decimal(str(self.api_cost))
        if not isinstance(self.infrastructure_cost, Decimal):
            self.infrastructure_cost = Decimal(str(self.infrastructure_cost))
        if not isinstance(self.compute_cost, Decimal):
            self.compute_cost = Decimal(str(self.compute_cost))
        if not isinstance(self.total_cost, Decimal):
            self.total_cost = Decimal(str(self.total_cost))
        if not isinstance(self.cost_savings, Decimal):
            self.cost_savings = Decimal(str(self.cost_savings))
        
        # Calculate total if not provided
        if self.total_cost == Decimal('0.0'):
            self.total_cost = self.api_cost + self.infrastructure_cost + self.compute_cost
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation_id": self.operation_id,
            "wrapper_name": self.wrapper_name,
            "timestamp": self.timestamp.isoformat(),
            "api_cost": float(self.api_cost),
            "infrastructure_cost": float(self.infrastructure_cost),
            "compute_cost": float(self.compute_cost),
            "total_cost": float(self.total_cost),
            "tokens_used": self.tokens_used,
            "requests_made": self.requests_made,
            "compute_time_ms": self.compute_time_ms,
            "bandwidth_bytes": self.bandwidth_bytes,
            "cost_savings": float(self.cost_savings),
            "efficiency_score": self.efficiency_score,
            "optimization_applied": self.optimization_applied,
            "model_used": self.model_used,
            "routing_decision": self.routing_decision,
            "fallback_used": self.fallback_used,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "region": self.region
        }


@dataclass
class BudgetConfig:
    """Budget configuration for cost tracking."""
    
    wrapper_name: str
    daily_budget: Decimal
    monthly_budget: Decimal
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "warning": 0.8,
        "critical": 0.95
    })
    notification_channels: List[str] = field(default_factory=list)
    auto_disable_on_exceed: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Ensure decimal precision for budget fields."""
        if not isinstance(self.daily_budget, Decimal):
            self.daily_budget = Decimal(str(self.daily_budget))
        if not isinstance(self.monthly_budget, Decimal):
            self.monthly_budget = Decimal(str(self.monthly_budget))


@dataclass
class CostAnalytics:
    """Cost analytics data for a time period."""
    
    wrapper_name: str
    period_start: datetime
    period_end: datetime
    
    # Aggregate costs
    total_cost: Decimal = field(default_factory=lambda: Decimal('0.0'))
    average_cost_per_operation: Decimal = field(default_factory=lambda: Decimal('0.0'))
    median_cost_per_operation: Decimal = field(default_factory=lambda: Decimal('0.0'))
    
    # Cost breakdown
    api_cost_total: Decimal = field(default_factory=lambda: Decimal('0.0'))
    infrastructure_cost_total: Decimal = field(default_factory=lambda: Decimal('0.0'))
    compute_cost_total: Decimal = field(default_factory=lambda: Decimal('0.0'))
    
    # Operations metrics
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    fallback_operations: int = 0
    
    # Optimization metrics
    total_savings: Decimal = field(default_factory=lambda: Decimal('0.0'))
    average_efficiency_score: float = 1.0
    optimization_rate: float = 0.0
    
    # Usage metrics
    total_tokens: int = 0
    total_requests: int = 0
    total_compute_time_ms: float = 0.0
    
    # Cost efficiency metrics
    cost_per_token: Decimal = field(default_factory=lambda: Decimal('0.0'))
    cost_per_request: Decimal = field(default_factory=lambda: Decimal('0.0'))
    cost_per_compute_hour: Decimal = field(default_factory=lambda: Decimal('0.0'))
    
    def calculate_derived_metrics(self, cost_metrics: List[CostMetrics]) -> None:
        """Calculate derived metrics from cost data."""
        if not cost_metrics:
            return
        
        # Aggregate costs
        self.total_cost = sum(m.total_cost for m in cost_metrics)
        self.api_cost_total = sum(m.api_cost for m in cost_metrics)
        self.infrastructure_cost_total = sum(m.infrastructure_cost for m in cost_metrics)
        self.compute_cost_total = sum(m.compute_cost for m in cost_metrics)
        
        # Operations
        self.total_operations = len(cost_metrics)
        self.successful_operations = len([m for m in cost_metrics if not m.fallback_used])
        self.fallback_operations = len([m for m in cost_metrics if m.fallback_used])
        
        # Average costs
        if self.total_operations > 0:
            self.average_cost_per_operation = self.total_cost / self.total_operations
            
            # Median cost
            costs = [m.total_cost for m in cost_metrics]
            costs.sort()
            mid = len(costs) // 2
            if len(costs) % 2 == 0:
                self.median_cost_per_operation = (costs[mid-1] + costs[mid]) / 2
            else:
                self.median_cost_per_operation = costs[mid]
        
        # Optimization metrics
        self.total_savings = sum(m.cost_savings for m in cost_metrics)
        optimized_operations = [m for m in cost_metrics if m.optimization_applied]
        if optimized_operations:
            self.average_efficiency_score = statistics.mean(m.efficiency_score for m in optimized_operations)
            self.optimization_rate = len(optimized_operations) / self.total_operations
        
        # Usage metrics
        self.total_tokens = sum(m.tokens_used for m in cost_metrics)
        self.total_requests = sum(m.requests_made for m in cost_metrics)
        self.total_compute_time_ms = sum(m.compute_time_ms for m in cost_metrics)
        
        # Efficiency metrics
        if self.total_tokens > 0:
            self.cost_per_token = self.total_cost / self.total_tokens
        if self.total_requests > 0:
            self.cost_per_request = self.total_cost / self.total_requests
        if self.total_compute_time_ms > 0:
            compute_hours = Decimal(str(self.total_compute_time_ms / (1000 * 3600)))
            if compute_hours > 0:
                self.cost_per_compute_hour = self.total_cost / compute_hours


class CostMonitoringIntegration:
    """Integration with cost tracking and monitoring systems."""
    
    def __init__(
        self,
        monitoring: WrapperMonitoring,
        retention_days: int = 90,
        analytics_cache_hours: int = 1
    ):
        """
        Initialize cost monitoring integration.
        
        Args:
            monitoring: WrapperMonitoring instance to integrate with
            retention_days: How long to keep cost data
            analytics_cache_hours: How long to cache analytics
        """
        self.monitoring = monitoring
        self.retention_days = retention_days
        self.analytics_cache_hours = analytics_cache_hours
        
        # Data storage
        self._cost_metrics: deque = deque(maxlen=100000)  # Limit memory usage
        self._budget_configs: Dict[str, BudgetConfig] = {}
        self._analytics_cache: Dict[str, Tuple[CostAnalytics, datetime]] = {}
        
        # Thread safety
        self._lock = Lock()
        
        # Budget tracking
        self._daily_spending: Dict[str, Decimal] = defaultdict(lambda: Decimal('0.0'))
        self._monthly_spending: Dict[str, Decimal] = defaultdict(lambda: Decimal('0.0'))
        self._last_reset_date = datetime.utcnow().date()
        self._last_reset_month = datetime.utcnow().replace(day=1).date()
        
        logger.info("Initialized cost monitoring integration")
    
    def record_operation_cost(
        self,
        operation_id: str,
        wrapper_name: str,
        cost_data: Dict[str, Any]
    ) -> None:
        """Record cost metrics for a wrapper operation."""
        
        with self._lock:
            try:
                cost_metrics = CostMetrics(
                    operation_id=operation_id,
                    wrapper_name=wrapper_name,
                    timestamp=datetime.utcnow(),
                    api_cost=Decimal(str(cost_data.get('api_cost', '0.0'))),
                    infrastructure_cost=Decimal(str(cost_data.get('infrastructure_cost', '0.0'))),
                    compute_cost=Decimal(str(cost_data.get('compute_cost', '0.0'))),
                    total_cost=Decimal(str(cost_data.get('total_cost', '0.0'))),
                    tokens_used=cost_data.get('tokens_used', 0),
                    requests_made=cost_data.get('requests_made', 1),
                    compute_time_ms=cost_data.get('compute_time_ms', 0.0),
                    bandwidth_bytes=cost_data.get('bandwidth_bytes', 0),
                    cost_savings=Decimal(str(cost_data.get('cost_savings', '0.0'))),
                    efficiency_score=cost_data.get('efficiency_score', 1.0),
                    optimization_applied=cost_data.get('optimization_applied', False),
                    model_used=cost_data.get('model_used'),
                    routing_decision=cost_data.get('routing_decision'),
                    fallback_used=cost_data.get('fallback_used', False),
                    user_id=cost_data.get('user_id'),
                    session_id=cost_data.get('session_id'),
                    region=cost_data.get('region')
                )
                
                # Store cost metrics
                self._cost_metrics.append(cost_metrics)
                
                # Update budget tracking
                self._update_budget_tracking(wrapper_name, cost_metrics)
                
                # Update wrapper monitoring with cost information
                if hasattr(self.monitoring, '_active_operations') and operation_id in self.monitoring._active_operations:
                    operation = self.monitoring._active_operations[operation_id]
                    operation.cost_estimate = float(cost_metrics.total_cost)
                    operation.custom_metrics.update({
                        'api_cost': float(cost_metrics.api_cost),
                        'infrastructure_cost': float(cost_metrics.infrastructure_cost),
                        'compute_cost': float(cost_metrics.compute_cost),
                        'cost_savings': float(cost_metrics.cost_savings),
                        'efficiency_score': cost_metrics.efficiency_score,
                        'tokens_used': cost_metrics.tokens_used,
                        'model_used': cost_metrics.model_used
                    })
                
                # Clear analytics cache for this wrapper
                cache_key = f"{wrapper_name}_analytics"
                if cache_key in self._analytics_cache:
                    del self._analytics_cache[cache_key]
                
                logger.debug(f"Recorded cost metrics for {operation_id}: ${cost_metrics.total_cost}")
                
            except Exception as e:
                logger.error(f"Failed to record cost metrics for {operation_id}: {e}")
    
    def set_budget(
        self,
        wrapper_name: str,
        daily_budget: Union[Decimal, float],
        monthly_budget: Union[Decimal, float],
        alert_thresholds: Optional[Dict[str, float]] = None,
        notification_channels: Optional[List[str]] = None,
        auto_disable_on_exceed: bool = False
    ) -> None:
        """Set cost budget for a wrapper with alert thresholds."""
        
        with self._lock:
            budget_config = BudgetConfig(
                wrapper_name=wrapper_name,
                daily_budget=Decimal(str(daily_budget)),
                monthly_budget=Decimal(str(monthly_budget)),
                alert_thresholds=alert_thresholds or {"warning": 0.8, "critical": 0.95},
                notification_channels=notification_channels or [],
                auto_disable_on_exceed=auto_disable_on_exceed
            )
            
            self._budget_configs[wrapper_name] = budget_config
            
            logger.info(
                f"Set budget for {wrapper_name}: "
                f"Daily ${budget_config.daily_budget}, Monthly ${budget_config.monthly_budget}"
            )
    
    def get_cost_analysis(
        self,
        wrapper_name: Optional[str] = None,
        time_range_hours: int = 24,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Get comprehensive cost analysis."""
        
        # Check cache first
        if use_cache and wrapper_name:
            cache_key = f"{wrapper_name}_analytics_{time_range_hours}h"
            if cache_key in self._analytics_cache:
                analytics, cached_at = self._analytics_cache[cache_key]
                if datetime.utcnow() - cached_at < timedelta(hours=self.analytics_cache_hours):
                    return self._format_cost_analysis(analytics)
        
        with self._lock:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
            
            # Filter metrics
            filtered_metrics = [
                m for m in self._cost_metrics
                if m.timestamp >= cutoff_time and (not wrapper_name or m.wrapper_name == wrapper_name)
            ]
            
            if not filtered_metrics:
                return {
                    "total_cost": 0.0,
                    "operations": 0,
                    "average_cost": 0.0,
                    "period": f"{time_range_hours}h",
                    "wrapper_name": wrapper_name
                }
            
            # Create analytics
            analytics = CostAnalytics(
                wrapper_name=wrapper_name or "all_wrappers",
                period_start=cutoff_time,
                period_end=datetime.utcnow()
            )
            
            analytics.calculate_derived_metrics(filtered_metrics)
            
            # Cache if wrapper-specific
            if wrapper_name and use_cache:
                cache_key = f"{wrapper_name}_analytics_{time_range_hours}h"
                self._analytics_cache[cache_key] = (analytics, datetime.utcnow())
            
            return self._format_cost_analysis(analytics)
    
    def get_budget_status(self, wrapper_name: Optional[str] = None) -> Dict[str, Any]:
        """Get budget status for wrappers."""
        
        with self._lock:
            self._reset_budget_periods()
            
            if wrapper_name:
                if wrapper_name not in self._budget_configs:
                    return {"error": f"No budget configured for {wrapper_name}"}
                
                return self._get_wrapper_budget_status(wrapper_name)
            else:
                # Return status for all configured budgets
                return {
                    name: self._get_wrapper_budget_status(name)
                    for name in self._budget_configs.keys()
                }
    
    def get_cost_optimization_recommendations(
        self,
        wrapper_name: str,
        analysis_days: int = 7
    ) -> List[Dict[str, Any]]:
        """Get cost optimization recommendations based on usage patterns."""
        
        with self._lock:
            cutoff_time = datetime.utcnow() - timedelta(days=analysis_days)
            
            # Get recent metrics for this wrapper
            recent_metrics = [
                m for m in self._cost_metrics
                if m.wrapper_name == wrapper_name and m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return []
            
            recommendations = []
            
            # Analyze model usage patterns
            model_usage = defaultdict(list)
            for metric in recent_metrics:
                if metric.model_used:
                    model_usage[metric.model_used].append(metric)
            
            # Recommend model optimization
            if len(model_usage) > 1:
                model_costs = {
                    model: sum(m.total_cost for m in metrics)
                    for model, metrics in model_usage.items()
                }
                
                highest_cost_model = max(model_costs, key=model_costs.get)
                total_cost = sum(model_costs.values())
                
                if model_costs[highest_cost_model] / total_cost > 0.7:
                    recommendations.append({
                        "type": "model_optimization",
                        "priority": "high",
                        "description": f"Consider using more cost-effective alternatives to {highest_cost_model}",
                        "potential_savings": float(model_costs[highest_cost_model] * Decimal('0.3')),
                        "details": {
                            "dominant_model": highest_cost_model,
                            "cost_share": float(model_costs[highest_cost_model] / total_cost),
                            "total_operations": len(model_usage[highest_cost_model])
                        }
                    })
            
            # Analyze efficiency scores
            efficiency_scores = [m.efficiency_score for m in recent_metrics if m.efficiency_score < 1.0]
            if efficiency_scores and statistics.mean(efficiency_scores) < 0.8:
                avg_efficiency = statistics.mean(efficiency_scores)
                potential_savings = sum(
                    m.total_cost * Decimal(str(1.0 - m.efficiency_score))
                    for m in recent_metrics
                    if m.efficiency_score < 1.0
                )
                
                recommendations.append({
                    "type": "efficiency_improvement",
                    "priority": "medium",
                    "description": f"Low efficiency detected (avg: {avg_efficiency:.2f}). Review routing decisions.",
                    "potential_savings": float(potential_savings),
                    "details": {
                        "average_efficiency": avg_efficiency,
                        "inefficient_operations": len(efficiency_scores),
                        "total_operations": len(recent_metrics)
                    }
                })
            
            # Analyze usage patterns
            tokens_per_request = [
                m.tokens_used / max(1, m.requests_made)
                for m in recent_metrics
                if m.tokens_used > 0 and m.requests_made > 0
            ]
            
            if tokens_per_request and statistics.mean(tokens_per_request) > 1000:
                recommendations.append({
                    "type": "token_optimization",
                    "priority": "low",
                    "description": "High token usage detected. Consider prompt optimization or caching.",
                    "potential_savings": "Variable",
                    "details": {
                        "average_tokens_per_request": statistics.mean(tokens_per_request),
                        "max_tokens_per_request": max(tokens_per_request),
                        "total_requests": sum(m.requests_made for m in recent_metrics)
                    }
                })
            
            return recommendations
    
    def export_cost_data(
        self,
        wrapper_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        format: str = "dict"
    ) -> Union[List[Dict[str, Any]], str]:
        """Export cost data for external analysis."""
        
        with self._lock:
            # Apply filters
            filtered_metrics = list(self._cost_metrics)
            
            if wrapper_name:
                filtered_metrics = [m for m in filtered_metrics if m.wrapper_name == wrapper_name]
            
            if start_time:
                filtered_metrics = [m for m in filtered_metrics if m.timestamp >= start_time]
            
            if end_time:
                filtered_metrics = [m for m in filtered_metrics if m.timestamp <= end_time]
            
            # Convert to requested format
            if format == "dict":
                return [m.to_dict() for m in filtered_metrics]
            elif format == "csv":
                if not filtered_metrics:
                    return ""
                
                # Generate CSV
                headers = list(filtered_metrics[0].to_dict().keys())
                csv_lines = [",".join(headers)]
                
                for metric in filtered_metrics:
                    values = [str(v) for v in metric.to_dict().values()]
                    csv_lines.append(",".join(values))
                
                return "\n".join(csv_lines)
            else:
                raise ValueError(f"Unsupported export format: {format}")
    
    def cleanup_old_data(self) -> int:
        """Remove cost data older than retention period."""
        
        with self._lock:
            cutoff_time = datetime.utcnow() - timedelta(days=self.retention_days)
            original_count = len(self._cost_metrics)
            
            # Filter out old data
            self._cost_metrics = deque(
                (m for m in self._cost_metrics if m.timestamp >= cutoff_time),
                maxlen=self._cost_metrics.maxlen
            )
            
            cleaned = original_count - len(self._cost_metrics)
            
            # Clear old cache entries
            current_time = datetime.utcnow()
            cache_cutoff = timedelta(hours=self.analytics_cache_hours)
            
            expired_keys = [
                key for key, (_, cached_at) in self._analytics_cache.items()
                if current_time - cached_at > cache_cutoff
            ]
            
            for key in expired_keys:
                del self._analytics_cache[key]
            
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} old cost records")
            
            return cleaned
    
    def _update_budget_tracking(self, wrapper_name: str, cost_metrics: CostMetrics) -> None:
        """Update budget tracking and check thresholds."""
        
        # Reset budget periods if needed
        self._reset_budget_periods()
        
        # Update spending
        self._daily_spending[wrapper_name] += cost_metrics.total_cost
        self._monthly_spending[wrapper_name] += cost_metrics.total_cost
        
        # Check budget thresholds
        if wrapper_name in self._budget_configs:
            self._check_budget_thresholds(wrapper_name, cost_metrics)
    
    def _reset_budget_periods(self) -> None:
        """Reset daily and monthly budget tracking if needed."""
        
        current_date = datetime.utcnow().date()
        current_month = datetime.utcnow().replace(day=1).date()
        
        # Reset daily spending
        if current_date > self._last_reset_date:
            self._daily_spending.clear()
            self._last_reset_date = current_date
        
        # Reset monthly spending
        if current_month > self._last_reset_month:
            self._monthly_spending.clear()
            self._last_reset_month = current_month
    
    def _check_budget_thresholds(self, wrapper_name: str, cost_metrics: CostMetrics) -> None:
        """Check if cost metrics exceed budget thresholds."""
        
        budget_config = self._budget_configs[wrapper_name]
        daily_spending = self._daily_spending[wrapper_name]
        monthly_spending = self._monthly_spending[wrapper_name]
        
        # Check thresholds
        for threshold_name, threshold_pct in budget_config.alert_thresholds.items():
            daily_threshold = budget_config.daily_budget * Decimal(str(threshold_pct))
            monthly_threshold = budget_config.monthly_budget * Decimal(str(threshold_pct))
            
            # Daily budget alert
            if daily_spending >= daily_threshold:
                self._trigger_budget_alert(
                    wrapper_name,
                    "daily",
                    daily_spending,
                    budget_config.daily_budget,
                    threshold_name,
                    budget_config
                )
            
            # Monthly budget alert
            if monthly_spending >= monthly_threshold:
                self._trigger_budget_alert(
                    wrapper_name,
                    "monthly",
                    monthly_spending,
                    budget_config.monthly_budget,
                    threshold_name,
                    budget_config
                )
    
    def _trigger_budget_alert(
        self,
        wrapper_name: str,
        period: str,
        current_spending: Decimal,
        budget: Decimal,
        threshold_name: str,
        budget_config: BudgetConfig
    ) -> None:
        """Trigger budget threshold alert."""
        
        percentage = (current_spending / budget) * 100
        severity = AlertSeverity.WARNING if threshold_name == "warning" else AlertSeverity.CRITICAL
        
        alert = Alert(
            wrapper_name=wrapper_name,
            rule_name=f"budget_{threshold_name}_{period}",
            severity=severity,
            message=(
                f"Budget {threshold_name} threshold exceeded: "
                f"{percentage:.1f}% of {period} budget "
                f"(${current_spending:.2f}/${budget:.2f})"
            )
        )
        
        # Add to monitoring system's alerts
        self.monitoring._alerts.append(alert)
        
        # Auto-disable if configured
        if (threshold_name == "critical" and 
            budget_config.auto_disable_on_exceed and 
            current_spending >= budget):
            
            logger.critical(
                f"Budget exceeded for {wrapper_name}, auto-disabling wrapper"
            )
            # Note: This would need to integrate with the wrapper configuration
            # to actually disable the wrapper
        
        logger.warning(f"Budget alert triggered for {wrapper_name}: {alert.message}")
    
    def _get_wrapper_budget_status(self, wrapper_name: str) -> Dict[str, Any]:
        """Get budget status for a specific wrapper."""
        
        budget_config = self._budget_configs[wrapper_name]
        daily_spending = self._daily_spending[wrapper_name]
        monthly_spending = self._monthly_spending[wrapper_name]
        
        # Calculate percentages
        daily_percentage = float((daily_spending / budget_config.daily_budget) * 100)
        monthly_percentage = float((monthly_spending / budget_config.monthly_budget) * 100)
        
        # Determine status
        def get_status(percentage: float, thresholds: Dict[str, float]) -> str:
            if percentage >= thresholds.get("critical", 95):
                return "critical"
            elif percentage >= thresholds.get("warning", 80):
                return "warning"
            else:
                return "healthy"
        
        daily_status = get_status(daily_percentage, budget_config.alert_thresholds)
        monthly_status = get_status(monthly_percentage, budget_config.alert_thresholds)
        
        return {
            "wrapper_name": wrapper_name,
            "daily": {
                "budget": float(budget_config.daily_budget),
                "spent": float(daily_spending),
                "remaining": float(budget_config.daily_budget - daily_spending),
                "percentage": daily_percentage,
                "status": daily_status
            },
            "monthly": {
                "budget": float(budget_config.monthly_budget),
                "spent": float(monthly_spending),
                "remaining": float(budget_config.monthly_budget - monthly_spending),
                "percentage": monthly_percentage,
                "status": monthly_status
            },
            "alert_thresholds": budget_config.alert_thresholds,
            "auto_disable_enabled": budget_config.auto_disable_on_exceed
        }
    
    def _format_cost_analysis(self, analytics: CostAnalytics) -> Dict[str, Any]:
        """Format cost analytics for API response."""
        
        return {
            "wrapper_name": analytics.wrapper_name,
            "period": {
                "start": analytics.period_start.isoformat(),
                "end": analytics.period_end.isoformat(),
                "duration_hours": (analytics.period_end - analytics.period_start).total_seconds() / 3600
            },
            "costs": {
                "total": float(analytics.total_cost),
                "average_per_operation": float(analytics.average_cost_per_operation),
                "median_per_operation": float(analytics.median_cost_per_operation),
                "breakdown": {
                    "api": float(analytics.api_cost_total),
                    "infrastructure": float(analytics.infrastructure_cost_total),
                    "compute": float(analytics.compute_cost_total)
                }
            },
            "operations": {
                "total": analytics.total_operations,
                "successful": analytics.successful_operations,
                "failed": analytics.failed_operations,
                "fallback": analytics.fallback_operations,
                "success_rate": analytics.successful_operations / max(1, analytics.total_operations)
            },
            "optimization": {
                "total_savings": float(analytics.total_savings),
                "average_efficiency_score": analytics.average_efficiency_score,
                "optimization_rate": analytics.optimization_rate
            },
            "usage": {
                "total_tokens": analytics.total_tokens,
                "total_requests": analytics.total_requests,
                "total_compute_time_hours": analytics.total_compute_time_ms / (1000 * 3600)
            },
            "efficiency": {
                "cost_per_token": float(analytics.cost_per_token),
                "cost_per_request": float(analytics.cost_per_request),
                "cost_per_compute_hour": float(analytics.cost_per_compute_hour)
            }
        }


# Factory function for easy instantiation
def create_cost_monitoring_integration(
    monitoring: WrapperMonitoring,
    retention_days: int = 90,
    analytics_cache_hours: int = 1
) -> CostMonitoringIntegration:
    """Create a cost monitoring integration."""
    return CostMonitoringIntegration(monitoring, retention_days, analytics_cache_hours)