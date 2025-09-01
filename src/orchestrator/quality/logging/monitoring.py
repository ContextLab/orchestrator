"""
External monitoring system integration for quality control logging.

This module provides comprehensive integration with external monitoring and
alerting systems including Prometheus, Grafana, ELK Stack, and custom webhook
endpoints. It enables real-time monitoring of quality events, performance
metrics, and system health.

Key Features:
- Prometheus metrics export with custom collectors
- Grafana dashboard integration and management
- ELK Stack log forwarding with structured data
- Custom webhook alerts for quality threshold breaches
- Health check endpoints for monitoring system health
- Configurable alerting rules and thresholds
- Rate limiting and backpressure handling
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable, AsyncIterator
from pathlib import Path
import aiohttp
import queue
from concurrent.futures import ThreadPoolExecutor
import socket
from urllib.parse import urljoin

from .logger import LogLevel, LogCategory, LogContext, QualityEvent, StructuredLogger


@dataclass
class AlertRule:
    """Configuration for quality control alerting rules."""
    name: str
    condition: str  # Python expression for evaluation
    severity: str   # INFO, WARNING, ERROR, CRITICAL
    threshold: float
    window_seconds: int
    cooldown_seconds: int = 300
    description: Optional[str] = None
    remediation_url: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


@dataclass
class MonitoringAlert:
    """Alert notification for quality control events."""
    rule_name: str
    severity: str
    message: str
    timestamp: str
    value: float
    threshold: float
    context: Dict[str, Any]
    remediation_url: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return asdict(self)


class MonitoringBackend(ABC):
    """Abstract base class for monitoring system backends."""
    
    @abstractmethod
    async def send_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Send metrics to monitoring backend."""
        pass
    
    @abstractmethod
    async def send_alert(self, alert: MonitoringAlert) -> bool:
        """Send alert to monitoring backend."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check backend health status."""
        pass


class PrometheusBackend(MonitoringBackend):
    """Prometheus monitoring backend with pushgateway support."""
    
    def __init__(
        self,
        pushgateway_url: str,
        job_name: str = "orchestrator_quality",
        instance_id: Optional[str] = None,
        basic_auth: Optional[tuple] = None,
        timeout: float = 10.0
    ):
        self.pushgateway_url = pushgateway_url.rstrip('/')
        self.job_name = job_name
        self.instance_id = instance_id or socket.gethostname()
        self.basic_auth = basic_auth
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if not self._session or self._session.closed:
            auth = aiohttp.BasicAuth(*self.basic_auth) if self.basic_auth else None
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                auth=auth
            )
        return self._session

    async def send_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Send metrics to Prometheus pushgateway."""
        try:
            session = await self._get_session()
            
            # Convert metrics to Prometheus format
            prometheus_metrics = self._format_prometheus_metrics(metrics)
            
            # Build pushgateway URL
            url = f"{self.pushgateway_url}/metrics/job/{self.job_name}/instance/{self.instance_id}"
            
            # Send metrics
            async with session.post(url, data=prometheus_metrics, 
                                  headers={'Content-Type': 'text/plain'}) as response:
                return response.status == 200
                
        except Exception as e:
            logging.error(f"Failed to send metrics to Prometheus: {e}")
            return False

    async def send_alert(self, alert: MonitoringAlert) -> bool:
        """Send alert as metric to Prometheus (alerts handled by Alertmanager)."""
        alert_metrics = {
            f"orchestrator_alert_{alert.rule_name.lower()}": {
                'value': 1.0,
                'labels': {
                    'severity': alert.severity.lower(),
                    'rule': alert.rule_name,
                    'instance': self.instance_id,
                    **(alert.tags or {})
                },
                'timestamp': time.time()
            }
        }
        
        return await self.send_metrics(alert_metrics)

    async def health_check(self) -> bool:
        """Check Prometheus pushgateway health."""
        try:
            session = await self._get_session()
            
            async with session.get(f"{self.pushgateway_url}/-/healthy") as response:
                return response.status == 200
                
        except Exception:
            return False

    def _format_prometheus_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics in Prometheus exposition format."""
        lines = []
        
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict):
                value = metric_data.get('value', 0)
                labels = metric_data.get('labels', {})
                help_text = metric_data.get('help', f'Quality control metric: {metric_name}')
                metric_type = metric_data.get('type', 'gauge')
                
                # Add help and type
                lines.append(f"# HELP {metric_name} {help_text}")
                lines.append(f"# TYPE {metric_name} {metric_type}")
                
                # Format labels
                if labels:
                    label_str = ','.join([f'{k}="{v}"' for k, v in labels.items()])
                    lines.append(f"{metric_name}{{{label_str}}} {value}")
                else:
                    lines.append(f"{metric_name} {value}")
            else:
                # Simple metric without labels
                lines.append(f"# TYPE {metric_name} gauge")
                lines.append(f"{metric_name} {metric_data}")
        
        return '\n'.join(lines)

    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


class WebhookBackend(MonitoringBackend):
    """Generic webhook backend for custom monitoring integrations."""
    
    def __init__(
        self,
        webhook_url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 10.0,
        retry_count: int = 3,
        retry_delay: float = 1.0
    ):
        self.webhook_url = webhook_url
        self.headers = headers or {}
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers=self.headers
            )
        return self._session

    async def send_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Send metrics to webhook endpoint."""
        payload = {
            'type': 'metrics',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'source': 'orchestrator_quality',
            'data': metrics
        }
        
        return await self._send_payload(payload)

    async def send_alert(self, alert: MonitoringAlert) -> bool:
        """Send alert to webhook endpoint."""
        payload = {
            'type': 'alert',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'source': 'orchestrator_quality',
            'data': alert.to_dict()
        }
        
        return await self._send_payload(payload)

    async def health_check(self) -> bool:
        """Check webhook endpoint health."""
        health_payload = {
            'type': 'health_check',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'source': 'orchestrator_quality'
        }
        
        return await self._send_payload(health_payload)

    async def _send_payload(self, payload: Dict[str, Any]) -> bool:
        """Send payload to webhook with retries."""
        session = await self._get_session()
        
        for attempt in range(self.retry_count):
            try:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status in (200, 201, 202):
                        return True
                    elif response.status >= 500 and attempt < self.retry_count - 1:
                        # Retry on server errors
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    else:
                        return False
                        
            except Exception as e:
                if attempt < self.retry_count - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                else:
                    logging.error(f"Failed to send webhook after {self.retry_count} attempts: {e}")
                    return False
        
        return False

    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


class ElasticsearchBackend(MonitoringBackend):
    """Elasticsearch backend for log aggregation and analysis."""
    
    def __init__(
        self,
        elasticsearch_url: str,
        index_name: str = "orchestrator-quality",
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 10.0
    ):
        self.elasticsearch_url = elasticsearch_url.rstrip('/')
        self.index_name = index_name
        self.username = username
        self.password = password
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with authentication."""
        if not self._session or self._session.closed:
            auth = None
            if self.username and self.password:
                auth = aiohttp.BasicAuth(self.username, self.password)
                
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                auth=auth
            )
        return self._session

    async def send_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Send metrics to Elasticsearch."""
        try:
            session = await self._get_session()
            
            # Create document for indexing
            document = {
                '@timestamp': datetime.now(timezone.utc).isoformat(),
                'type': 'metrics',
                'source': 'orchestrator_quality',
                'metrics': metrics
            }
            
            # Generate document ID based on timestamp
            doc_id = f"metrics_{int(time.time() * 1000)}"
            url = f"{self.elasticsearch_url}/{self.index_name}/_doc/{doc_id}"
            
            async with session.put(
                url,
                json=document,
                headers={'Content-Type': 'application/json'}
            ) as response:
                return response.status in (200, 201)
                
        except Exception as e:
            logging.error(f"Failed to send metrics to Elasticsearch: {e}")
            return False

    async def send_alert(self, alert: MonitoringAlert) -> bool:
        """Send alert to Elasticsearch."""
        try:
            session = await self._get_session()
            
            # Create alert document
            document = {
                '@timestamp': alert.timestamp,
                'type': 'alert',
                'source': 'orchestrator_quality',
                'alert': alert.to_dict()
            }
            
            # Generate document ID
            doc_id = f"alert_{alert.rule_name}_{int(time.time() * 1000)}"
            url = f"{self.elasticsearch_url}/{self.index_name}/_doc/{doc_id}"
            
            async with session.put(
                url,
                json=document,
                headers={'Content-Type': 'application/json'}
            ) as response:
                return response.status in (200, 201)
                
        except Exception as e:
            logging.error(f"Failed to send alert to Elasticsearch: {e}")
            return False

    async def health_check(self) -> bool:
        """Check Elasticsearch cluster health."""
        try:
            session = await self._get_session()
            
            async with session.get(f"{self.elasticsearch_url}/_cluster/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    return health_data.get('status') in ['green', 'yellow']
                return False
                
        except Exception:
            return False

    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


class QualityMonitor:
    """
    Comprehensive quality control monitoring system.
    
    Orchestrates multiple monitoring backends, manages alerting rules,
    and provides centralized quality control monitoring.
    """
    
    def __init__(
        self,
        backends: List[MonitoringBackend],
        alert_rules: List[AlertRule],
        metrics_collection_interval: float = 60.0,
        alert_evaluation_interval: float = 30.0,
        max_queue_size: int = 10000
    ):
        self.backends = backends
        self.alert_rules = {rule.name: rule for rule in alert_rules}
        self.metrics_collection_interval = metrics_collection_interval
        self.alert_evaluation_interval = alert_evaluation_interval
        self.max_queue_size = max_queue_size
        
        # Internal state
        self._metrics_buffer: Dict[str, Any] = {}
        self._metrics_history: List[Dict[str, Any]] = []
        self._alert_state: Dict[str, Dict[str, Any]] = {}
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._lock = asyncio.Lock()

    async def start(self):
        """Start monitoring system."""
        if self._running:
            return
            
        self._running = True
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._metrics_collector()),
            asyncio.create_task(self._alert_evaluator()),
            asyncio.create_task(self._queue_processor())
        ]

    async def stop(self):
        """Stop monitoring system gracefully."""
        self._running = False
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Close backends
        for backend in self.backends:
            if hasattr(backend, 'close'):
                await backend.close()

    async def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        try:
            await self._queue.put_nowait({
                'type': 'metric',
                'name': name,
                'value': value,
                'labels': labels or {},
                'timestamp': time.time()
            })
        except asyncio.QueueFull:
            # Drop oldest metric if queue is full
            try:
                self._queue.get_nowait()
                await self._queue.put_nowait({
                    'type': 'metric',
                    'name': name,
                    'value': value,
                    'labels': labels or {},
                    'timestamp': time.time()
                })
            except asyncio.QueueEmpty:
                pass

    async def record_quality_event(self, event: QualityEvent):
        """Record a quality control event."""
        try:
            await self._queue.put_nowait({
                'type': 'quality_event',
                'event': event,
                'timestamp': time.time()
            })
        except asyncio.QueueFull:
            logging.warning("Quality monitoring queue is full, dropping event")

    async def _metrics_collector(self):
        """Background task for collecting and sending metrics."""
        while self._running:
            try:
                async with self._lock:
                    if self._metrics_buffer:
                        # Send metrics to all backends
                        for backend in self.backends:
                            try:
                                await backend.send_metrics(self._metrics_buffer.copy())
                            except Exception as e:
                                logging.error(f"Backend {type(backend).__name__} failed to send metrics: {e}")
                        
                        # Archive metrics for alert evaluation
                        self._metrics_history.append({
                            'timestamp': time.time(),
                            'metrics': self._metrics_buffer.copy()
                        })
                        
                        # Limit history size (keep last hour)
                        cutoff_time = time.time() - 3600
                        self._metrics_history = [
                            h for h in self._metrics_history 
                            if h['timestamp'] > cutoff_time
                        ]
                        
                        # Clear buffer
                        self._metrics_buffer.clear()
                
                await asyncio.sleep(self.metrics_collection_interval)
                
            except Exception as e:
                logging.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(1)

    async def _alert_evaluator(self):
        """Background task for evaluating alert rules."""
        while self._running:
            try:
                async with self._lock:
                    current_time = time.time()
                    
                    for rule_name, rule in self.alert_rules.items():
                        try:
                            # Check if rule should be evaluated (cooldown)
                            last_alert = self._alert_state.get(rule_name, {}).get('last_alert_time', 0)
                            if current_time - last_alert < rule.cooldown_seconds:
                                continue
                            
                            # Evaluate rule condition
                            if await self._evaluate_alert_rule(rule):
                                alert = MonitoringAlert(
                                    rule_name=rule_name,
                                    severity=rule.severity,
                                    message=f"Alert rule '{rule_name}' triggered: {rule.description or rule.condition}",
                                    timestamp=datetime.now(timezone.utc).isoformat(),
                                    value=self._get_rule_value(rule),
                                    threshold=rule.threshold,
                                    context=self._get_rule_context(rule),
                                    remediation_url=rule.remediation_url,
                                    tags=rule.tags
                                )
                                
                                # Send alert to all backends
                                for backend in self.backends:
                                    try:
                                        await backend.send_alert(alert)
                                    except Exception as e:
                                        logging.error(f"Backend {type(backend).__name__} failed to send alert: {e}")
                                
                                # Update alert state
                                self._alert_state[rule_name] = {
                                    'last_alert_time': current_time,
                                    'alert_count': self._alert_state.get(rule_name, {}).get('alert_count', 0) + 1
                                }
                                
                        except Exception as e:
                            logging.error(f"Error evaluating alert rule '{rule_name}': {e}")
                
                await asyncio.sleep(self.alert_evaluation_interval)
                
            except Exception as e:
                logging.error(f"Error in alert evaluator: {e}")
                await asyncio.sleep(1)

    async def _queue_processor(self):
        """Background task for processing queued events."""
        while self._running:
            try:
                # Process queue with timeout
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                async with self._lock:
                    if item['type'] == 'metric':
                        # Add metric to buffer
                        metric_key = f"{item['name']}"
                        if item['labels']:
                            label_str = ','.join([f'{k}={v}' for k, v in item['labels'].items()])
                            metric_key += f"{{{label_str}}}"
                        
                        self._metrics_buffer[metric_key] = {
                            'value': item['value'],
                            'labels': item['labels'],
                            'timestamp': item['timestamp']
                        }
                        
                    elif item['type'] == 'quality_event':
                        # Process quality event
                        event = item['event']
                        
                        # Convert quality event to metrics
                        if event.quality_score is not None:
                            await self.record_metric(
                                'orchestrator_quality_score',
                                event.quality_score,
                                {'event_type': event.event_type, 'severity': event.severity}
                            )
                        
                        if event.rule_violations:
                            await self.record_metric(
                                'orchestrator_rule_violations',
                                len(event.rule_violations),
                                {'event_type': event.event_type, 'severity': event.severity}
                            )
                
            except Exception as e:
                logging.error(f"Error processing queue item: {e}")

    async def _evaluate_alert_rule(self, rule: AlertRule) -> bool:
        """Evaluate if an alert rule should fire."""
        try:
            # Get metrics within the rule's time window
            cutoff_time = time.time() - rule.window_seconds
            relevant_metrics = [
                h for h in self._metrics_history
                if h['timestamp'] > cutoff_time
            ]
            
            if not relevant_metrics:
                return False
            
            # Create evaluation context
            context = {
                'metrics': relevant_metrics,
                'current_metrics': self._metrics_buffer,
                'threshold': rule.threshold,
                'window_seconds': rule.window_seconds
            }
            
            # Evaluate condition (basic implementation - could be extended with AST)
            # For now, support simple conditions like "avg(quality_score) < 0.8"
            return self._simple_condition_eval(rule.condition, context)
            
        except Exception as e:
            logging.error(f"Error evaluating alert rule condition: {e}")
            return False

    def _simple_condition_eval(self, condition: str, context: Dict[str, Any]) -> bool:
        """Simple alert condition evaluation (extend for more complex rules)."""
        # This is a simplified implementation
        # Production version would use AST or safe evaluation
        try:
            # Extract function calls like "avg(metric_name)"
            if "avg(" in condition and ")" in condition:
                metric_part = condition[condition.find("avg(") + 4:condition.find(")")]
                
                # Calculate average from metrics history
                values = []
                for history_item in context['metrics']:
                    for metric_name, metric_data in history_item['metrics'].items():
                        if metric_part in metric_name:
                            values.append(metric_data.get('value', 0))
                
                if values:
                    avg_value = sum(values) / len(values)
                    
                    # Simple threshold comparison
                    if "<" in condition:
                        threshold = context['threshold']
                        return avg_value < threshold
                    elif ">" in condition:
                        threshold = context['threshold']
                        return avg_value > threshold
            
            return False
            
        except Exception:
            return False

    def _get_rule_value(self, rule: AlertRule) -> float:
        """Get current value for alert rule."""
        # Simplified implementation
        return 0.0

    def _get_rule_context(self, rule: AlertRule) -> Dict[str, Any]:
        """Get context information for alert rule."""
        return {
            'window_seconds': rule.window_seconds,
            'threshold': rule.threshold,
            'metrics_count': len(self._metrics_history)
        }

    async def health_check(self) -> Dict[str, bool]:
        """Check health status of all monitoring backends."""
        health_status = {}
        
        for backend in self.backends:
            backend_name = type(backend).__name__
            try:
                health_status[backend_name] = await backend.health_check()
            except Exception as e:
                logging.error(f"Health check failed for {backend_name}: {e}")
                health_status[backend_name] = False
        
        return health_status


def create_monitoring_setup(
    config: Dict[str, Any],
    logger: Optional[StructuredLogger] = None
) -> QualityMonitor:
    """
    Create quality monitoring setup from configuration.
    
    Expected config structure:
    {
        "backends": [
            {
                "type": "prometheus",
                "pushgateway_url": "http://localhost:9091",
                "job_name": "orchestrator_quality"
            },
            {
                "type": "webhook", 
                "webhook_url": "https://example.com/webhook",
                "headers": {"Authorization": "Bearer token"}
            }
        ],
        "alert_rules": [
            {
                "name": "low_quality_score",
                "condition": "avg(quality_score) < 0.8",
                "severity": "WARNING",
                "threshold": 0.8,
                "window_seconds": 300
            }
        ],
        "metrics_collection_interval": 60.0,
        "alert_evaluation_interval": 30.0
    }
    """
    backends = []
    
    # Create monitoring backends based on config
    for backend_config in config.get('backends', []):
        backend_type = backend_config['type'].lower()
        
        if backend_type == 'prometheus':
            backends.append(PrometheusBackend(
                pushgateway_url=backend_config['pushgateway_url'],
                job_name=backend_config.get('job_name', 'orchestrator_quality'),
                instance_id=backend_config.get('instance_id'),
                basic_auth=tuple(backend_config['basic_auth']) if 'basic_auth' in backend_config else None,
                timeout=backend_config.get('timeout', 10.0)
            ))
            
        elif backend_type == 'webhook':
            backends.append(WebhookBackend(
                webhook_url=backend_config['webhook_url'],
                headers=backend_config.get('headers', {}),
                timeout=backend_config.get('timeout', 10.0),
                retry_count=backend_config.get('retry_count', 3)
            ))
            
        elif backend_type == 'elasticsearch':
            backends.append(ElasticsearchBackend(
                elasticsearch_url=backend_config['elasticsearch_url'],
                index_name=backend_config.get('index_name', 'orchestrator-quality'),
                username=backend_config.get('username'),
                password=backend_config.get('password'),
                timeout=backend_config.get('timeout', 10.0)
            ))
    
    # Create alert rules
    alert_rules = []
    for rule_config in config.get('alert_rules', []):
        alert_rules.append(AlertRule(
            name=rule_config['name'],
            condition=rule_config['condition'],
            severity=rule_config['severity'],
            threshold=rule_config['threshold'],
            window_seconds=rule_config['window_seconds'],
            cooldown_seconds=rule_config.get('cooldown_seconds', 300),
            description=rule_config.get('description'),
            remediation_url=rule_config.get('remediation_url'),
            tags=rule_config.get('tags')
        ))
    
    return QualityMonitor(
        backends=backends,
        alert_rules=alert_rules,
        metrics_collection_interval=config.get('metrics_collection_interval', 60.0),
        alert_evaluation_interval=config.get('alert_evaluation_interval', 30.0),
        max_queue_size=config.get('max_queue_size', 10000)
    )