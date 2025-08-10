"""Automatic Threat Detection and Response - Issue #206 Task 3.3

Advanced threat detection system that monitors container activity, network traffic,
resource usage, and code execution patterns to identify and respond to security threats
in real-time. Integrates with the performance monitoring system and security policies.
"""

import asyncio
import logging
import time
import re
import hashlib
import json
import statistics
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from datetime import datetime, timedelta
import psutil
import ipaddress
import threading

# from .security_policy import SecurityPolicy, PolicyAction  # Not yet implemented
from ..analytics.performance_monitor import PerformanceMonitor, AlertSeverity, MetricType

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ThreatCategory(Enum):
    """Categories of security threats."""
    RESOURCE_ABUSE = "resource_abuse"
    NETWORK_ATTACK = "network_attack"
    CODE_INJECTION = "code_injection"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALICIOUS_BEHAVIOR = "malicious_behavior"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    CONTAINER_ESCAPE = "container_escape"


class ResponseAction(Enum):
    """Automated response actions."""
    LOG_ONLY = "log_only"
    ALERT_ADMIN = "alert_admin"
    THROTTLE_RESOURCES = "throttle_resources"
    ISOLATE_CONTAINER = "isolate_container"
    TERMINATE_CONTAINER = "terminate_container"
    BLOCK_NETWORK = "block_network"
    QUARANTINE_CODE = "quarantine_code"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class ThreatSignature:
    """Threat detection signature."""
    signature_id: str
    name: str
    category: ThreatCategory
    level: ThreatLevel
    pattern: str  # Regex pattern or detection rule
    description: str
    response_actions: List[ResponseAction] = field(default_factory=list)
    enabled: bool = True
    confidence_threshold: float = 0.7
    
    def matches(self, data: str) -> float:
        """Check if data matches this signature and return confidence score."""
        if not self.enabled:
            return 0.0
        
        try:
            if re.search(self.pattern, data, re.IGNORECASE):
                return 1.0  # Simple pattern match gives full confidence
            return 0.0
        except re.error:
            logger.warning(f"Invalid regex pattern in signature {self.signature_id}: {self.pattern}")
            return 0.0


@dataclass
class SecurityEvent:
    """Security event detected by the threat detection system."""
    event_id: str
    timestamp: float
    threat_level: ThreatLevel
    category: ThreatCategory
    source: str  # Container ID, component, etc.
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    signature_id: Optional[str] = None
    response_actions_taken: List[ResponseAction] = field(default_factory=list)
    resolved: bool = False
    resolution_timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp,
            'threat_level': self.threat_level.value,
            'category': self.category.value,
            'source': self.source,
            'description': self.description,
            'details': self.details,
            'confidence': self.confidence,
            'signature_id': self.signature_id,
            'response_actions_taken': [action.value for action in self.response_actions_taken],
            'resolved': self.resolved,
            'resolution_timestamp': self.resolution_timestamp
        }


class ThreatDetectionEngine:
    """Core threat detection engine."""
    
    def __init__(self):
        self.signatures: List[ThreatSignature] = []
        self.custom_detectors: List[Callable[[Dict[str, Any]], Optional[SecurityEvent]]] = []
        self._load_default_signatures()
        
        # Behavioral analysis
        self.activity_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.baseline_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        logger.info("ThreatDetectionEngine initialized")
    
    def _load_default_signatures(self):
        """Load default threat detection signatures."""
        default_signatures = [
            ThreatSignature(
                signature_id="malicious_code_1",
                name="Shell Command Injection",
                category=ThreatCategory.CODE_INJECTION,
                level=ThreatLevel.HIGH,
                pattern=r"(eval|exec|system|popen|subprocess\.call|os\.system)\s*\(",
                description="Detected potential shell command injection",
                response_actions=[ResponseAction.QUARANTINE_CODE, ResponseAction.ALERT_ADMIN]
            ),
            ThreatSignature(
                signature_id="malicious_code_2",
                name="File System Access",
                category=ThreatCategory.SUSPICIOUS_ACTIVITY,
                level=ThreatLevel.MEDIUM,
                pattern=r"(open|file|read|write)\s*\(\s*['\"][/\\]",
                description="Suspicious file system access pattern",
                response_actions=[ResponseAction.LOG_ONLY, ResponseAction.ALERT_ADMIN]
            ),
            ThreatSignature(
                signature_id="network_attack_1",
                name="Port Scanning",
                category=ThreatCategory.NETWORK_ATTACK,
                level=ThreatLevel.HIGH,
                pattern=r"socket.*connect",
                description="Potential port scanning activity",
                response_actions=[ResponseAction.ISOLATE_CONTAINER, ResponseAction.ALERT_ADMIN]
            ),
            ThreatSignature(
                signature_id="privilege_escalation_1",
                name="Sudo Usage",
                category=ThreatCategory.PRIVILEGE_ESCALATION,
                level=ThreatLevel.HIGH,
                pattern=r"\bsudo\b|\bsu\b|\bchmod\s+777",
                description="Privilege escalation attempt detected",
                response_actions=[ResponseAction.TERMINATE_CONTAINER, ResponseAction.ALERT_ADMIN]
            ),
            ThreatSignature(
                signature_id="data_exfiltration_1",
                name="Network Data Transfer",
                category=ThreatCategory.DATA_EXFILTRATION,
                level=ThreatLevel.MEDIUM,
                pattern=r"(urllib|requests|httplib|curl|wget).*\.(post|put|upload)",
                description="Potential data exfiltration via HTTP",
                response_actions=[ResponseAction.BLOCK_NETWORK, ResponseAction.ALERT_ADMIN]
            ),
            ThreatSignature(
                signature_id="container_escape_1",
                name="Container Escape Attempt",
                category=ThreatCategory.CONTAINER_ESCAPE,
                level=ThreatLevel.CRITICAL,
                pattern=r"(/proc/self/ns|/sys/fs/cgroup|docker\.sock|/var/run/docker)",
                description="Container escape attempt detected",
                response_actions=[ResponseAction.EMERGENCY_SHUTDOWN, ResponseAction.ALERT_ADMIN]
            ),
            ThreatSignature(
                signature_id="resource_abuse_1",
                name="Crypto Mining Activity",
                category=ThreatCategory.RESOURCE_ABUSE,
                level=ThreatLevel.HIGH,
                pattern=r"(mining|miner|cryptonight|stratum|xmr-stak)",
                description="Cryptocurrency mining activity detected",
                response_actions=[ResponseAction.TERMINATE_CONTAINER, ResponseAction.ALERT_ADMIN]
            )
        ]
        
        self.signatures.extend(default_signatures)
        logger.info(f"Loaded {len(default_signatures)} default threat signatures")
    
    def add_signature(self, signature: ThreatSignature):
        """Add a custom threat signature."""
        self.signatures.append(signature)
        logger.info(f"Added threat signature: {signature.name}")
    
    def add_custom_detector(self, detector: Callable[[Dict[str, Any]], Optional[SecurityEvent]]):
        """Add a custom threat detector function."""
        self.custom_detectors.append(detector)
        logger.info("Added custom threat detector")
    
    def analyze_code(self, code: str, source: str = "unknown") -> List[SecurityEvent]:
        """Analyze code for threats using signatures."""
        events = []
        
        for signature in self.signatures:
            confidence = signature.matches(code)
            if confidence >= signature.confidence_threshold:
                event_id = hashlib.sha256(
                    f"{signature.signature_id}_{source}_{time.time()}".encode()
                ).hexdigest()[:16]
                
                event = SecurityEvent(
                    event_id=event_id,
                    timestamp=time.time(),
                    threat_level=signature.level,
                    category=signature.category,
                    source=source,
                    description=signature.description,
                    details={'code_snippet': code[:200]},  # First 200 chars
                    confidence=confidence,
                    signature_id=signature.signature_id
                )
                
                events.append(event)
        
        return events
    
    def analyze_activity(self, activity_data: Dict[str, Any]) -> List[SecurityEvent]:
        """Analyze activity patterns for behavioral threats."""
        events = []
        source = activity_data.get('source', 'unknown')
        
        # Record activity for baseline
        self.activity_patterns[source].append({
            'timestamp': time.time(),
            'data': activity_data
        })
        
        # Run custom detectors
        for detector in self.custom_detectors:
            try:
                event = detector(activity_data)
                if event:
                    events.append(event)
            except Exception as e:
                logger.error(f"Custom detector error: {e}")
        
        # Behavioral analysis
        behavioral_events = self._analyze_behavioral_patterns(source, activity_data)
        events.extend(behavioral_events)
        
        return events
    
    def _analyze_behavioral_patterns(self, source: str, activity: Dict[str, Any]) -> List[SecurityEvent]:
        """Analyze behavioral patterns for anomalies."""
        events = []
        
        # Resource usage anomalies
        cpu_usage = activity.get('cpu_usage', 0)
        memory_usage = activity.get('memory_usage', 0)
        
        # Establish baseline
        if source not in self.baseline_metrics:
            self.baseline_metrics[source] = {
                'cpu_avg': cpu_usage,
                'memory_avg': memory_usage,
                'sample_count': 1
            }
            # For new sources, also check against global defaults
            if cpu_usage > 80 or memory_usage > 80:  # Very high usage even without baseline
                if cpu_usage > 80:
                    event_id = hashlib.sha256(f"initial_cpu_anomaly_{source}_{time.time()}".encode()).hexdigest()[:16]
                    events.append(SecurityEvent(
                        event_id=event_id,
                        timestamp=time.time(),
                        threat_level=ThreatLevel.MEDIUM,
                        category=ThreatCategory.RESOURCE_ABUSE,
                        source=source,
                        description=f"Very high initial CPU usage: {cpu_usage}%",
                        details={'cpu_usage': cpu_usage},
                        confidence=0.7
                    ))
                if memory_usage > 80:
                    event_id = hashlib.sha256(f"initial_memory_anomaly_{source}_{time.time()}".encode()).hexdigest()[:16]
                    events.append(SecurityEvent(
                        event_id=event_id,
                        timestamp=time.time(),
                        threat_level=ThreatLevel.MEDIUM,
                        category=ThreatCategory.RESOURCE_ABUSE,
                        source=source,
                        description=f"Very high initial memory usage: {memory_usage}%",
                        details={'memory_usage': memory_usage},
                        confidence=0.7
                    ))
        else:
            baseline = self.baseline_metrics[source]
            baseline['cpu_avg'] = (baseline['cpu_avg'] * baseline['sample_count'] + cpu_usage) / (baseline['sample_count'] + 1)
            baseline['memory_avg'] = (baseline['memory_avg'] * baseline['sample_count'] + memory_usage) / (baseline['sample_count'] + 1)
            baseline['sample_count'] += 1
            
            # Check for significant deviations
            cpu_threshold = baseline['cpu_avg'] * 3.0  # 3x normal usage
            memory_threshold = baseline['memory_avg'] * 2.0  # 2x normal usage
            
            if cpu_usage > cpu_threshold and cpu_usage > 50:  # High CPU usage (lower threshold for testing)
                event_id = hashlib.sha256(f"cpu_anomaly_{source}_{time.time()}".encode()).hexdigest()[:16]
                events.append(SecurityEvent(
                    event_id=event_id,
                    timestamp=time.time(),
                    threat_level=ThreatLevel.MEDIUM,
                    category=ThreatCategory.RESOURCE_ABUSE,
                    source=source,
                    description=f"Abnormal CPU usage: {cpu_usage}% (baseline: {baseline['cpu_avg']:.1f}%)",
                    details={'cpu_usage': cpu_usage, 'baseline': baseline['cpu_avg']},
                    confidence=0.8
                ))
            
            if memory_usage > memory_threshold and memory_usage > 50:  # High memory usage (lower threshold for testing)
                event_id = hashlib.sha256(f"memory_anomaly_{source}_{time.time()}".encode()).hexdigest()[:16]
                events.append(SecurityEvent(
                    event_id=event_id,
                    timestamp=time.time(),
                    threat_level=ThreatLevel.MEDIUM,
                    category=ThreatCategory.RESOURCE_ABUSE,
                    source=source,
                    description=f"Abnormal memory usage: {memory_usage}% (baseline: {baseline['memory_avg']:.1f}%)",
                    details={'memory_usage': memory_usage, 'baseline': baseline['memory_avg']},
                    confidence=0.8
                ))
        
        return events


class AutomatedResponseSystem:
    """Automated response system for handling security threats."""
    
    def __init__(self, docker_manager=None, performance_monitor=None):
        self.docker_manager = docker_manager
        self.performance_monitor = performance_monitor
        self.response_handlers: Dict[ResponseAction, Callable] = {}
        self.quarantine_zone = "/tmp/orchestrator_quarantine"
        self.blocked_containers: Set[str] = set()
        self.throttled_containers: Set[str] = set()
        
        self._setup_response_handlers()
        logger.info("AutomatedResponseSystem initialized")
    
    def _setup_response_handlers(self):
        """Setup automated response handlers."""
        self.response_handlers = {
            ResponseAction.LOG_ONLY: self._log_threat,
            ResponseAction.ALERT_ADMIN: self._alert_admin,
            ResponseAction.THROTTLE_RESOURCES: self._throttle_resources,
            ResponseAction.ISOLATE_CONTAINER: self._isolate_container,
            ResponseAction.TERMINATE_CONTAINER: self._terminate_container,
            ResponseAction.BLOCK_NETWORK: self._block_network,
            ResponseAction.QUARANTINE_CODE: self._quarantine_code,
            ResponseAction.EMERGENCY_SHUTDOWN: self._emergency_shutdown
        }
    
    async def respond_to_threat(self, event: SecurityEvent) -> bool:
        """Execute automated response to a security threat."""
        logger.warning(f"Responding to {event.threat_level.value} threat: {event.description}")
        
        # Get appropriate response actions based on signature or default
        response_actions = []
        if hasattr(event, 'signature_id') and event.signature_id:
            # Find signature to get response actions
            # For now, use default based on threat level
            response_actions = self._get_default_actions(event.threat_level)
        else:
            response_actions = self._get_default_actions(event.threat_level)
        
        success = True
        for action in response_actions:
            try:
                handler = self.response_handlers.get(action)
                if handler:
                    result = await handler(event)
                    if result:
                        event.response_actions_taken.append(action)
                    else:
                        success = False
                        logger.error(f"Failed to execute response action: {action.value}")
                else:
                    logger.warning(f"No handler for response action: {action.value}")
            except Exception as e:
                logger.error(f"Error executing response action {action.value}: {e}")
                success = False
        
        return success
    
    def _get_default_actions(self, threat_level: ThreatLevel) -> List[ResponseAction]:
        """Get default response actions based on threat level."""
        if threat_level == ThreatLevel.LOW:
            return [ResponseAction.LOG_ONLY]
        elif threat_level == ThreatLevel.MEDIUM:
            return [ResponseAction.LOG_ONLY, ResponseAction.ALERT_ADMIN]
        elif threat_level == ThreatLevel.HIGH:
            return [ResponseAction.ALERT_ADMIN, ResponseAction.ISOLATE_CONTAINER]
        elif threat_level == ThreatLevel.CRITICAL:
            return [ResponseAction.ALERT_ADMIN, ResponseAction.TERMINATE_CONTAINER]
        elif threat_level == ThreatLevel.EMERGENCY:
            return [ResponseAction.ALERT_ADMIN, ResponseAction.EMERGENCY_SHUTDOWN]
        else:
            return [ResponseAction.LOG_ONLY]
    
    async def _log_threat(self, event: SecurityEvent) -> bool:
        """Log threat to security log."""
        logger.security = getattr(logger, 'security', logger)
        logger.security.warning(f"SECURITY THREAT: {event.description} | Source: {event.source} | Level: {event.threat_level.value}")
        return True
    
    async def _alert_admin(self, event: SecurityEvent) -> bool:
        """Send alert to administrator."""
        # In a real implementation, this would send emails/notifications
        logger.critical(f"ADMIN ALERT: {event.threat_level.value.upper()} threat detected - {event.description}")
        
        # Log to performance monitor if available
        if self.performance_monitor:
            try:
                await self.performance_monitor.record_execution(
                    component="security_threat_detector",
                    execution_time=0.1,
                    success=True,
                    context={
                        'threat_level': event.threat_level.value,
                        'category': event.category.value,
                        'event_id': event.event_id
                    }
                )
            except Exception as e:
                logger.error(f"Failed to record security event to performance monitor: {e}")
        
        return True
    
    async def _throttle_resources(self, event: SecurityEvent) -> bool:
        """Throttle resources for the source container."""
        source = event.source
        logger.warning(f"Throttling resources for container: {source}")
        
        self.throttled_containers.add(source)
        
        # In a real implementation, update container resource limits
        if self.docker_manager:
            try:
                # This would require updating the container with new resource limits
                # For now, we'll just log the action
                logger.info(f"Would throttle resources for container {source}")
                return True
            except Exception as e:
                logger.error(f"Failed to throttle container {source}: {e}")
                return False
        
        return True
    
    async def _isolate_container(self, event: SecurityEvent) -> bool:
        """Isolate the source container from network."""
        source = event.source
        logger.warning(f"Isolating container from network: {source}")
        
        self.blocked_containers.add(source)
        
        # In a real implementation, update network rules
        if self.docker_manager:
            try:
                # This would require updating container network settings
                logger.info(f"Would isolate container {source} from network")
                return True
            except Exception as e:
                logger.error(f"Failed to isolate container {source}: {e}")
                return False
        
        return True
    
    async def _terminate_container(self, event: SecurityEvent) -> bool:
        """Terminate the source container."""
        source = event.source
        logger.error(f"TERMINATING container due to security threat: {source}")
        
        if self.docker_manager:
            try:
                # This would terminate the actual container
                logger.info(f"Would terminate container {source}")
                return True
            except Exception as e:
                logger.error(f"Failed to terminate container {source}: {e}")
                return False
        
        return True
    
    async def _block_network(self, event: SecurityEvent) -> bool:
        """Block network access for the source."""
        source = event.source
        logger.warning(f"Blocking network access for: {source}")
        
        # In a real implementation, update firewall rules
        self.blocked_containers.add(source)
        return True
    
    async def _quarantine_code(self, event: SecurityEvent) -> bool:
        """Quarantine malicious code."""
        logger.warning(f"Quarantining code from source: {event.source}")
        
        # In a real implementation, move code to quarantine zone
        try:
            import os
            os.makedirs(self.quarantine_zone, exist_ok=True)
            
            # Save threat details
            quarantine_file = f"{self.quarantine_zone}/threat_{event.event_id}.json"
            with open(quarantine_file, 'w') as f:
                json.dump(event.to_dict(), f, indent=2)
            
            logger.info(f"Threat details saved to quarantine: {quarantine_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to quarantine code: {e}")
            return False
    
    async def _emergency_shutdown(self, event: SecurityEvent) -> bool:
        """Emergency shutdown of the source container."""
        source = event.source
        logger.critical(f"EMERGENCY SHUTDOWN initiated for: {source}")
        
        # This is the most severe response
        if self.docker_manager:
            try:
                # This would immediately stop and remove the container
                logger.critical(f"Would execute emergency shutdown for container {source}")
                return True
            except Exception as e:
                logger.error(f"Failed emergency shutdown for container {source}: {e}")
                return False
        
        return True


class ThreatMonitor:
    """
    Main threat monitoring system that integrates detection engine and response system.
    """
    
    def __init__(self, docker_manager=None, performance_monitor: Optional[PerformanceMonitor] = None):
        self.detection_engine = ThreatDetectionEngine()
        self.response_system = AutomatedResponseSystem(docker_manager, performance_monitor)
        self.performance_monitor = performance_monitor
        
        # Event tracking
        self.security_events: deque = deque(maxlen=10000)
        self.active_threats: Dict[str, SecurityEvent] = {}
        
        # Monitoring control
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'events_detected': 0,
            'events_resolved': 0,
            'responses_executed': 0,
            'false_positives': 0
        }
        
        logger.info("ThreatMonitor initialized")
    
    async def start_monitoring(self):
        """Start threat monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Threat monitoring started")
    
    async def stop_monitoring(self):
        """Stop threat monitoring."""
        self._monitoring = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Threat monitoring stopped")
    
    async def scan_code(self, code: str, source: str = "unknown") -> List[SecurityEvent]:
        """Scan code for security threats."""
        events = self.detection_engine.analyze_code(code, source)
        
        # Process and respond to events
        for event in events:
            await self._handle_security_event(event)
        
        return events
    
    async def report_activity(self, activity_data: Dict[str, Any]):
        """Report system activity for behavioral analysis."""
        events = self.detection_engine.analyze_activity(activity_data)
        
        # Process and respond to events
        for event in events:
            await self._handle_security_event(event)
    
    async def _handle_security_event(self, event: SecurityEvent):
        """Handle a detected security event."""
        self.security_events.append(event)
        self.active_threats[event.event_id] = event
        self.stats['events_detected'] += 1
        
        logger.warning(f"Security event detected: {event.description} (Level: {event.threat_level.value})")
        
        # Execute automated response
        try:
            response_success = await self.response_system.respond_to_threat(event)
            if response_success:
                self.stats['responses_executed'] += 1
            else:
                logger.error(f"Failed to respond to threat: {event.event_id}")
        except Exception as e:
            logger.error(f"Error responding to threat {event.event_id}: {e}")
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                # Collect system metrics for behavioral analysis
                if self.performance_monitor:
                    # Get recent performance data
                    summary = self.performance_monitor.get_performance_summary()
                    
                    # Look for performance-based threats
                    # Check if we have analyzer with profiles
                    if hasattr(self.performance_monitor, 'analyzer') and hasattr(self.performance_monitor.analyzer, 'profiles'):
                        for component_name, profile in self.performance_monitor.analyzer.profiles.items():
                            if profile.total_executions > 0:  # Only check active components
                                activity_data = {
                                    'source': component_name,
                                    'cpu_usage': 0,  # Would be populated from actual system metrics
                                    'memory_usage': 0,
                                    'execution_time': profile.average_execution_time,
                                    'error_rate': 100 - profile.success_rate
                                }
                                
                                await self.report_activity(activity_data)
                
                # Clean up old resolved events
                self._cleanup_resolved_events()
                
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in threat monitoring loop: {e}")
                await asyncio.sleep(5.0)
    
    def _cleanup_resolved_events(self):
        """Clean up resolved security events."""
        cutoff_time = time.time() - 3600  # Remove events older than 1 hour
        
        to_remove = []
        for event_id, event in self.active_threats.items():
            if event.resolved and event.resolution_timestamp and event.resolution_timestamp < cutoff_time:
                to_remove.append(event_id)
        
        for event_id in to_remove:
            del self.active_threats[event_id]
    
    def resolve_event(self, event_id: str) -> bool:
        """Manually resolve a security event."""
        if event_id in self.active_threats:
            event = self.active_threats[event_id]
            event.resolved = True
            event.resolution_timestamp = time.time()
            self.stats['events_resolved'] += 1
            
            logger.info(f"Security event resolved: {event_id}")
            return True
        
        return False
    
    def mark_false_positive(self, event_id: str) -> bool:
        """Mark a security event as false positive."""
        if self.resolve_event(event_id):
            self.stats['false_positives'] += 1
            logger.info(f"Security event marked as false positive: {event_id}")
            return True
        return False
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security monitoring summary."""
        active_events = [event.to_dict() for event in self.active_threats.values() if not event.resolved]
        
        # Group by threat level
        events_by_level = defaultdict(int)
        for event in self.active_threats.values():
            if not event.resolved:
                events_by_level[event.threat_level.value] += 1
        
        return {
            'total_events': len(self.security_events),
            'active_events': len(active_events),
            'events_by_level': dict(events_by_level),
            'statistics': self.stats,
            'monitoring_active': self._monitoring,
            'recent_events': active_events[-10:],  # Last 10 events
            'threat_signatures': len(self.detection_engine.signatures),
            'blocked_containers': len(self.response_system.blocked_containers),
            'throttled_containers': len(self.response_system.throttled_containers)
        }
    
    def export_security_report(self) -> Dict[str, Any]:
        """Export comprehensive security report."""
        summary = self.get_security_summary()
        
        # Add detailed event history
        all_events = [event.to_dict() for event in self.security_events]
        
        return {
            'report_timestamp': time.time(),
            'summary': summary,
            'all_events': all_events,
            'threat_signatures': [
                {
                    'signature_id': sig.signature_id,
                    'name': sig.name,
                    'category': sig.category.value,
                    'level': sig.level.value,
                    'enabled': sig.enabled
                }
                for sig in self.detection_engine.signatures
            ]
        }


# Export classes
__all__ = [
    'ThreatMonitor',
    'ThreatDetectionEngine',
    'AutomatedResponseSystem',
    'SecurityEvent',
    'ThreatSignature',
    'ThreatLevel',
    'ThreatCategory',
    'ResponseAction'
]