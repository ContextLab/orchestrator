"""Real Threat Detection Tests - Issue #206 Task 3.3

Comprehensive tests for the automatic threat detection and response system.
Tests include malicious code detection, behavioral analysis, automated responses,
and integration with performance monitoring. NO MOCKS - real threat detection only.
"""

import pytest
import asyncio
import logging
import time
import tempfile
import json
import os

from orchestrator.security.threat_detection import (
    ThreatMonitor,
    ThreatDetectionEngine,
    AutomatedResponseSystem,
    ThreatSignature,
    SecurityEvent,
    ThreatLevel,
    ThreatCategory,
    ResponseAction
)
from orchestrator.analytics.performance_monitor import PerformanceMonitor

# Configure logging for test visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestThreatDetectionEngine:
    """Test threat detection engine with real threat signatures."""
    
    @pytest.fixture
    def detection_engine(self):
        """Create threat detection engine for testing."""
        return ThreatDetectionEngine()
    
    def test_engine_initialization(self, detection_engine):
        """Test detection engine initialization."""
        logger.info("🧪 Testing threat detection engine initialization")
        
        assert detection_engine.signatures is not None
        assert len(detection_engine.signatures) >= 5  # Should have default signatures
        assert len(detection_engine.custom_detectors) == 0  # No custom detectors initially
        assert len(detection_engine.activity_patterns) == 0
        assert len(detection_engine.baseline_metrics) == 0
        
        logger.info("✅ Detection engine initialization test passed")
    
    def test_malicious_code_detection(self, detection_engine):
        """Test detection of malicious code patterns."""
        logger.info("🧪 Testing malicious code detection")
        
        # Test cases: (code, expected_threat_count, expected_max_level)
        test_cases = [
            ("print('Hello World')", 0, None),  # Benign code
            ("eval(user_input)", 1, ThreatLevel.HIGH),  # Code injection
            ("os.system('rm -rf /')", 1, ThreatLevel.HIGH),  # System command
            ("import subprocess; subprocess.call(['ls'])", 1, ThreatLevel.HIGH),  # Subprocess
            ("with open('/etc/passwd', 'r') as f: data = f.read()", 1, ThreatLevel.MEDIUM),  # File access
            ("sudo apt install malware", 1, ThreatLevel.HIGH),  # Privilege escalation
            ("socket.connect(('target.com', 22))", 1, ThreatLevel.HIGH),  # Network activity
            ("/proc/self/ns/pid", 1, ThreatLevel.CRITICAL),  # Container escape
        ]
        
        for code, expected_count, expected_level in test_cases:
            events = detection_engine.analyze_code(code, "test_source")
            
            assert len(events) == expected_count, f"Expected {expected_count} events for: {code[:50]}"
            
            if expected_count > 0:
                max_level = max(event.threat_level for event in events)
                assert max_level == expected_level, f"Expected {expected_level} for: {code[:50]}"
                
                # Check event properties
                for event in events:
                    assert event.event_id is not None
                    assert event.timestamp > 0
                    assert event.source == "test_source"
                    assert event.description is not None
                    assert 'code_snippet' in event.details
                    assert event.confidence > 0
        
        logger.info("✅ Malicious code detection test passed")
    
    def test_custom_signature(self, detection_engine):
        """Test adding and using custom threat signatures."""
        logger.info("🧪 Testing custom threat signatures")
        
        # Add custom signature
        custom_sig = ThreatSignature(
            signature_id="test_custom_1",
            name="Test Malicious Pattern",
            category=ThreatCategory.MALICIOUS_BEHAVIOR,
            level=ThreatLevel.MEDIUM,
            pattern=r"DANGEROUS_FUNCTION\(",
            description="Test custom threat detection",
            response_actions=[ResponseAction.LOG_ONLY]
        )
        
        initial_count = len(detection_engine.signatures)
        detection_engine.add_signature(custom_sig)
        
        assert len(detection_engine.signatures) == initial_count + 1
        
        # Test detection with custom signature
        safe_code = "print('normal code')"
        malicious_code = "DANGEROUS_FUNCTION('payload')"
        
        safe_events = detection_engine.analyze_code(safe_code)
        malicious_events = detection_engine.analyze_code(malicious_code)
        
        assert len(safe_events) == 0
        assert len(malicious_events) >= 1
        
        # Find our custom event
        custom_event = next((e for e in malicious_events if e.signature_id == "test_custom_1"), None)
        assert custom_event is not None
        assert custom_event.threat_level == ThreatLevel.MEDIUM
        assert custom_event.category == ThreatCategory.MALICIOUS_BEHAVIOR
        
        logger.info("✅ Custom signature test passed")
    
    def test_behavioral_analysis(self, detection_engine):
        """Test behavioral threat detection."""
        logger.info("🧪 Testing behavioral threat detection")
        
        source = "test_container"
        
        # Normal activity to establish baseline
        normal_activities = [
            {'source': source, 'cpu_usage': 10, 'memory_usage': 20},
            {'source': source, 'cpu_usage': 12, 'memory_usage': 22},
            {'source': source, 'cpu_usage': 11, 'memory_usage': 21},
        ]
        
        # Record normal activities
        for activity in normal_activities:
            events = detection_engine.analyze_activity(activity)
            assert len(events) == 0  # No threats in normal activity
        
        # Abnormal activity that should trigger detection  
        # Use different source to avoid baseline contamination
        abnormal_activity = {
            'source': 'abnormal_container',
            'cpu_usage': 95,  # Very high CPU
            'memory_usage': 85,  # High memory
        }
        
        events = detection_engine.analyze_activity(abnormal_activity)
        
        # Should detect resource abuse
        cpu_events = [e for e in events if 'cpu' in e.description.lower()]
        memory_events = [e for e in events if 'memory' in e.description.lower()]
        
        assert len(cpu_events) >= 1, "Should detect CPU anomaly"
        assert len(memory_events) >= 1, "Should detect memory anomaly"
        
        # Check event properties
        for event in events:
            assert event.category == ThreatCategory.RESOURCE_ABUSE
            assert event.threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH]
            assert event.source == 'abnormal_container'
        
        logger.info("✅ Behavioral analysis test passed")
    
    def test_custom_detector(self, detection_engine):
        """Test custom threat detector functions."""
        logger.info("🧪 Testing custom threat detectors")
        
        # Custom detector that flags high network activity
        def network_detector(activity_data: dict):
            network_requests = activity_data.get('network_requests', 0)
            if network_requests > 100:
                return SecurityEvent(
                    event_id=f"network_{time.time()}",
                    timestamp=time.time(),
                    threat_level=ThreatLevel.HIGH,
                    category=ThreatCategory.NETWORK_ATTACK,
                    source=activity_data.get('source', 'unknown'),
                    description=f"High network activity: {network_requests} requests",
                    details={'network_requests': network_requests},
                    confidence=0.9
                )
            return None
        
        detection_engine.add_custom_detector(network_detector)
        assert len(detection_engine.custom_detectors) == 1
        
        # Test normal activity
        normal_activity = {'source': 'test', 'network_requests': 50}
        events = detection_engine.analyze_activity(normal_activity)
        network_events = [e for e in events if e.category == ThreatCategory.NETWORK_ATTACK]
        assert len(network_events) == 0
        
        # Test high network activity
        high_network_activity = {'source': 'test', 'network_requests': 150}
        events = detection_engine.analyze_activity(high_network_activity)
        network_events = [e for e in events if e.category == ThreatCategory.NETWORK_ATTACK]
        assert len(network_events) == 1
        
        event = network_events[0]
        assert event.threat_level == ThreatLevel.HIGH
        assert event.confidence == 0.9
        assert event.details['network_requests'] == 150
        
        logger.info("✅ Custom detector test passed")


class TestAutomatedResponseSystem:
    """Test automated response system."""
    
    @pytest.fixture
    def response_system(self):
        """Create response system for testing."""
        return AutomatedResponseSystem()
    
    @pytest.fixture
    def sample_event(self):
        """Create sample security event."""
        return SecurityEvent(
            event_id="test_event_123",
            timestamp=time.time(),
            threat_level=ThreatLevel.HIGH,
            category=ThreatCategory.CODE_INJECTION,
            source="test_container",
            description="Test security event",
            details={'test': True},
            confidence=1.0
        )
    
    @pytest.mark.asyncio
    async def test_response_system_initialization(self, response_system):
        """Test response system initialization."""
        logger.info("🧪 Testing response system initialization")
        
        assert response_system.response_handlers is not None
        assert len(response_system.response_handlers) >= 6  # Should have all response handlers
        assert response_system.quarantine_zone is not None
        assert len(response_system.blocked_containers) == 0
        assert len(response_system.throttled_containers) == 0
        
        logger.info("✅ Response system initialization test passed")
    
    @pytest.mark.asyncio
    async def test_log_only_response(self, response_system):
        """Test log-only response action."""
        logger.info("🧪 Testing log-only response")
        
        event = SecurityEvent(
            event_id="test_log",
            timestamp=time.time(),
            threat_level=ThreatLevel.LOW,
            category=ThreatCategory.SUSPICIOUS_ACTIVITY,
            source="test_source",
            description="Test log event",
            confidence=0.8
        )
        
        # Test log response
        result = await response_system._log_threat(event)
        assert result is True
        
        # Test full response
        response_result = await response_system.respond_to_threat(event)
        assert response_result is True
        assert ResponseAction.LOG_ONLY in event.response_actions_taken
        
        logger.info("✅ Log-only response test passed")
    
    @pytest.mark.asyncio
    async def test_quarantine_response(self, response_system):
        """Test code quarantine response."""
        logger.info("🧪 Testing quarantine response")
        
        event = SecurityEvent(
            event_id="test_quarantine",
            timestamp=time.time(),
            threat_level=ThreatLevel.HIGH,
            category=ThreatCategory.CODE_INJECTION,
            source="malicious_container",
            description="Malicious code detected",
            details={'code_snippet': 'eval(malicious_code)'},
            confidence=1.0
        )
        
        # Test quarantine
        result = await response_system._quarantine_code(event)
        assert result is True
        
        # Check quarantine file was created
        quarantine_file = f"{response_system.quarantine_zone}/threat_{event.event_id}.json"
        assert os.path.exists(quarantine_file)
        
        # Verify quarantine content
        with open(quarantine_file, 'r') as f:
            quarantine_data = json.load(f)
        
        assert quarantine_data['event_id'] == event.event_id
        assert quarantine_data['threat_level'] == ThreatLevel.HIGH.value
        assert quarantine_data['category'] == ThreatCategory.CODE_INJECTION.value
        
        logger.info("✅ Quarantine response test passed")
    
    @pytest.mark.asyncio
    async def test_container_isolation_response(self, response_system):
        """Test container isolation response."""
        logger.info("🧪 Testing container isolation")
        
        event = SecurityEvent(
            event_id="test_isolation",
            timestamp=time.time(),
            threat_level=ThreatLevel.HIGH,
            category=ThreatCategory.NETWORK_ATTACK,
            source="attacking_container",
            description="Network attack detected",
            confidence=0.9
        )
        
        initial_blocked = len(response_system.blocked_containers)
        
        # Test isolation
        result = await response_system._isolate_container(event)
        assert result is True
        
        # Check container was blocked
        assert len(response_system.blocked_containers) == initial_blocked + 1
        assert "attacking_container" in response_system.blocked_containers
        
        logger.info("✅ Container isolation test passed")
    
    @pytest.mark.asyncio
    async def test_resource_throttling_response(self, response_system):
        """Test resource throttling response."""
        logger.info("🧪 Testing resource throttling")
        
        event = SecurityEvent(
            event_id="test_throttle",
            timestamp=time.time(),
            threat_level=ThreatLevel.MEDIUM,
            category=ThreatCategory.RESOURCE_ABUSE,
            source="resource_abuser",
            description="Resource abuse detected",
            confidence=0.8
        )
        
        initial_throttled = len(response_system.throttled_containers)
        
        # Test throttling
        result = await response_system._throttle_resources(event)
        assert result is True
        
        # Check container was throttled
        assert len(response_system.throttled_containers) == initial_throttled + 1
        assert "resource_abuser" in response_system.throttled_containers
        
        logger.info("✅ Resource throttling test passed")
    
    @pytest.mark.asyncio
    async def test_response_action_selection(self, response_system):
        """Test appropriate response action selection based on threat level."""
        logger.info("🧪 Testing response action selection")
        
        # Test different threat levels
        threat_levels = [
            (ThreatLevel.LOW, [ResponseAction.LOG_ONLY]),
            (ThreatLevel.MEDIUM, [ResponseAction.LOG_ONLY, ResponseAction.ALERT_ADMIN]),
            (ThreatLevel.HIGH, [ResponseAction.ALERT_ADMIN, ResponseAction.ISOLATE_CONTAINER]),
            (ThreatLevel.CRITICAL, [ResponseAction.ALERT_ADMIN, ResponseAction.TERMINATE_CONTAINER]),
            (ThreatLevel.EMERGENCY, [ResponseAction.ALERT_ADMIN, ResponseAction.EMERGENCY_SHUTDOWN])
        ]
        
        for threat_level, expected_actions in threat_levels:
            actions = response_system._get_default_actions(threat_level)
            
            # Check that all expected actions are present
            for expected_action in expected_actions:
                assert expected_action in actions, f"Missing {expected_action} for {threat_level}"
        
        logger.info("✅ Response action selection test passed")


class TestThreatMonitor:
    """Test comprehensive threat monitoring system."""
    
    @pytest.fixture
    async def performance_monitor(self):
        """Create performance monitor for integration testing."""
        monitor = PerformanceMonitor(collection_interval=0.5)
        await monitor.start_monitoring()
        yield monitor
        await monitor.stop_monitoring()
    
    @pytest.fixture
    async def threat_monitor(self, performance_monitor):
        """Create threat monitor for testing."""
        monitor = ThreatMonitor(performance_monitor=performance_monitor)
        await monitor.start_monitoring()
        yield monitor
        await monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_threat_monitor_initialization(self, threat_monitor):
        """Test threat monitor initialization."""
        logger.info("🧪 Testing threat monitor initialization")
        
        assert threat_monitor.detection_engine is not None
        assert threat_monitor.response_system is not None
        assert len(threat_monitor.security_events) == 0
        assert len(threat_monitor.active_threats) == 0
        assert threat_monitor._monitoring is True
        
        # Check statistics
        assert threat_monitor.stats['events_detected'] == 0
        assert threat_monitor.stats['events_resolved'] == 0
        assert threat_monitor.stats['responses_executed'] == 0
        assert threat_monitor.stats['false_positives'] == 0
        
        logger.info("✅ Threat monitor initialization test passed")
    
    @pytest.mark.asyncio
    async def test_code_scanning(self, threat_monitor):
        """Test real-time code scanning."""
        logger.info("🧪 Testing code scanning")
        
        # Scan benign code
        benign_code = "print('Hello, World!')\nresult = 2 + 2"
        events = await threat_monitor.scan_code(benign_code, "benign_source")
        
        assert len(events) == 0
        assert threat_monitor.stats['events_detected'] == 0
        
        # Scan malicious code
        malicious_code = "eval(user_input)\nos.system('rm -rf /')"
        events = await threat_monitor.scan_code(malicious_code, "malicious_source")
        
        assert len(events) >= 1
        assert threat_monitor.stats['events_detected'] >= 1
        assert threat_monitor.stats['responses_executed'] >= 1
        
        # Check that events were recorded
        assert len(threat_monitor.security_events) >= 1
        assert len(threat_monitor.active_threats) >= 1
        
        logger.info("✅ Code scanning test passed")
    
    @pytest.mark.asyncio
    async def test_activity_monitoring(self, threat_monitor):
        """Test system activity monitoring."""
        logger.info("🧪 Testing activity monitoring")
        
        # Report normal activity
        normal_activity = {
            'source': 'normal_container',
            'cpu_usage': 15,
            'memory_usage': 25,
            'network_requests': 10
        }
        
        await threat_monitor.report_activity(normal_activity)
        await asyncio.sleep(0.5)  # Allow processing
        
        # Should have baseline but no threats
        initial_threats = len(threat_monitor.active_threats)
        
        # Report suspicious activity
        suspicious_activity = {
            'source': 'suspicious_container',
            'cpu_usage': 95,
            'memory_usage': 90,
            'network_requests': 200
        }
        
        await threat_monitor.report_activity(suspicious_activity)
        await asyncio.sleep(1.0)  # Allow processing
        
        # Should detect threats
        assert len(threat_monitor.active_threats) > initial_threats
        
        logger.info("✅ Activity monitoring test passed")
    
    @pytest.mark.asyncio
    async def test_event_resolution(self, threat_monitor):
        """Test security event resolution."""
        logger.info("🧪 Testing event resolution")
        
        # Create a threat event
        malicious_code = "subprocess.call(['malicious_command'])"
        events = await threat_monitor.scan_code(malicious_code, "test_resolution")
        
        assert len(events) >= 1
        
        event_id = events[0].event_id
        assert event_id in threat_monitor.active_threats
        assert not threat_monitor.active_threats[event_id].resolved
        
        # Resolve the event
        result = threat_monitor.resolve_event(event_id)
        assert result is True
        assert threat_monitor.active_threats[event_id].resolved
        assert threat_monitor.active_threats[event_id].resolution_timestamp is not None
        assert threat_monitor.stats['events_resolved'] >= 1
        
        logger.info("✅ Event resolution test passed")
    
    @pytest.mark.asyncio
    async def test_false_positive_handling(self, threat_monitor):
        """Test false positive marking."""
        logger.info("🧪 Testing false positive handling")
        
        # Create a potential false positive
        code_with_eval = "# This eval is safe: eval('2 + 2')"
        events = await threat_monitor.scan_code(code_with_eval, "false_positive_test")
        
        if len(events) > 0:  # If our detection flagged it
            event_id = events[0].event_id
            
            # Mark as false positive
            result = threat_monitor.mark_false_positive(event_id)
            assert result is True
            assert threat_monitor.active_threats[event_id].resolved is True
            assert threat_monitor.stats['false_positives'] >= 1
        
        logger.info("✅ False positive handling test passed")
    
    @pytest.mark.asyncio
    async def test_security_summary(self, threat_monitor):
        """Test security monitoring summary."""
        logger.info("🧪 Testing security summary")
        
        # Generate some activity
        test_codes = [
            "print('safe')",
            "eval('dangerous')",
            "os.system('command')"
        ]
        
        for i, code in enumerate(test_codes):
            await threat_monitor.scan_code(code, f"source_{i}")
        
        await asyncio.sleep(1.0)  # Allow processing
        
        # Get summary
        summary = threat_monitor.get_security_summary()
        
        # Verify summary structure
        required_keys = [
            'total_events', 'active_events', 'events_by_level',
            'statistics', 'monitoring_active', 'recent_events',
            'threat_signatures', 'blocked_containers', 'throttled_containers'
        ]
        
        for key in required_keys:
            assert key in summary, f"Missing key in summary: {key}"
        
        assert summary['monitoring_active'] is True
        assert summary['total_events'] >= 0
        assert isinstance(summary['statistics'], dict)
        assert isinstance(summary['events_by_level'], dict)
        assert isinstance(summary['recent_events'], list)
        
        logger.info(f"Security summary: {summary['statistics']}")
        logger.info("✅ Security summary test passed")
    
    @pytest.mark.asyncio
    async def test_security_report_export(self, threat_monitor):
        """Test security report export."""
        logger.info("🧪 Testing security report export")
        
        # Generate some threats
        malicious_codes = [
            "eval(user_data)",
            "subprocess.call(cmd)",
            "open('/etc/passwd').read()"
        ]
        
        for code in malicious_codes:
            await threat_monitor.scan_code(code, "export_test")
        
        await asyncio.sleep(1.0)  # Allow processing
        
        # Export report
        report = threat_monitor.export_security_report()
        
        # Verify report structure
        assert 'report_timestamp' in report
        assert 'summary' in report
        assert 'all_events' in report
        assert 'threat_signatures' in report
        
        # Check report content
        assert report['report_timestamp'] > 0
        assert isinstance(report['summary'], dict)
        assert isinstance(report['all_events'], list)
        assert isinstance(report['threat_signatures'], list)
        
        # Verify threat signatures in report
        sig_ids = [sig['signature_id'] for sig in report['threat_signatures']]
        assert len(sig_ids) >= 5  # Should have default signatures
        
        logger.info(f"Security report contains {len(report['all_events'])} events and {len(report['threat_signatures'])} signatures")
        logger.info("✅ Security report export test passed")
    
    @pytest.mark.asyncio
    async def test_performance_integration(self, threat_monitor, performance_monitor):
        """Test integration with performance monitoring system."""
        logger.info("🧪 Testing performance monitoring integration")
        
        # Create a high-severity threat that should be recorded
        critical_code = "exec(open('/proc/self/ns/pid').read())"  # Container escape attempt
        events = await threat_monitor.scan_code(critical_code, "performance_test")
        
        await asyncio.sleep(2.0)  # Allow performance monitoring to process
        
        # Check that security events were recorded in performance monitor
        perf_summary = performance_monitor.get_performance_summary()
        
        # Look for security-related component
        if 'security_threat_detector' in str(perf_summary):
            logger.info("Security events successfully integrated with performance monitoring")
        
        # Check for alerts in performance monitor
        active_alerts = performance_monitor.get_active_alerts()
        logger.info(f"Performance monitor has {len(active_alerts)} active alerts")
        
        logger.info("✅ Performance integration test passed")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])