#!/usr/bin/env python3
"""
Threat Detection Integration Test - Task 3.3 Validation

Quick integration test to validate threat detection works with the Docker manager
and multi-language execution system.
"""

import asyncio
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_threat_detection_integration():
    """Test threat detection with real code execution."""
    from orchestrator.security.threat_detection import ThreatMonitor
    from orchestrator.security.docker_manager import EnhancedDockerManager
    from orchestrator.analytics.performance_monitor import PerformanceMonitor
    from orchestrator.tools.multi_language_executor import MultiLanguageExecutor, Language
    
    logger.info("🧪 Testing threat detection integration...")
    
    # Create components
    performance_monitor = PerformanceMonitor(collection_interval=0.5)
    docker_manager = EnhancedDockerManager(enable_advanced_pooling=True, performance_monitor=performance_monitor)
    threat_monitor = ThreatMonitor(docker_manager, performance_monitor)
    
    try:
        # Start monitoring systems
        await performance_monitor.start_monitoring()
        await docker_manager.start_background_tasks()
        await threat_monitor.start_monitoring()
        
        logger.info("✅ All monitoring systems started")
        
        # Test 1: Scan safe code
        safe_code = "print('Hello, safe world!')\nresult = 2 + 2"
        safe_events = await threat_monitor.scan_code(safe_code, "safe_test")
        
        logger.info(f"Safe code generated {len(safe_events)} threat events")
        assert len(safe_events) == 0, "Safe code should not trigger threats"
        
        # Test 2: Scan malicious code
        malicious_code = """
import subprocess
import os

# This is clearly malicious
eval(user_input)
os.system('rm -rf /')
subprocess.call(['malicious_command'])
"""
        
        malicious_events = await threat_monitor.scan_code(malicious_code, "malicious_test")
        
        logger.info(f"Malicious code generated {len(malicious_events)} threat events")
        assert len(malicious_events) >= 1, "Malicious code should trigger threats"
        
        # Test 3: Report suspicious system activity
        suspicious_activity = {
            'source': 'suspicious_container',
            'cpu_usage': 95,
            'memory_usage': 90,
            'network_requests': 200
        }
        
        await threat_monitor.report_activity(suspicious_activity)
        await asyncio.sleep(1.0)  # Allow processing
        
        # Test 4: Check security summary
        summary = threat_monitor.get_security_summary()
        
        logger.info(f"Security Summary:")
        logger.info(f"  - Total events: {summary['total_events']}")
        logger.info(f"  - Active events: {summary['active_events']}")
        logger.info(f"  - Events by level: {summary['events_by_level']}")
        logger.info(f"  - Statistics: {summary['statistics']}")
        
        assert summary['total_events'] > 0, "Should have detected some threats"
        assert summary['statistics']['events_detected'] > 0, "Should have processed events"
        assert summary['statistics']['responses_executed'] > 0, "Should have executed responses"
        
        # Test 5: Try to execute potentially dangerous code with multi-language executor
        executor = MultiLanguageExecutor(docker_manager)
        
        # This should work but be monitored
        test_code = "print('Testing with executor')"
        result = await executor.execute_code(test_code, Language.PYTHON, timeout=30)
        
        logger.info(f"Code execution result: {'✅' if result.success else '❌'}")
        
        # Wait a bit for threat detection to process any container activity
        await asyncio.sleep(2.0)
        
        # Final summary
        final_summary = threat_monitor.get_security_summary()
        logger.info(f"Final security state: {final_summary['statistics']}")
        
        logger.info("✅ Threat detection integration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Threat detection integration test failed: {e}")
        return False
        
    finally:
        # Cleanup
        await threat_monitor.stop_monitoring()
        await docker_manager.shutdown()
        await performance_monitor.stop_monitoring()

async def main():
    """Run threat detection integration test."""
    logger.info("🚀 Starting Threat Detection Integration Test...")
    
    success = await test_threat_detection_integration()
    
    if success:
        logger.info("🎉 THREAT DETECTION INTEGRATION TEST PASSED!")
    else:
        logger.info("⚠️  THREAT DETECTION INTEGRATION TEST FAILED!")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)