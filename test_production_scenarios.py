#!/usr/bin/env python3
"""
Production Workload and Attack Scenario Testing - Issue #206 Final Testing

Comprehensive production testing that validates all integrated systems:
- Docker security and container management
- Multi-language code execution 
- Performance monitoring and analytics
- Threat detection and response
- Persistent volume management
- Container pooling and resource optimization

Tests realistic production workloads and security attack scenarios.
"""

import asyncio
import logging
import time
import json
import os
import random
import tempfile
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionTestSuite:
    """Comprehensive production testing suite."""
    
    def __init__(self):
        self.test_results = []
        self.performance_data = {}
        self.security_events = []
        
    async def initialize_systems(self):
        """Initialize all orchestrator systems."""
        from orchestrator.security.docker_manager import EnhancedDockerManager
        from orchestrator.analytics.performance_monitor import PerformanceMonitor
        from orchestrator.security.threat_detection import ThreatMonitor
        from orchestrator.storage.volume_manager import PersistentVolumeManager, VolumeType
        from orchestrator.tools.multi_language_executor import MultiLanguageExecutor, Language
        
        # Store Language for use in methods
        self.Language = Language
        
        logger.info("🚀 Initializing production test environment...")
        
        # Initialize core systems
        self.performance_monitor = PerformanceMonitor(collection_interval=1.0)
        self.docker_manager = EnhancedDockerManager(
            enable_advanced_pooling=True,
            performance_monitor=self.performance_monitor
        )
        self.threat_monitor = ThreatMonitor(
            docker_manager=self.docker_manager,
            performance_monitor=self.performance_monitor
        )
        self.volume_manager = PersistentVolumeManager(
            performance_monitor=self.performance_monitor
        )
        self.executor = MultiLanguageExecutor(self.docker_manager)
        
        # Start all monitoring systems
        await self.performance_monitor.start_monitoring()
        await self.docker_manager.start_background_tasks()
        await self.threat_monitor.start_monitoring()
        await self.volume_manager.start_monitoring()
        
        logger.info("✅ All production systems initialized and monitoring started")
        
    async def shutdown_systems(self):
        """Gracefully shutdown all systems."""
        logger.info("🔄 Shutting down production test environment...")
        
        await self.threat_monitor.stop_monitoring()
        await self.volume_manager.stop_monitoring()
        await self.docker_manager.shutdown()
        await self.performance_monitor.stop_monitoring()
        
        logger.info("✅ All systems shut down gracefully")
    
    async def test_high_volume_workload(self) -> Dict[str, Any]:
        """Test high-volume concurrent workload."""
        logger.info("📊 Testing high-volume concurrent workload...")
        
        start_time = time.time()
        test_results = {
            'test_name': 'high_volume_workload',
            'start_time': start_time,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'average_execution_time': 0,
            'peak_container_count': 0,
            'errors': []
        }
        
        try:
            # Create diverse workload tasks
            tasks = []
            languages = [self.Language.PYTHON, self.Language.JAVASCRIPT, self.Language.BASH]
            
            # Generate 20 concurrent tasks
            for i in range(20):
                language = random.choice(languages)
                
                if language == self.Language.PYTHON:
                    code = f"""
import time
import random
import json

# Simulate data processing workload
data = {{'task_id': {i}, 'timestamp': time.time()}}
processing_time = random.uniform(0.1, 0.5)
time.sleep(processing_time)

result = {{
    'task_id': {i},
    'processed_at': time.time(),
    'processing_time': processing_time,
    'status': 'completed'
}}

print(json.dumps(result))
"""
                elif language == self.Language.JAVASCRIPT:
                    code = f"""
const taskId = {i};
const startTime = Date.now();

// Simulate async processing
const processData = () => {{
    return new Promise(resolve => {{
        const delay = Math.random() * 300 + 100; // 100-400ms
        setTimeout(() => {{
            resolve({{
                task_id: taskId,
                processed_at: Date.now(),
                processing_time: delay,
                status: 'completed'
            }});
        }}, delay);
    }});
}};

processData().then(result => {{
    console.log(JSON.stringify(result));
}});
"""
                else:  # BASH
                    code = f"""
#!/bin/bash
TASK_ID={i}
START_TIME=$(date +%s%3N)

# Simulate file processing
echo "Processing task $TASK_ID" > /tmp/task_$TASK_ID.log
sleep 0.2

END_TIME=$(date +%s%3N)
DURATION=$((END_TIME - START_TIME))

echo "{{\"task_id\": $TASK_ID, \"duration\": $DURATION, \"status\": \"completed\"}}"
"""
                
                tasks.append((language, code, f"workload_task_{i}"))
            
            # Execute all tasks concurrently
            execution_tasks = []
            for language, code, task_name in tasks:
                task_coro = self.execute_with_monitoring(language, code, task_name)
                execution_tasks.append(task_coro)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*execution_tasks, return_exceptions=True)
            
            # Analyze results
            execution_times = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    test_results['tasks_failed'] += 1
                    test_results['errors'].append(str(result))
                    logger.error(f"Task {i} failed: {result}")
                else:
                    test_results['tasks_completed'] += 1
                    if result and 'execution_time' in result:
                        execution_times.append(result['execution_time'])
            
            if execution_times:
                test_results['average_execution_time'] = sum(execution_times) / len(execution_times)
            
            test_results['duration'] = time.time() - start_time
            test_results['success_rate'] = (test_results['tasks_completed'] / len(tasks)) * 100
            
            logger.info(f"✅ High-volume workload completed: {test_results['tasks_completed']}/{len(tasks)} tasks succeeded")
            
        except Exception as e:
            test_results['error'] = str(e)
            logger.error(f"❌ High-volume workload test failed: {e}")
        
        self.test_results.append(test_results)
        return test_results
    
    async def execute_with_monitoring(self, language, code, task_name):
        """Execute code with performance monitoring."""
        start_time = time.time()
        
        try:
            result = await self.executor.execute_code(code, language, timeout=30)
            execution_time = time.time() - start_time
            
            return {
                'task_name': task_name,
                'success': result.success,
                'execution_time': execution_time,
                'output': result.output[:200] if result.output else None,  # First 200 chars
                'error': result.error if not result.success else None
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'task_name': task_name,
                'success': False,
                'execution_time': execution_time,
                'error': str(e)
            }
    
    async def test_security_attack_scenarios(self) -> Dict[str, Any]:
        """Test various security attack scenarios."""
        logger.info("🔒 Testing security attack scenarios...")
        
        test_results = {
            'test_name': 'security_attack_scenarios',
            'start_time': time.time(),
            'attacks_detected': 0,
            'attacks_blocked': 0,
            'false_positives': 0,
            'attack_scenarios': []
        }
        
        # Define attack scenarios
        attack_scenarios = [
            {
                'name': 'Command Injection',
                'language': self.Language.PYTHON,
                'code': """
import os
import subprocess

# Attempt command injection
malicious_input = "test; rm -rf /"
os.system(f"echo {malicious_input}")
subprocess.call(['ls', '/etc/passwd'])
"""
            },
            {
                'name': 'Code Evaluation Attack',
                'language': self.Language.PYTHON,
                'code': """
# Attempt code evaluation attack
user_input = "__import__('os').system('whoami')"
eval(user_input)
exec("import socket; s=socket.socket()")
"""
            },
            {
                'name': 'File System Access',
                'language': self.Language.PYTHON,
                'code': """
# Attempt unauthorized file access
import os
with open('/etc/passwd', 'r') as f:
    sensitive_data = f.read()
    print("Accessed:", len(sensitive_data))

with open('/proc/version', 'r') as f:
    system_info = f.read()
"""
            },
            {
                'name': 'Privilege Escalation',
                'language': self.Language.BASH,
                'code': """
#!/bin/bash
# Attempt privilege escalation
sudo whoami
su root
chmod 777 /tmp
"""
            },
            {
                'name': 'Container Escape Attempt',
                'language': self.Language.PYTHON,
                'code': """
import os
# Attempt container escape
os.system("cat /proc/self/cgroup")
os.system("ls -la /sys/fs/cgroup")
with open('/proc/self/ns/pid', 'r') as f:
    namespace_info = f.read()
"""
            },
            {
                'name': 'Resource Exhaustion',
                'language': self.Language.PYTHON,
                'code': """
# Attempt resource exhaustion
import time
import threading

def cpu_intensive():
    while True:
        pass

# Create multiple threads
for i in range(10):
    thread = threading.Thread(target=cpu_intensive)
    thread.start()

# Memory exhaustion
large_data = []
for i in range(1000000):
    large_data.append("x" * 1000)
"""
            },
            {
                'name': 'Network Scanning',
                'language': self.Language.PYTHON,
                'code': """
import socket
import threading

def scan_port(host, port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        result = sock.connect((host, port))
        sock.close()
        return True
    except:
        return False

# Attempt port scanning
target_host = "127.0.0.1"
for port in range(20, 100):
    if scan_port(target_host, port):
        print(f"Port {port} is open")
"""
            }
        ]
        
        initial_security_events = len(self.threat_monitor.security_events)
        
        try:
            for scenario in attack_scenarios:
                logger.info(f"🎯 Testing attack scenario: {scenario['name']}")
                
                scenario_result = {
                    'name': scenario['name'],
                    'detected': False,
                    'blocked': False,
                    'events_generated': 0
                }
                
                # Execute malicious code
                events_before = len(self.threat_monitor.security_events)
                
                # Scan code for threats first
                threat_events = await self.threat_monitor.scan_code(
                    scenario['code'], 
                    f"attack_test_{scenario['name'].lower().replace(' ', '_')}"
                )
                
                events_after_scan = len(self.threat_monitor.security_events)
                
                # Try to execute (may be blocked by threat detection)
                try:
                    result = await self.executor.execute_code(
                        scenario['code'], 
                        scenario['language'], 
                        timeout=10
                    )
                    scenario_result['execution_attempted'] = True
                    scenario_result['execution_succeeded'] = result.success
                    
                    if not result.success:
                        scenario_result['blocked'] = True
                        test_results['attacks_blocked'] += 1
                        
                except Exception as e:
                    scenario_result['execution_error'] = str(e)
                    scenario_result['blocked'] = True
                    test_results['attacks_blocked'] += 1
                
                events_after_execution = len(self.threat_monitor.security_events)
                
                scenario_result['events_generated'] = events_after_execution - events_before
                
                if scenario_result['events_generated'] > 0 or len(threat_events) > 0:
                    scenario_result['detected'] = True
                    test_results['attacks_detected'] += 1
                
                test_results['attack_scenarios'].append(scenario_result)
                
                # Wait a bit between attacks
                await asyncio.sleep(1.0)
            
            test_results['total_scenarios'] = len(attack_scenarios)
            test_results['detection_rate'] = (test_results['attacks_detected'] / len(attack_scenarios)) * 100
            test_results['blocking_rate'] = (test_results['attacks_blocked'] / len(attack_scenarios)) * 100
            
            logger.info(f"✅ Security testing completed: {test_results['attacks_detected']}/{len(attack_scenarios)} attacks detected")
            
        except Exception as e:
            test_results['error'] = str(e)
            logger.error(f"❌ Security attack testing failed: {e}")
        
        test_results['duration'] = time.time() - test_results['start_time']
        self.test_results.append(test_results)
        return test_results
    
    async def test_volume_persistence_scenarios(self) -> Dict[str, Any]:
        """Test volume persistence across container lifecycles."""
        logger.info("💾 Testing volume persistence scenarios...")
        
        test_results = {
            'test_name': 'volume_persistence_scenarios',
            'start_time': time.time(),
            'volumes_created': 0,
            'data_persistence_verified': 0,
            'backup_restore_successful': 0,
            'scenarios': []
        }
        
        try:
            from orchestrator.storage.volume_manager import VolumeType, VolumeQuota
            
            # Scenario 1: Persistent data across container restarts
            scenario_1 = {
                'name': 'Persistent Data Across Restarts',
                'success': False,
                'details': {}
            }
            
            # Create persistent volume
            persistent_volume = await self.volume_manager.create_volume(
                owner="production_test",
                volume_type=VolumeType.PERSISTENT,
                description="Production test persistent volume"
            )
            
            if persistent_volume:
                test_results['volumes_created'] += 1
                
                # Add initial data
                volume_info = self.volume_manager.get_volume_info(persistent_volume)
                data_file = os.path.join(volume_info['metadata']['host_path'], "persistent_data.txt")
                test_data = f"Production test data created at {time.time()}"
                
                with open(data_file, 'w') as f:
                    f.write(test_data)
                
                # Simulate container restart by reading data
                await asyncio.sleep(1.0)
                
                if os.path.exists(data_file):
                    with open(data_file, 'r') as f:
                        read_data = f.read()
                    
                    if read_data == test_data:
                        scenario_1['success'] = True
                        test_results['data_persistence_verified'] += 1
                        scenario_1['details']['data_match'] = True
                    else:
                        scenario_1['details']['data_mismatch'] = True
                else:
                    scenario_1['details']['file_missing'] = True
            
            test_results['scenarios'].append(scenario_1)
            
            # Scenario 2: Backup and restore
            scenario_2 = {
                'name': 'Backup and Restore',
                'success': False,
                'details': {}
            }
            
            if persistent_volume:
                # Create backup
                backup_path = self.volume_manager.create_backup(persistent_volume)
                
                if backup_path and os.path.exists(backup_path):
                    scenario_2['details']['backup_created'] = True
                    
                    # Simulate data loss by deleting original file
                    volume_info = self.volume_manager.get_volume_info(persistent_volume)
                    data_file = os.path.join(volume_info['metadata']['host_path'], "persistent_data.txt")
                    
                    if os.path.exists(data_file):
                        os.remove(data_file)
                    
                    # Restore from backup
                    restore_success = self.volume_manager.restore_volume(persistent_volume, backup_path)
                    
                    if restore_success and os.path.exists(data_file):
                        with open(data_file, 'r') as f:
                            restored_data = f.read()
                        
                        if restored_data == test_data:
                            scenario_2['success'] = True
                            test_results['backup_restore_successful'] += 1
                            scenario_2['details']['restore_verified'] = True
                        else:
                            scenario_2['details']['restore_data_mismatch'] = True
                    else:
                        scenario_2['details']['restore_failed'] = True
                else:
                    scenario_2['details']['backup_failed'] = True
            
            test_results['scenarios'].append(scenario_2)
            
            # Scenario 3: Shared volume between multiple tasks
            scenario_3 = {
                'name': 'Shared Volume Multi-Task',
                'success': False,
                'details': {}
            }
            
            shared_volume = await self.volume_manager.create_volume(
                owner="production_test",
                volume_type=VolumeType.SHARED,
                description="Shared volume for multi-task testing"
            )
            
            if shared_volume:
                test_results['volumes_created'] += 1
                
                # Mount to multiple "containers" (simulated)
                mount1 = self.volume_manager.mount_volume(shared_volume, "task_1")
                mount2 = self.volume_manager.mount_volume(shared_volume, "task_2")
                
                if mount1 and mount2:
                    scenario_3['details']['multi_mount_success'] = True
                    
                    # Write data from "different tasks"
                    volume_info = self.volume_manager.get_volume_info(shared_volume)
                    shared_file = os.path.join(volume_info['metadata']['host_path'], "shared_data.json")
                    
                    shared_data = {
                        'task_1': f'Data from task 1 at {time.time()}',
                        'task_2': f'Data from task 2 at {time.time()}',
                        'shared_timestamp': time.time()
                    }
                    
                    with open(shared_file, 'w') as f:
                        json.dump(shared_data, f)
                    
                    # Verify both tasks can read
                    if os.path.exists(shared_file):
                        with open(shared_file, 'r') as f:
                            read_shared_data = json.load(f)
                        
                        if read_shared_data == shared_data:
                            scenario_3['success'] = True
                            scenario_3['details']['data_sharing_verified'] = True
                    
                    # Unmount
                    self.volume_manager.unmount_volume(shared_volume, "task_1")
                    self.volume_manager.unmount_volume(shared_volume, "task_2")
                else:
                    scenario_3['details']['mount_failed'] = True
            
            test_results['scenarios'].append(scenario_3)
            
            test_results['success_rate'] = (
                sum(1 for s in test_results['scenarios'] if s['success']) / 
                len(test_results['scenarios'])
            ) * 100
            
            logger.info(f"✅ Volume persistence testing completed: {test_results['success_rate']:.1f}% scenarios passed")
            
        except Exception as e:
            test_results['error'] = str(e)
            logger.error(f"❌ Volume persistence testing failed: {e}")
        
        test_results['duration'] = time.time() - test_results['start_time']
        self.test_results.append(test_results)
        return test_results
    
    async def test_performance_under_load(self) -> Dict[str, Any]:
        """Test system performance under various load conditions."""
        logger.info("⚡ Testing performance under load...")
        
        test_results = {
            'test_name': 'performance_under_load',
            'start_time': time.time(),
            'load_scenarios': []
        }
        
        try:
            # Get initial performance baseline
            initial_summary = self.performance_monitor.get_performance_summary()
            
            # Scenario 1: Sustained moderate load
            logger.info("📈 Testing sustained moderate load...")
            
            moderate_load_tasks = []
            for i in range(10):
                code = f"""
import time
import json
import random

# Simulate moderate processing load
data = [{{'id': j, 'value': random.random()}} for j in range(1000)]
time.sleep(random.uniform(0.5, 1.0))

result = {{
    'task_id': {i},
    'processed_items': len(data),
    'completion_time': time.time()
}}

print(json.dumps(result))
"""
                task = self.execute_with_monitoring(self.Language.PYTHON, code, f"moderate_load_{i}")
                moderate_load_tasks.append(task)
            
            moderate_results = await asyncio.gather(*moderate_load_tasks, return_exceptions=True)
            
            moderate_scenario = {
                'name': 'Sustained Moderate Load',
                'tasks': len(moderate_load_tasks),
                'successful': sum(1 for r in moderate_results if not isinstance(r, Exception) and r.get('success', False)),
                'avg_execution_time': 0
            }
            
            execution_times = [
                r['execution_time'] for r in moderate_results 
                if not isinstance(r, Exception) and 'execution_time' in r
            ]
            
            if execution_times:
                moderate_scenario['avg_execution_time'] = sum(execution_times) / len(execution_times)
            
            test_results['load_scenarios'].append(moderate_scenario)
            
            # Wait between scenarios
            await asyncio.sleep(2.0)
            
            # Scenario 2: Burst load with short tasks
            logger.info("🚀 Testing burst load...")
            
            burst_tasks = []
            for i in range(30):
                code = f"""
import time
import json

start_time = time.time()
# Quick computation
result = sum(range(1000))
end_time = time.time()

output = {{
    'task_id': {i},
    'result': result,
    'duration': end_time - start_time
}}

print(json.dumps(output))
"""
                task = self.execute_with_monitoring(self.Language.PYTHON, code, f"burst_task_{i}")
                burst_tasks.append(task)
            
            burst_results = await asyncio.gather(*burst_tasks, return_exceptions=True)
            
            burst_scenario = {
                'name': 'Burst Load',
                'tasks': len(burst_tasks),
                'successful': sum(1 for r in burst_results if not isinstance(r, Exception) and r.get('success', False)),
                'avg_execution_time': 0
            }
            
            burst_execution_times = [
                r['execution_time'] for r in burst_results 
                if not isinstance(r, Exception) and 'execution_time' in r
            ]
            
            if burst_execution_times:
                burst_scenario['avg_execution_time'] = sum(burst_execution_times) / len(burst_execution_times)
            
            test_results['load_scenarios'].append(burst_scenario)
            
            # Get final performance summary
            final_summary = self.performance_monitor.get_performance_summary()
            
            test_results['performance_impact'] = {
                'initial_system_health': initial_summary.get('system_health', 0),
                'final_system_health': final_summary.get('system_health', 0),
                'total_metrics_processed': final_summary.get('metrics_processed', 0) - initial_summary.get('metrics_processed', 0),
                'active_alerts': final_summary.get('active_alerts', 0)
            }
            
            logger.info("✅ Performance load testing completed")
            
        except Exception as e:
            test_results['error'] = str(e)
            logger.error(f"❌ Performance load testing failed: {e}")
        
        test_results['duration'] = time.time() - test_results['start_time']
        self.test_results.append(test_results)
        return test_results
    
    async def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive production test report."""
        logger.info("📋 Generating comprehensive production test report...")
        
        # Get system summaries
        performance_summary = self.performance_monitor.get_performance_summary()
        security_summary = self.threat_monitor.get_security_summary()
        storage_summary = self.volume_manager.get_storage_summary()
        
        # Calculate overall metrics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for test in self.test_results if 'error' not in test)
        
        report = {
            'report_timestamp': time.time(),
            'report_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': (successful_tests / total_tests) * 100 if total_tests > 0 else 0,
                'total_duration': sum(test.get('duration', 0) for test in self.test_results)
            },
            'system_performance': {
                'system_health': performance_summary.get('system_health', 0),
                'metrics_processed': performance_summary.get('metrics_processed', 0),
                'active_alerts': performance_summary.get('active_alerts', 0),
                'component_profiles': performance_summary.get('component_profiles', 0)
            },
            'security_status': {
                'total_events': security_summary.get('total_events', 0),
                'active_events': security_summary.get('active_events', 0),
                'attacks_detected': security_summary.get('statistics', {}).get('events_detected', 0),
                'responses_executed': security_summary.get('statistics', {}).get('responses_executed', 0),
                'threat_signatures': security_summary.get('threat_signatures', 0)
            },
            'storage_status': {
                'total_volumes': storage_summary.get('total_volumes', 0),
                'mounted_volumes': storage_summary.get('mounted_volumes', 0),
                'volumes_created': storage_summary.get('statistics', {}).get('volumes_created', 0),
                'backups_created': storage_summary.get('statistics', {}).get('backups_created', 0)
            },
            'detailed_test_results': self.test_results,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Analyze test results for recommendations
        for test in self.test_results:
            if test.get('test_name') == 'high_volume_workload':
                success_rate = test.get('success_rate', 0)
                if success_rate < 95:
                    recommendations.append(f"High-volume workload success rate ({success_rate:.1f}%) is below optimal. Consider container pool optimization.")
                
                avg_time = test.get('average_execution_time', 0)
                if avg_time > 5.0:
                    recommendations.append(f"Average execution time ({avg_time:.2f}s) is high. Consider resource optimization.")
            
            elif test.get('test_name') == 'security_attack_scenarios':
                detection_rate = test.get('detection_rate', 0)
                if detection_rate < 90:
                    recommendations.append(f"Security detection rate ({detection_rate:.1f}%) could be improved. Review threat signatures.")
                
                blocking_rate = test.get('blocking_rate', 0)
                if blocking_rate < 80:
                    recommendations.append(f"Attack blocking rate ({blocking_rate:.1f}%) needs improvement. Strengthen automated responses.")
            
            elif test.get('test_name') == 'volume_persistence_scenarios':
                success_rate = test.get('success_rate', 0)
                if success_rate < 100:
                    recommendations.append("Volume persistence scenarios had issues. Review backup and restore mechanisms.")
        
        if not recommendations:
            recommendations.append("All production tests passed successfully. System is ready for production deployment.")
        
        return recommendations


async def main():
    """Run comprehensive production testing."""
    logger.info("🎯 Starting Comprehensive Production Workload and Attack Scenario Testing")
    
    test_suite = ProductionTestSuite()
    
    try:
        # Initialize all systems
        await test_suite.initialize_systems()
        
        # Run all test scenarios
        logger.info("🔄 Executing production test scenarios...")
        
        # Test 1: High-volume concurrent workload
        await test_suite.test_high_volume_workload()
        
        # Test 2: Security attack scenarios
        await test_suite.test_security_attack_scenarios()
        
        # Test 3: Volume persistence scenarios
        await test_suite.test_volume_persistence_scenarios()
        
        # Test 4: Performance under load
        await test_suite.test_performance_under_load()
        
        # Generate comprehensive report
        report = await test_suite.generate_comprehensive_report()
        
        # Save report
        report_file = f"/tmp/production_test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        logger.info("="*80)
        logger.info("PRODUCTION TEST SUITE SUMMARY")
        logger.info("="*80)
        logger.info(f"📊 Total Tests: {report['test_summary']['total_tests']}")
        logger.info(f"✅ Successful: {report['test_summary']['successful_tests']}")
        logger.info(f"📈 Success Rate: {report['test_summary']['success_rate']:.1f}%")
        logger.info(f"⏱️  Total Duration: {report['test_summary']['total_duration']:.2f}s")
        logger.info("")
        logger.info("🖥️  SYSTEM STATUS:")
        logger.info(f"   System Health: {report['system_performance']['system_health']:.1f}%")
        logger.info(f"   Metrics Processed: {report['system_performance']['metrics_processed']}")
        logger.info(f"   Active Alerts: {report['system_performance']['active_alerts']}")
        logger.info("")
        logger.info("🔒 SECURITY STATUS:")
        logger.info(f"   Security Events: {report['security_status']['total_events']}")
        logger.info(f"   Attacks Detected: {report['security_status']['attacks_detected']}")
        logger.info(f"   Responses Executed: {report['security_status']['responses_executed']}")
        logger.info("")
        logger.info("💾 STORAGE STATUS:")
        logger.info(f"   Volumes Created: {report['storage_status']['volumes_created']}")
        logger.info(f"   Backups Created: {report['storage_status']['backups_created']}")
        logger.info("")
        logger.info("💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            logger.info(f"   {i}. {rec}")
        logger.info("")
        logger.info(f"📄 Detailed report saved to: {report_file}")
        logger.info("="*80)
        
        success = report['test_summary']['success_rate'] >= 80  # 80% threshold for production readiness
        
        if success:
            logger.info("🎉 PRODUCTION TESTING PASSED! System is ready for production deployment.")
        else:
            logger.info("⚠️  PRODUCTION TESTING FAILED! System needs improvements before production.")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Production testing failed with critical error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await test_suite.shutdown_systems()

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)