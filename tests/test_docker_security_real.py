"""Real Docker Security and Isolation Validation Tests - Issue #206 Phase 1

Comprehensive security testing using REAL attack scenarios, REAL containers,
and REAL security violations. NO MOCKS - only actual security testing.
"""

import pytest
import asyncio
import docker
import logging
import time
import tempfile
import os
import signal
import subprocess
from pathlib import Path

# Import our security components
from orchestrator.security.docker_manager import (
    EnhancedDockerManager, 
    ResourceLimits, 
    SecurityConfig,
    create_strict_security_config
)
from orchestrator.security.policy_engine import (
    SecurityPolicyEngine,
    ThreatLevel,
    SandboxingLevel
)
from orchestrator.security.dependency_manager import (
    IntelligentDependencyManager,
    PackageInfo,
    PackageEcosystem
)
from orchestrator.security.resource_monitor import (
    ResourceMonitor,
    AlertSeverity
)

# Configure logging for test visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRealDockerSecurityIsolation:
    """Test Docker container security isolation with real attack scenarios."""
    
    @pytest.fixture(scope="session")
    def docker_client(self):
        """Get Docker client for testing."""
        client = docker.from_env()
        yield client
        client.close()
    
    @pytest.fixture
    async def docker_manager(self, docker_client):
        """Get Docker manager for testing."""
        manager = EnhancedDockerManager(enable_container_pooling=False)  # Disable pooling for tests
        await manager.start_background_tasks()
        yield manager
        await manager.cleanup_all_containers(force=True)
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_container_escape_prevention_real(self, docker_manager):
        """Test real container escape attempts are blocked."""
        logger.info("🧪 Testing real container escape prevention")
        
        # Create container with strict security
        security_config = create_strict_security_config()
        resource_limits = ResourceLimits(memory_mb=128, cpu_cores=0.25, execution_timeout=15)
        
        container = await docker_manager.create_secure_container(
            image="python:3.11-slim",
            name="escape_test_container",
            resource_limits=resource_limits,
            security_config=security_config
        )
        
        # Test 1: Try to access host filesystem
        result = await docker_manager.execute_in_container(
            container,
            "/bin/sh -c \"ls /host && echo 'HOST_COMPROMISED'\"",
            timeout=10
        )
        assert not result['success'], "Host filesystem should not be accessible"
        assert 'HOST_COMPROMISED' not in result.get('output', ''), "Host access should be blocked"
        
        # Test 2: Try to mount host filesystems
        result = await docker_manager.execute_in_container(
            container,
            "/bin/sh -c \"mount -t proc proc /proc && echo 'PROC_MOUNTED'\"",
            timeout=10
        )
        assert not result['success'], "Mounting should be blocked"
        assert 'PROC_MOUNTED' not in result.get('output', ''), "Proc mounting should fail"
        
        # Test 3: Try to create privileged processes
        result = await docker_manager.execute_in_container(
            container,
            "/bin/sh -c \"su - root && echo 'ROOT_ACCESS'\"",
            timeout=10
        )
        assert not result['success'], "Root access should be blocked"
        assert 'ROOT_ACCESS' not in result.get('output', ''), "Root access should fail"
        
        # Test 4: Try to access Docker socket
        result = await docker_manager.execute_in_container(
            container,
            "/bin/sh -c \"ls /var/run/docker.sock && echo 'DOCKER_SOCKET_FOUND'\"",
            timeout=10
        )
        assert 'DOCKER_SOCKET_FOUND' not in result.get('output', ''), "Docker socket should not be accessible"
        
        # Test 5: Verify that /proc escape just shows container filesystem, not host
        result = await docker_manager.execute_in_container(
            container,
            "/bin/sh -c \"ls /proc/self/root/../../../usr/bin/docker 2>/dev/null && echo 'HOST_DOCKER_ACCESSIBLE' || echo 'HOST_DOCKER_BLOCKED'\"",
            timeout=10
        )
        # The docker binary should not be accessible inside the container  
        assert 'HOST_DOCKER_ACCESSIBLE' not in result.get('output', ''), "Host Docker binary should not be accessible"
        
        await docker_manager.destroy_container(container, force=True)
        logger.info("✅ Container escape prevention tests passed")
    
    @pytest.mark.asyncio
    async def test_resource_limit_enforcement_real(self, docker_manager):
        """Test real resource limit enforcement with actual attacks."""
        logger.info("🧪 Testing real resource limit enforcement")
        
        # Create container with tight resource limits
        resource_limits = ResourceLimits(
            memory_mb=64,  # Very low memory limit
            cpu_cores=0.1,  # Very low CPU limit
            execution_timeout=10,
            pids_limit=20
        )
        
        container = await docker_manager.create_secure_container(
            image="python:3.11-slim",
            name="resource_test_container",
            resource_limits=resource_limits,
            security_config=SecurityConfig()
        )
        
        # Test 1: Memory bomb attack
        memory_bomb_code = """
import sys
data = []
try:
    # Try to allocate 1GB of memory (should fail with 64MB limit)
    for i in range(1000):
        data.append('A' * 1024 * 1024)  # 1MB chunks
        if i % 10 == 0:
            print(f'Allocated {i}MB')
    print('MEMORY_BOMB_SUCCEEDED')
except MemoryError:
    print('MEMORY_LIMIT_ENFORCED')
except Exception as e:
    print(f'MEMORY_ERROR: {e}')
"""
        
        result = await docker_manager.execute_in_container(
            container, 
            f'python3 -c "{memory_bomb_code}"',
            timeout=15
        )
        
        # Should fail due to memory limit
        output = result.get('output', '') + result.get('error', '')
        exit_code = result.get('exit_code', 0)
        
        assert 'MEMORY_BOMB_SUCCEEDED' not in output, "Memory bomb should be blocked"
        
        # Check for memory limit enforcement (multiple ways this can manifest)
        memory_limited = (
            'MEMORY_LIMIT_ENFORCED' in output or 
            'MemoryError' in output or 
            'killed' in output.lower() or
            exit_code == 137 or  # SIGKILL from Docker memory limit
            exit_code == 9 or    # SIGKILL 
            not result.get('success', True)  # Execution failed
        )
        assert memory_limited, f"Memory limit should be enforced. Got exit_code={exit_code}, output='{output}'"
        
        # Test 2: CPU exhaustion attack
        cpu_bomb_code = """
import time
import multiprocessing
start_time = time.time()
try:
    # Try to spawn many CPU-intensive processes
    processes = []
    for i in range(50):  # Try to spawn 50 processes (should fail with pids_limit=20)
        p = multiprocessing.Process(target=lambda: [x*x for x in range(10000000)])
        p.start()
        processes.append(p)
        print(f'Started process {i}')
    
    print('CPU_BOMB_SUCCEEDED')
    
    for p in processes:
        p.join()
except Exception as e:
    print(f'CPU_LIMIT_ENFORCED: {e}')
finally:
    print(f'Execution time: {time.time() - start_time:.2f}s')
"""
        
        result = await docker_manager.execute_in_container(
            container,
            f'python3 -c "{cpu_bomb_code}"',
            timeout=20
        )
        
        output = result.get('output', '') + result.get('error', '')
        # Should be limited by PID limit or CPU constraints
        assert 'CPU_BOMB_SUCCEEDED' not in output, "CPU bomb should be limited"
        
        # Test 3: Fork bomb attack
        fork_bomb_code = """
import os
import sys
fork_count = 0
try:
    # Try to fork unlimited processes
    for i in range(100):
        pid = os.fork()
        if pid == 0:
            # Child process
            print(f'Child process {i}')
            sys.exit(0)
        else:
            fork_count += 1
            print(f'Forked {fork_count} processes')
    print('FORK_BOMB_SUCCEEDED')
except OSError as e:
    print(f'FORK_LIMIT_ENFORCED: {e}')
except Exception as e:
    print(f'FORK_ERROR: {e}')
"""
        
        result = await docker_manager.execute_in_container(
            container,
            f'python3 -c "{fork_bomb_code}"',
            timeout=15
        )
        
        output = result.get('output', '') + result.get('error', '')
        # Should be limited by PID limit
        assert 'FORK_BOMB_SUCCEEDED' not in output, "Fork bomb should be limited"
        assert ('FORK_LIMIT_ENFORCED' in output or 
                'Resource temporarily unavailable' in output), "Fork limit should be enforced"
        
        await docker_manager.destroy_container(container, force=True)
        logger.info("✅ Resource limit enforcement tests passed")
    
    @pytest.mark.asyncio
    async def test_malicious_code_execution_real(self, docker_manager):
        """Test execution of real malicious code patterns."""
        logger.info("🧪 Testing malicious code pattern detection and blocking")
        
        # Initialize security policy engine
        policy_engine = SecurityPolicyEngine()
        
        # Test malicious code samples
        malicious_codes = [
            # Code injection attempt
            {
                'name': 'code_injection',
                'code': '''
import os
user_input = "'; os.system('rm -rf /'); print('"
eval(f"print('{user_input}')")
print('CODE_INJECTION_EXECUTED')
                ''',
                'should_block': True
            },
            
            # Command injection
            {
                'name': 'command_injection', 
                'code': '''
import subprocess
user_command = "ls; cat /etc/passwd; echo 'SYSTEM_COMPROMISED'"
subprocess.run(user_command, shell=True, capture_output=True, text=True)
print('COMMAND_INJECTION_EXECUTED')
                ''',
                'should_block': True
            },
            
            # Network backdoor attempt
            {
                'name': 'network_backdoor',
                'code': '''
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("0.0.0.0", 4444))
    s.listen(1)
    print("BACKDOOR_LISTENING_ON_4444")
    import time
    time.sleep(0.1)  # Short sleep instead of blocking accept
    print("BACKDOOR_SETUP_COMPLETE")
except Exception as e:
    print(f"NETWORK_BLOCKED: {e}")
                ''',
                'should_block': True
            },
            
            # File system attack
            {
                'name': 'filesystem_attack',
                'code': '''
import os
try:
    # Try to access sensitive files
    with open('/etc/passwd', 'r') as f:
        content = f.read()
        print('PASSWD_FILE_ACCESSED')
        print(content[:100])
    
    # Try to write to system directories
    with open('/etc/malicious_file', 'w') as f:
        f.write('MALICIOUS_CONTENT')
    print('SYSTEM_FILE_WRITTEN')
    
except PermissionError as e:
    print(f'FILESYSTEM_ACCESS_BLOCKED: {e}')
except Exception as e:
    print(f'FILESYSTEM_ERROR: {e}')
                ''',
                'should_block': True
            }
        ]
        
        # Test each malicious code sample
        for test_case in malicious_codes:
            logger.info(f"Testing {test_case['name']}")
            
            # First, analyze with security policy engine
            assessment = await policy_engine.evaluate_execution_request(
                test_case['code'], language='python'
            )
            
            if test_case['should_block']:
                # Check that violations are detected (most important)
                assert len(assessment.violations) > 0, \
                    f"{test_case['name']} should have security violations"
                
                # Check that at least one violation is high or critical severity
                high_severity_violations = [v for v in assessment.violations 
                                          if v.severity in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]]
                assert len(high_severity_violations) > 0, \
                    f"{test_case['name']} should have at least one high/critical severity violation"
                
                # If marked as BLOCKED, don't execute
                if assessment.sandboxing_level == SandboxingLevel.BLOCKED:
                    logger.info(f"✅ {test_case['name']} correctly blocked by security policy")
                    continue
            
            # Execute in container with strict security
            container = await docker_manager.create_secure_container(
                image="python:3.11-slim",
                name=f"malicious_test_{test_case['name']}",
                resource_limits=ResourceLimits(memory_mb=128, execution_timeout=10),
                security_config=create_strict_security_config()
            )
            
            result = await docker_manager.execute_in_container(
                container,
                f'python3 -c "{test_case["code"]}"',
                timeout=15
            )
            
            output = result.get('output', '') + result.get('error', '')
            
            # Verify malicious operations were blocked
            if test_case['name'] == 'code_injection':
                assert 'CODE_INJECTION_EXECUTED' not in output, "Code injection should be blocked"
            elif test_case['name'] == 'command_injection':
                assert 'COMMAND_INJECTION_EXECUTED' not in output, "Command injection should be blocked"
                assert 'SYSTEM_COMPROMISED' not in output, "System should not be compromised"
            elif test_case['name'] == 'network_backdoor':
                assert 'BACKDOOR_LISTENING_ON_4444' not in output, "Backdoor should not be established"
                # Network operations may be blocked or allowed but contained
                network_contained = (
                    'NETWORK_BLOCKED' in output or 
                    'BACKDOOR_SETUP_COMPLETE' not in output or
                    not result.get('success', True)
                )
                assert network_contained, f"Network operations should be contained or blocked. Got: {output}"
            elif test_case['name'] == 'filesystem_attack':
                # Reading container's passwd file is expected behavior, but writing should be blocked
                assert 'SYSTEM_FILE_WRITTEN' not in output, "System files should not be writable"
                assert 'FILESYSTEM_ERROR' in output, "Filesystem write operations should be blocked"
                assert 'Read-only file system' in output, "Should encounter read-only filesystem error"
            
            await docker_manager.destroy_container(container, force=True)
            logger.info(f"✅ {test_case['name']} properly contained")
        
        logger.info("✅ Malicious code execution tests passed")
    
    @pytest.mark.asyncio
    async def test_dependency_vulnerability_scanning_real(self, docker_manager):
        """Test real dependency vulnerability scanning with actual malicious packages."""
        logger.info("🧪 Testing real dependency vulnerability scanning")
        
        # Initialize dependency manager
        dep_manager = IntelligentDependencyManager(enable_validation=True)
        
        # Test malicious and vulnerable packages
        test_packages = [
            # Known malicious packages (simulated)
            PackageInfo(name="requests-malicious", ecosystem=PackageEcosystem.PYPI),
            PackageInfo(name="urllib3-fake", ecosystem=PackageEcosystem.PYPI),
            PackageInfo(name="numpy-evil", ecosystem=PackageEcosystem.PYPI),
            
            # Typosquatting attempts
            PackageInfo(name="requsts", ecosystem=PackageEcosystem.PYPI),  # requests typo
            PackageInfo(name="nummpy", ecosystem=PackageEcosystem.PYPI),   # numpy typo
            PackageInfo(name="pandass", ecosystem=PackageEcosystem.PYPI),  # pandas typo
        ]
        
        container = await docker_manager.create_secure_container(
            image="python:3.11-slim",
            name="dependency_test_container",
            resource_limits=ResourceLimits(memory_mb=256, execution_timeout=60),
            security_config=SecurityConfig()
        )
        
        # Test each package
        for package in test_packages:
            logger.info(f"Testing package: {package.name}")
            
            # Test installation with validation
            result = await dep_manager.install_dependencies_securely([package], container)
            
            # Malicious packages should be blocked
            if 'malicious' in package.name or 'fake' in package.name or 'evil' in package.name:
                assert result.blocked_installs > 0, f"Malicious package {package.name} should be blocked"
                assert result.successful_installs == 0, f"Malicious package {package.name} should not install"
            
            # Typosquatting packages should be detected
            if package.name in ['requsts', 'nummpy', 'pandass']:
                assert result.blocked_installs > 0 or result.failed_installs > 0, \
                    f"Typosquatting package {package.name} should be blocked or fail"
        
        # Test legitimate package installation
        legitimate_packages = [
            PackageInfo(name="requests", version="2.31.0", ecosystem=PackageEcosystem.PYPI),
        ]
        
        result = await dep_manager.install_dependencies_securely(legitimate_packages, container)
        assert result.successful_installs > 0, "Legitimate packages should install successfully"
        
        # Verify the legitimate package is actually installed and working
        test_result = await docker_manager.execute_in_container(
            container,
            'python3 -c "import requests; print(f\'Requests version: {requests.__version__}\')"',
            timeout=10
        )
        assert test_result['success'], "Installed package should work correctly"
        assert 'Requests version:' in test_result['output'], "Package should be properly installed"
        
        await docker_manager.destroy_container(container, force=True)
        logger.info("✅ Dependency vulnerability scanning tests passed")
    
    @pytest.mark.asyncio
    async def test_network_isolation_real(self, docker_manager):
        """Test real network isolation and access controls."""
        logger.info("🧪 Testing real network isolation")
        
        # Create container with network isolation
        security_config = SecurityConfig(network_isolation=True)
        
        container = await docker_manager.create_secure_container(
            image="python:3.11-slim",
            name="network_isolation_test",
            resource_limits=ResourceLimits(execution_timeout=15),
            security_config=security_config
        )
        
        # Test 1: External network access should be blocked
        network_test_code = '''
import socket
import urllib.request
import sys

print("Testing network isolation...")

# Test 1: Try to connect to external server
try:
    response = urllib.request.urlopen('http://httpbin.org/ip', timeout=5)
    content = response.read().decode()
    print(f"EXTERNAL_ACCESS_SUCCEEDED: {content}")
except Exception as e:
    print(f"EXTERNAL_ACCESS_BLOCKED: {e}")

# Test 2: Try to create listening socket
try:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 8080))
    server_socket.listen(1)
    print("LISTENING_SOCKET_CREATED")
    server_socket.close()
except Exception as e:
    print(f"SOCKET_CREATION_BLOCKED: {e}")

# Test 3: Try DNS resolution
try:
    ip = socket.gethostbyname('google.com')
    print(f"DNS_RESOLUTION_SUCCEEDED: {ip}")
except Exception as e:
    print(f"DNS_RESOLUTION_BLOCKED: {e}")
'''
        
        result = await docker_manager.execute_in_container(
            container,
            f'python3 -c "{network_test_code}"',
            timeout=20
        )
        
        output = result.get('output', '')
        
        # Network access should be blocked
        assert 'EXTERNAL_ACCESS_SUCCEEDED' not in output, "External network access should be blocked"
        assert 'EXTERNAL_ACCESS_BLOCKED' in output, "External access should be explicitly blocked"
        
        # DNS resolution should fail in isolated network
        assert 'DNS_RESOLUTION_SUCCEEDED' not in output, "DNS resolution should be blocked"
        
        await docker_manager.destroy_container(container, force=True)
        logger.info("✅ Network isolation tests passed")
    
    @pytest.mark.asyncio
    async def test_resource_monitoring_real(self, docker_manager, docker_client):
        """Test real-time resource monitoring with actual resource usage."""
        logger.info("🧪 Testing real-time resource monitoring")
        
        # Create resource monitor
        monitor = ResourceMonitor(docker_client, monitoring_interval=0.5)
        
        # Create container with monitoring
        container = await docker_manager.create_secure_container(
            image="python:3.11-slim",
            name="resource_monitor_test",
            resource_limits=ResourceLimits(memory_mb=128, cpu_cores=0.25),
            security_config=SecurityConfig()
        )
        
        # Add container to monitoring
        monitor.add_container(container)
        await monitor.start_monitoring()
        
        # Execute resource-intensive code
        resource_intensive_code = '''
import time
import threading

print("Starting resource intensive operations...")

# CPU intensive task
def cpu_task():
    for i in range(1000000):
        _ = i * i * i

# Memory allocation task  
def memory_task():
    data = []
    for i in range(100):
        data.append([0] * 100000)  # Allocate memory
        time.sleep(0.01)

# Start tasks
cpu_thread = threading.Thread(target=cpu_task)
memory_thread = threading.Thread(target=memory_task)

cpu_thread.start()
memory_thread.start()

# Wait for completion
cpu_thread.join()
memory_thread.join()

print("Resource intensive operations completed")
'''
        
        # Start execution
        execution_task = asyncio.create_task(
            docker_manager.execute_in_container(
                container,
                f'python3 -c "{resource_intensive_code}"',
                timeout=30
            )
        )
        
        # Monitor for 10 seconds
        await asyncio.sleep(10)
        
        # Get monitoring statistics
        stats = monitor.get_container_statistics(container.container_id)
        
        # Verify monitoring collected data
        assert stats.get('sample_count', 0) > 0, "Should have collected monitoring samples"
        assert 'cpu' in stats, "Should have CPU statistics"
        assert 'memory' in stats, "Should have memory statistics"
        
        # Check for resource usage
        cpu_stats = stats.get('cpu', {})
        memory_stats = stats.get('memory', {})
        
        assert cpu_stats.get('max', 0) > 0, "Should have recorded CPU usage"
        assert memory_stats.get('max_mb', 0) > 0, "Should have recorded memory usage"
        
        # Wait for execution to complete
        try:
            result = await asyncio.wait_for(execution_task, timeout=25)
            assert result['success'], "Resource intensive code should complete successfully"
        except asyncio.TimeoutError:
            logger.warning("Execution timed out, but monitoring test continues")
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        # Get final statistics
        final_stats = monitor.get_system_statistics()
        assert final_stats['monitoring_stats']['samples_collected'] > 0, "Should have collected samples"
        
        await monitor.cleanup()
        await docker_manager.destroy_container(container, force=True)
        logger.info("✅ Resource monitoring tests passed")
    
    @pytest.mark.asyncio 
    async def test_concurrent_container_security(self, docker_manager):
        """Test security with multiple concurrent containers."""
        logger.info("🧪 Testing concurrent container security")
        
        # Create multiple containers concurrently
        container_tasks = []
        container_count = 5
        
        for i in range(container_count):
            task = docker_manager.create_secure_container(
                image="python:3.11-slim",
                name=f"concurrent_test_{i}",
                resource_limits=ResourceLimits(memory_mb=64, cpu_cores=0.1, execution_timeout=10),
                security_config=create_strict_security_config()
            )
            container_tasks.append(task)
        
        # Wait for all containers to be created
        containers = await asyncio.gather(*container_tasks)
        
        # Execute security tests in all containers concurrently
        execution_tasks = []
        test_code = '''
import os
import sys
try:
    # Try to access host filesystem
    os.listdir('/host')
    print('SECURITY_VIOLATION_HOST_ACCESS')
except:
    print('HOST_ACCESS_BLOCKED')

try:
    # Try to execute privileged commands
    os.system('mount')
    print('SECURITY_VIOLATION_MOUNT')
except:
    print('MOUNT_BLOCKED')

print(f'Container test completed: {os.getpid()}')
'''
        
        for container in containers:
            task = docker_manager.execute_in_container(
                container,
                f'python3 -c "{test_code}"',
                timeout=15
            )
            execution_tasks.append(task)
        
        # Wait for all executions to complete
        results = await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        # Verify all containers maintained security
        successful_executions = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Container {i} execution failed: {result}")
                continue
            
            output = result.get('output', '')
            
            # Security should be maintained in all containers
            assert 'SECURITY_VIOLATION_HOST_ACCESS' not in output, \
                f"Container {i} should not access host filesystem"
            assert 'SECURITY_VIOLATION_MOUNT' not in output, \
                f"Container {i} should not execute mount commands"
            assert 'HOST_ACCESS_BLOCKED' in output, \
                f"Container {i} should block host access"
            
            successful_executions += 1
        
        # Clean up all containers
        cleanup_tasks = [
            docker_manager.destroy_container(container, force=True) 
            for container in containers
        ]
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        assert successful_executions >= container_count - 1, \
            f"Most containers should execute successfully (got {successful_executions}/{container_count})"
        
        logger.info(f"✅ Concurrent container security tests passed ({successful_executions}/{container_count} successful)")
    
    @pytest.mark.asyncio
    async def test_container_lifecycle_security(self, docker_manager):
        """Test security throughout container lifecycle."""
        logger.info("🧪 Testing container lifecycle security")
        
        # Test 1: Container creation security
        container = await docker_manager.create_secure_container(
            image="python:3.11-slim",
            name="lifecycle_test_container",
            resource_limits=ResourceLimits(memory_mb=128),
            security_config=create_strict_security_config()
        )
        
        assert container.security_config.read_only_root, "Should have read-only root filesystem"
        assert container.security_config.drop_all_capabilities, "Should drop all capabilities"
        assert container.security_config.network_isolation, "Should have network isolation"
        
        # Test 2: Runtime security enforcement
        result = await docker_manager.execute_in_container(
            container,
            'python3 -c "import os; print(f\'UID: {os.getuid()}, GID: {os.getgid()}\')"',
            timeout=10
        )
        
        assert result['success'], "Basic execution should work"
        output = result['output']
        
        # Should be running as non-root user (uid 1000)
        assert 'UID: 1000' in output, "Should run as non-root user"
        assert 'GID: 1000' in output, "Should run as non-root group"
        
        # Test 3: Filesystem security
        result = await docker_manager.execute_in_container(
            container,
            'python3 -c "import os; os.makedirs(\'/test_write\', exist_ok=True); print(\'WRITE_SUCCESS\')"',
            timeout=10
        )
        
        # Should fail due to read-only filesystem
        assert not result['success'] or 'WRITE_SUCCESS' not in result.get('output', ''), \
            "Write operations should be blocked on read-only filesystem"
        
        # Test 4: Container cleanup security
        assert container.container_id in docker_manager.active_containers, \
            "Container should be tracked"
        
        cleanup_success = await docker_manager.destroy_container(container, force=True)
        assert cleanup_success, "Container cleanup should succeed"
        assert container.container_id not in docker_manager.active_containers, \
            "Container should be removed from tracking"
        
        logger.info("✅ Container lifecycle security tests passed")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])