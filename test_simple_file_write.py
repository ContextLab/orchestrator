#!/usr/bin/env python3
"""
Simple File Write Test - Test file writing approaches directly
"""

import asyncio
import logging
import sys
import os
sys.path.insert(0, '/Users/jmanning/orchestrator/src')

from orchestrator.security.docker_manager import EnhancedDockerManager
from orchestrator.security.docker_manager import ResourceLimits, SecurityConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_file_write_methods():
    """Test different file writing methods in containers."""
    logger.info("🧪 Testing file writing methods...")
    
    # Create minimal docker manager
    docker_manager = EnhancedDockerManager(enable_advanced_pooling=False, performance_monitor=None)
    
    try:
        await docker_manager.start_background_tasks()
        
        # Test 1: Python container with Python write method
        container1 = await docker_manager.create_secure_container(
            image="python:3.11-slim",
            name="test_python_write",
            resource_limits=ResourceLimits(memory_mb=128, cpu_cores=0.5, execution_timeout=30, pids_limit=10),
            security_config=SecurityConfig()
        )
        
        logger.info("✅ Created Python container")
        
        test_code = "print('Hello from Python!')\\nprint('Line 2')"
        
        # Method 1: Python stdin write (current implementation)
        python_write_cmd = f'python3 -c "import sys; open(\'/tmp/test.py\', \'w\').write(sys.stdin.read())" << \'EOFCODE\'\\n{test_code}\\nEOFCODE'
        
        result1 = await docker_manager.execute_in_container(
            container1,
            python_write_cmd,
            timeout=10
        )
        
        logger.info(f"Python write result: success={result1.get('success')}, error={result1.get('error')}")
        
        if result1.get('success'):
            # Test reading back
            read_result = await docker_manager.execute_in_container(
                container1,
                "cat /tmp/test.py",
                timeout=5
            )
            logger.info(f"Read result: {read_result.get('output', '')[:100]}")
            
            # Test execution
            exec_result = await docker_manager.execute_in_container(
                container1,
                "python3 /tmp/test.py",
                timeout=5
            )
            logger.info(f"Execute result: success={exec_result.get('success')}, output={exec_result.get('output', '')}")
        
        # Test 2: Ubuntu container with tee method
        container2 = await docker_manager.create_secure_container(
            image="ubuntu:22.04",
            name="test_bash_write",
            resource_limits=ResourceLimits(memory_mb=128, cpu_cores=0.5, execution_timeout=30, pids_limit=10),
            security_config=SecurityConfig()
        )
        
        logger.info("✅ Created Ubuntu container")
        
        bash_code = "echo 'Hello from Bash!'\\necho 'Line 2'"
        
        # Method 2: tee with heredoc (current implementation)
        tee_write_cmd = f"tee /tmp/test.sh << 'EOFCODE'\\n{bash_code}\\nEOFCODE"
        
        result2 = await docker_manager.execute_in_container(
            container2,
            tee_write_cmd,
            timeout=10
        )
        
        logger.info(f"Bash write result: success={result2.get('success')}, error={result2.get('error')}")
        
        if result2.get('success'):
            # Test reading back
            read_result2 = await docker_manager.execute_in_container(
                container2,
                "cat /tmp/test.sh",
                timeout=5
            )
            logger.info(f"Read result: {read_result2.get('output', '')[:100]}")
            
            # Test execution  
            exec_result2 = await docker_manager.execute_in_container(
                container2,
                "bash /tmp/test.sh",
                timeout=5
            )
            logger.info(f"Execute result: success={exec_result2.get('success')}, output={exec_result2.get('output', '')}")
        
        # Test 3: Base64 method (alternative)
        logger.info("🧪 Testing base64 file write method...")
        
        import base64
        simple_python = "print('Hello from base64!')"
        code_b64 = base64.b64encode(simple_python.encode('utf-8')).decode('ascii')
        
        result3 = await docker_manager.execute_in_container(
            container1,
            f"echo '{code_b64}' | base64 -d > /tmp/test_b64.py",
            timeout=10
        )
        
        logger.info(f"Base64 write result: success={result3.get('success')}")
        
        if result3.get('success'):
            exec_result3 = await docker_manager.execute_in_container(
                container1,
                "python3 /tmp/test_b64.py",
                timeout=5
            )
            logger.info(f"Base64 execute result: success={exec_result3.get('success')}, output={exec_result3.get('output', '')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ File write test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await docker_manager.shutdown()

async def main():
    """Run file write test.""" 
    success = await test_file_write_methods()
    logger.info(f"✅ Test {'PASSED' if success else 'FAILED'}")
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)