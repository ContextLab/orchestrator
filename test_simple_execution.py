#!/usr/bin/env python3
"""
Simple Code Execution Test - Debug workload execution issues
"""

import asyncio
import logging
from orchestrator.security.docker_manager import EnhancedDockerManager
from orchestrator.tools.multi_language_executor import MultiLanguageExecutor, Language
from orchestrator.analytics.performance_monitor import PerformanceMonitor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_simple_execution():
    """Test simple code execution."""
    logger.info("🧪 Testing simple code execution...")
    
    # Create components
    performance_monitor = PerformanceMonitor(collection_interval=0.5)
    docker_manager = EnhancedDockerManager(enable_advanced_pooling=False, performance_monitor=performance_monitor)
    executor = MultiLanguageExecutor(docker_manager)
    
    try:
        # Start monitoring systems
        await performance_monitor.start_monitoring()
        await docker_manager.start_background_tasks()
        
        logger.info("✅ Systems started")
        
        # Test simple Python code
        simple_python_code = "print('Hello from Python!')"
        
        logger.info("🐍 Executing simple Python code...")
        result = await executor.execute_code(simple_python_code, Language.PYTHON, timeout=30)
        
        logger.info(f"Python result: Success={result.success}")
        if result.success:
            logger.info(f"Output: {result.output}")
        else:
            logger.error(f"Error: {result.error}")
        
        # Test simple bash code
        simple_bash_code = "echo 'Hello from Bash!'"
        
        logger.info("💻 Executing simple Bash code...")
        result2 = await executor.execute_code(simple_bash_code, Language.BASH, timeout=30)
        
        logger.info(f"Bash result: Success={result2.success}")
        if result2.success:
            logger.info(f"Output: {result2.output}")
        else:
            logger.error(f"Error: {result2.error}")
        
        return result.success or result2.success
        
    except Exception as e:
        logger.error(f"❌ Simple execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        await docker_manager.shutdown()
        await performance_monitor.stop_monitoring()

async def main():
    """Run simple execution test.""" 
    success = await test_simple_execution()
    logger.info(f"✅ Test {'PASSED' if success else 'FAILED'}")
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)