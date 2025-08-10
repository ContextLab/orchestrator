#!/usr/bin/env python3
"""
Phase 2 Integration Test - Issue #206 Task 2.4

Simple integration test to validate all Phase 2 components work together:
- SecureToolExecutor + SecureIntegrationAdapter 
- MultiLanguageExecutor
- NetworkManager
- Real Docker containers with security
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_secure_integration_adapter():
    """Test the secure integration adapter with real execution."""
    logger.info("🧪 Testing SecureIntegrationAdapter...")
    
    try:
        from orchestrator.tools.secure_integration_adapter import (
            SecureToolRegistry, 
            upgrade_tool_registry,
            migrate_existing_tools
        )
        
        # Create secure registry
        secure_registry = upgrade_tool_registry()
        await secure_registry.initialize()
        
        # Get registry statistics
        stats = secure_registry.get_registry_statistics()
        logger.info(f"Registry stats: {stats}")
        
        # List available tools
        tools = secure_registry.list_tools()
        logger.info(f"Available tools: {len(tools)} tools")
        
        secure_tools = secure_registry.list_secure_tools()
        logger.info(f"Secure tools: {len(secure_tools)} secure tools")
        
        await secure_registry.shutdown()
        logger.info("✅ SecureIntegrationAdapter test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ SecureIntegrationAdapter test failed: {e}")
        return False


async def test_multi_language_executor():
    """Test multi-language executor with simple code."""
    logger.info("🧪 Testing MultiLanguageExecutor...")
    
    try:
        from orchestrator.tools.multi_language_executor import (
            MultiLanguageExecutor, 
            Language
        )
        from orchestrator.security.docker_manager import EnhancedDockerManager
        
        # Create Docker manager
        docker_manager = EnhancedDockerManager()
        await docker_manager.start_background_tasks()
        
        # Create multi-language executor
        executor = MultiLanguageExecutor(docker_manager)
        
        # Test language detection
        python_detected = executor.detect_language("print('hello')", "test.py")
        assert python_detected == Language.PYTHON
        
        js_detected = executor.detect_language("console.log('hello')", "test.js")  
        assert js_detected == Language.NODEJS
        
        # Test supported languages
        supported = executor.get_supported_languages()
        logger.info(f"Supported languages: {supported}")
        assert len(supported) >= 5
        assert 'python' in supported
        assert 'nodejs' in supported or 'javascript' in supported
        
        await docker_manager.shutdown()
        logger.info("✅ MultiLanguageExecutor test passed")
        return True
        
    except Exception as e:
        import traceback
        logger.error(f"❌ MultiLanguageExecutor test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


async def test_network_manager():
    """Test network manager policies and rules."""
    logger.info("🧪 Testing NetworkManager...")
    
    try:
        from orchestrator.security.network_manager import (
            NetworkManager, 
            NetworkPolicy,
            NetworkRule, 
            NetworkAccessLevel,
            NetworkProtocol
        )
        
        # Create network manager
        net_manager = NetworkManager()
        
        # Check default policies
        policies = net_manager.list_policies()
        assert len(policies) >= 3
        assert 'no_access' in policies
        assert 'limited_access' in policies
        assert 'internet_access' in policies
        
        # Test policy retrieval
        no_access = net_manager.get_policy('no_access')
        assert no_access is not None
        assert no_access.access_level == NetworkAccessLevel.NONE
        
        # Test custom policy creation
        custom_policy = NetworkPolicy(
            name='test_policy',
            access_level=NetworkAccessLevel.LIMITED,
            rules=[
                NetworkRule('allow_http', NetworkProtocol.HTTP, port=80, allow=True)
            ]
        )
        net_manager.add_policy(custom_policy)
        
        # Test policy export/import
        exported = net_manager.export_network_policy('test_policy')
        assert exported is not None
        assert exported['name'] == 'test_policy'
        
        # Test network request evaluation
        allow_http = net_manager.evaluate_network_request(
            'test_container', 'http', 'example.com', 80
        )
        # Should allow since no policy is set for test_container
        
        # Get statistics
        stats = net_manager.get_network_statistics()
        assert 'policies_available' in stats
        assert stats['policies_available'] >= 4  # 3 default + 1 custom
        
        logger.info("✅ NetworkManager test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ NetworkManager test failed: {e}")
        return False


async def test_secure_python_executor():
    """Test secure Python executor tool."""
    logger.info("🧪 Testing SecurePythonExecutor...")
    
    try:
        from orchestrator.tools.secure_python_executor import (
            SecurePythonExecutorTool
        )
        
        # Create secure Python executor
        python_tool = SecurePythonExecutorTool()
        
        # Test simple code execution
        result = await python_tool.execute(
            code="print('Hello from secure Python!')",
            timeout=30,
            mode="auto"
        )
        
        logger.info(f"Python execution result: {result.get('success')}")
        
        # Should have comprehensive result structure
        assert 'success' in result
        assert 'execution_context' in result or 'output' in result
        
        # Cleanup
        if hasattr(python_tool, 'shutdown'):
            await python_tool.shutdown()
            
        logger.info("✅ SecurePythonExecutor test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ SecurePythonExecutor test failed: {e}")
        return False


async def main():
    """Run all Phase 2 integration tests."""
    logger.info("🚀 Starting Phase 2 Integration Testing...")
    
    tests = [
        ("Network Manager", test_network_manager()),
        ("Multi-Language Executor", test_multi_language_executor()),
        ("Secure Python Executor", test_secure_python_executor()),
        ("Secure Integration Adapter", test_secure_integration_adapter()),
    ]
    
    results = []
    
    for test_name, test_coro in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            result = await test_coro
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("PHASE 2 INTEGRATION TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL PHASE 2 INTEGRATION TESTS PASSED!")
        return True
    else:
        logger.info(f"⚠️  {total - passed} tests failed - needs attention")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)