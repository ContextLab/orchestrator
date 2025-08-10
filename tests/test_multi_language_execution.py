"""Real Multi-Language Execution Tests - Issue #206 Task 2.2

Comprehensive tests for multi-language code execution with real Docker containers,
security validation, and language-specific execution environments. NO MOCKS.
"""

import pytest
import asyncio
import logging

from orchestrator.tools.multi_language_executor import (
    MultiLanguageExecutor,
    MultiLanguageExecutorTool,
    Language,
    LANGUAGE_CONFIGS
)
from orchestrator.security.docker_manager import EnhancedDockerManager, ResourceLimits, SecurityConfig
from orchestrator.tools.secure_tool_executor import ExecutionMode

# Configure logging for test visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMultiLanguageExecutor:
    """Test multi-language code execution with real containers."""
    
    @pytest.fixture
    async def docker_manager(self):
        """Create Docker manager for testing."""
        manager = EnhancedDockerManager()
        await manager.start_background_tasks()
        yield manager
        await manager.shutdown()
    
    @pytest.fixture
    async def multi_lang_executor(self, docker_manager):
        """Create multi-language executor for testing."""
        executor = MultiLanguageExecutor(docker_manager)
        yield executor
        # No cleanup needed - containers are destroyed after each execution
    
    def test_supported_languages(self, multi_lang_executor):
        """Test that all expected languages are supported."""
        logger.info("🧪 Testing supported languages")
        
        supported = multi_lang_executor.get_supported_languages()
        logger.info(f"Supported languages: {supported}")
        
        # Check that core languages are supported
        essential_languages = ['python', 'javascript', 'bash', 'java', 'go']
        for lang in essential_languages:
            assert lang in supported, f"Language {lang} should be supported"
        
        assert len(supported) >= 5, "Should support at least 5 languages"
        
        logger.info("✅ Supported languages test passed")
    
    @pytest.mark.parametrize("language,code,expected_output", [
        (
            Language.PYTHON,
            "print('Hello from Python!')\nprint(f'2 + 2 = {2 + 2}')",
            "Hello from Python!"
        ),
        (
            Language.NODEJS,
            "console.log('Hello from Node.js!');\nconsole.log(`3 + 3 = ${3 + 3}`);",
            "Hello from Node.js!"
        ),
        (
            Language.BASH,
            "echo 'Hello from Bash!'\necho 'Current date:'\ndate",
            "Hello from Bash!"
        ),
        (
            Language.GO,
            """package main
import "fmt"

func main() {
    fmt.Println("Hello from Go!")
    fmt.Printf("5 + 5 = %d\\n", 5+5)
}""",
            "Hello from Go!"
        ),
    ])
    @pytest.mark.asyncio
    async def test_language_execution_real(self, multi_lang_executor, language, code, expected_output):
        """Test real code execution for different languages."""
        logger.info(f"🧪 Testing {language.value} execution")
        
        result = await multi_lang_executor.execute_code(
            code=code,
            language=language,
            timeout=30
        )
        
        logger.info(f"Execution result for {language.value}: success={result.success}")
        if result.output:
            logger.info(f"Output: {result.output[:100]}...")
        if result.error:
            logger.info(f"Error: {result.error}")
        
        assert result.success, f"{language.value} execution should succeed. Error: {result.error}"
        assert expected_output in result.output, f"Output should contain '{expected_output}'"
        
        # Check performance metrics
        assert result.performance_metrics is not None, "Should have performance metrics"
        assert 'language' in result.performance_metrics, "Should record language"
        assert result.performance_metrics['language'] == language.value, "Should record correct language"
        
        logger.info(f"✅ {language.value} execution test passed")
    
    @pytest.mark.asyncio
    async def test_language_detection_real(self, multi_lang_executor):
        """Test automatic language detection from code patterns."""
        logger.info("🧪 Testing language detection")
        
        test_cases = [
            ("print('Hello Python')\nimport sys", Language.PYTHON),
            ("console.log('Hello JavaScript');\nconst x = 5;", Language.NODEJS),
            ("echo 'Hello Bash'\nif [ -f file.txt ]; then", Language.BASH),
            ("package main\nfunc main() {\n    fmt.Println()", Language.GO),
            ("public class Test {\n    public static void main", Language.JAVA),
        ]
        
        for code, expected_lang in test_cases:
            detected = multi_lang_executor.detect_language(code)
            assert detected == expected_lang, f"Should detect {expected_lang.value} from code pattern"
        
        logger.info("✅ Language detection test passed")
    
    @pytest.mark.asyncio
    async def test_filename_based_detection(self, multi_lang_executor):
        """Test language detection from filename extensions."""
        logger.info("🧪 Testing filename-based detection")
        
        test_cases = [
            ("test.py", Language.PYTHON),
            ("app.js", Language.NODEJS),
            ("script.sh", Language.BASH),
            ("main.go", Language.GO),
            ("Hello.java", Language.JAVA),
            ("program.cpp", Language.CPP),
            ("code.c", Language.C),
            ("script.rb", Language.RUBY),
        ]
        
        for filename, expected_lang in test_cases:
            detected = multi_lang_executor.detect_language("", filename=filename)
            assert detected == expected_lang, f"Should detect {expected_lang.value} from {filename}"
        
        logger.info("✅ Filename-based detection test passed")
    
    @pytest.mark.asyncio
    async def test_resource_limits_enforcement_real(self, multi_lang_executor):
        """Test resource limit enforcement across different languages."""
        logger.info("🧪 Testing resource limits enforcement")
        
        # Test with Python memory bomb
        memory_intensive_python = """
data = []
try:
    for i in range(50):  # Try to allocate ~50MB
        data.append([0] * 250000)  # ~1MB per iteration
        print(f"Allocated chunk {i+1}")
    print("MEMORY_BOMB_SUCCESS")
except MemoryError:
    print("MEMORY_LIMIT_HIT")
except Exception as e:
    print(f"OTHER_ERROR: {e}")
"""
        
        # Set tight memory limit
        tight_limits = ResourceLimits(
            memory_mb=32,  # 32MB limit
            cpu_cores=0.1,
            execution_timeout=20
        )
        
        result = await multi_lang_executor.execute_code(
            code=memory_intensive_python,
            language=Language.PYTHON,
            resource_limits=tight_limits,
            timeout=25
        )
        
        # Should be limited by memory constraints
        output = result.output or ""
        memory_constrained = (
            'MEMORY_LIMIT_HIT' in output or
            'MEMORY_BOMB_SUCCESS' not in output or
            not result.success
        )
        
        assert memory_constrained, f"Memory limits should be enforced. Output: {output}"
        
        logger.info("✅ Resource limits enforcement test passed")
    
    @pytest.mark.asyncio
    async def test_security_isolation_real(self, multi_lang_executor):
        """Test security isolation across different languages."""
        logger.info("🧪 Testing security isolation")
        
        # Test filesystem access restrictions
        test_cases = [
            (
                Language.PYTHON,
                """
try:
    with open('/etc/passwd', 'r') as f:
        print("FILESYSTEM_BREACH")
except Exception as e:
    print(f"FILESYSTEM_BLOCKED: {e}")
"""
            ),
            (
                Language.NODEJS,
                """
const fs = require('fs');
try {
    const data = fs.readFileSync('/etc/passwd', 'utf8');
    console.log('FILESYSTEM_BREACH');
} catch (e) {
    console.log('FILESYSTEM_BLOCKED:', e.message);
}
"""
            ),
            (
                Language.BASH,
                """
if cat /etc/passwd 2>/dev/null; then
    echo "FILESYSTEM_BREACH"
else
    echo "FILESYSTEM_BLOCKED"
fi
"""
            ),
        ]
        
        for language, code in test_cases:
            result = await multi_lang_executor.execute_code(
                code=code,
                language=language,
                timeout=15
            )
            
            output = result.output or ""
            
            # Should block or contain filesystem access
            filesystem_secured = (
                'FILESYSTEM_BLOCKED' in output or
                'FILESYSTEM_BREACH' not in output or
                not result.success
            )
            
            assert filesystem_secured, f"{language.value} filesystem access should be secured. Output: {output}"
        
        logger.info("✅ Security isolation test passed")
    
    @pytest.mark.asyncio
    async def test_compilation_languages_real(self, multi_lang_executor):
        """Test compiled languages (Java, C++, C) with real compilation."""
        logger.info("🧪 Testing compiled languages")
        
        # Test C++ compilation and execution
        cpp_code = """
#include <iostream>
#include <vector>

int main() {
    std::cout << "Hello from C++!" << std::endl;
    
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    int sum = 0;
    for (int num : numbers) {
        sum += num;
    }
    
    std::cout << "Sum: " << sum << std::endl;
    return 0;
}
"""
        
        result = await multi_lang_executor.execute_code(
            code=cpp_code,
            language=Language.CPP,
            timeout=60  # Allow time for compilation
        )
        
        logger.info(f"C++ execution result: success={result.success}")
        if result.output:
            logger.info(f"C++ output: {result.output}")
        if result.error:
            logger.info(f"C++ error: {result.error}")
        
        assert result.success, f"C++ compilation and execution should succeed. Error: {result.error}"
        assert "Hello from C++" in result.output, "Should execute compiled C++ code"
        assert "Sum: 15" in result.output, "Should calculate correct sum"
        
        # Check that compilation was performed
        if result.performance_metrics:
            assert result.performance_metrics.get('compiled') == True, "Should indicate compilation was performed"
        
        logger.info("✅ Compiled languages test passed")


class TestMultiLanguageExecutorTool:
    """Test the multi-language executor tool wrapper."""
    
    @pytest.fixture
    async def multi_lang_tool(self):
        """Create multi-language executor tool for testing."""
        tool = MultiLanguageExecutorTool()
        yield tool
        if hasattr(tool, 'shutdown'):
            await tool.shutdown()
    
    @pytest.mark.asyncio
    async def test_tool_python_execution(self, multi_lang_tool):
        """Test tool wrapper for Python execution."""
        logger.info("🧪 Testing multi-language tool Python execution")
        
        result = await multi_lang_tool.execute(
            code="print('Hello from tool!')\nprint(f'Math: {7 * 8}')",
            language="python",
            timeout=30
        )
        
        assert result['success'], f"Tool execution should succeed. Error: {result.get('error')}"
        
        # Check output format
        if 'output' in result:
            assert 'Hello from tool!' in result['output'], "Should contain expected output"
        
        # Check that supported languages are included
        assert 'supported_languages' in result, "Should include supported languages list"
        assert 'python' in result['supported_languages'], "Should list Python as supported"
        
        logger.info("✅ Multi-language tool test passed")
    
    @pytest.mark.asyncio
    async def test_tool_auto_detection(self, multi_lang_tool):
        """Test automatic language detection in tool."""
        logger.info("🧪 Testing tool auto-detection")
        
        # JavaScript code with auto-detection
        js_code = """
console.log('Auto-detected JavaScript!');
const numbers = [1, 2, 3, 4, 5];
const sum = numbers.reduce((a, b) => a + b, 0);
console.log(`Sum: ${sum}`);
"""
        
        result = await multi_lang_tool.execute(
            code=js_code,
            language="auto",  # Should auto-detect as JavaScript
            timeout=30
        )
        
        assert result['success'], f"Auto-detection should work. Error: {result.get('error')}"
        
        if 'performance_metrics' in result and result['performance_metrics']:
            assert 'language' in result['performance_metrics'], "Should record detected language"
        
        logger.info("✅ Tool auto-detection test passed")
    
    @pytest.mark.asyncio
    async def test_tool_error_handling(self, multi_lang_tool):
        """Test error handling in tool wrapper."""
        logger.info("🧪 Testing tool error handling")
        
        # Test with empty code
        result = await multi_lang_tool.execute(code="")
        assert not result['success'], "Should fail with empty code"
        assert 'error' in result, "Should provide error message"
        assert 'supported_languages' in result, "Should still provide supported languages"
        
        # Test with unsupported language
        result = await multi_lang_tool.execute(
            code="print 'Hello'",
            language="cobol"  # Unsupported language
        )
        
        # Should either fail gracefully or fall back to auto-detection
        assert 'supported_languages' in result, "Should provide supported languages info"
        
        logger.info("✅ Tool error handling test passed")
    
    @pytest.mark.asyncio
    async def test_tool_resource_customization(self, multi_lang_tool):
        """Test custom resource limits in tool."""
        logger.info("🧪 Testing tool resource customization")
        
        result = await multi_lang_tool.execute(
            code="import time; print('Starting...'); time.sleep(1); print('Done!')",
            language="python",
            timeout=5,
            memory_limit_mb=128,
            mode="sandboxed"
        )
        
        assert result['success'], f"Custom resource execution should work. Error: {result.get('error')}"
        
        if 'resource_usage' in result:
            assert result['resource_usage'] is not None, "Should include resource usage info"
        
        logger.info("✅ Tool resource customization test passed")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])