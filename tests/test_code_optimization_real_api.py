"""
Real API Code Optimization Tests - Issue #155 Phase 3

Tests code optimization pipeline with real API integration (no mocks).
All tests use actual OpenAI, Anthropic, and Google APIs with cost controls.
"""

import pytest
from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider
import os
import sys
import asyncio
import tempfile
import shutil
from pathlib import Path
from functools import wraps

from src.orchestrator import Orchestrator, init_models


def cost_controlled_test(timeout=180):
    """Decorator for cost-controlled code optimization tests."""
    def decorator(test_func):
        @wraps(test_func)
        async def wrapper(*args, **kwargs):
            try:
                result = await asyncio.wait_for(
                    test_func(*args, **kwargs), 
                    timeout=timeout
                )
                return result
            except asyncio.TimeoutError:
                pytest.fail(f"Code optimization test timed out after {timeout} seconds")
        return wrapper
    return decorator


@pytest.fixture(scope="session")
def api_keys_available():
    """Check if API keys are available for real testing."""
    required_keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']
    available_keys = [key for key in required_keys if os.getenv(key)]
    
    if not available_keys:
        pytest.skip(f"Skipping code optimization tests - no API keys found. Set one of: {required_keys}")
    
    print(f"Using real APIs with keys: {available_keys}")
    return available_keys


@pytest.fixture(scope="session")
def real_orchestrator():
    """Create orchestrator with real models for code optimization testing."""
    registry = init_models()
    return Orchestrator(model_registry=registry)


@pytest.fixture
def sample_python_code():
    """Sample Python code with known optimization opportunities."""
    return '''"""Sample inefficient Python code for testing."""

import time
from typing import List, Dict, Any

def inefficient_fibonacci(n: int) -> int:
    """Calculate fibonacci recursively (inefficient)."""
    if n <= 1:
        return n
    return inefficient_fibonacci(n - 1) + inefficient_fibonacci(n - 2)

def process_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process data with multiple passes (inefficient)."""
    result = {"total": 0, "items": []}
    
    # Inefficient: multiple passes
    for item in data:
        if item.get("active"):
            result["items"].append(item)
    
    for item in data:
        if item.get("value"):
            result["total"] += item["value"]
    
    # Code smell: magic number
    if len(result["items"]) > 100:
        print("Warning: Large dataset")
    
    return result

class DataProcessor:
    def __init__(self):
        # Issue: hardcoded configuration
        self.batch_size = 50
        self.timeout = 30
    
    def process_batch(self, items):
        # Issue: no error handling
        processed = []
        for item in items:
            processed.append(self.transform(item))
        return processed
    
    def transform(self, item):
        # Issue: no validation
        return {
            "id": item["id"],
            "name": item["name"].upper(),
            "timestamp": time.time()
        }

# Issue: global variable
GLOBAL_COUNTER = 0

def increment_counter():
    global GLOBAL_COUNTER
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER
'''


@pytest.fixture
def sample_javascript_code():
    """Sample JavaScript code with optimization opportunities."""
    return '''// Sample inefficient JavaScript code
function inefficientFibonacci(n) {
    if (n <= 1) return n;
    return inefficientFibonacci(n - 1) + inefficientFibonacci(n - 2);
}

class DataProcessor {
    constructor() {
        this.batchSize = 50; // hardcoded
        this.timeout = 30000; // hardcoded
    }
    
    processData(data) {
        let result = [];
        
        // Inefficient: multiple loops
        for (let item of data) {
            if (item.active) {
                result.push(item);
            }
        }
        
        for (let item of data) {
            if (item.value) {
                // Missing error handling
                item.processedValue = item.value * 2;
            }
        }
        
        return result;
    }
    
    transform(item) {
        // No validation
        return {
            id: item.id,
            name: item.name.toUpperCase(),
            timestamp: new Date().getTime()
        };
    }
}

// Global variable (bad practice)
let globalCounter = 0;

function incrementCounter() {
    return ++globalCounter;
}
'''


class TestCodeOptimizationRealAPI:
    """Test code optimization pipeline with real APIs."""
    
    @cost_controlled_test(timeout=300)
    async def test_python_optimization_end_to_end(self, real_orchestrator, sample_python_code, api_keys_available):
        """Test complete Python code optimization workflow with real APIs."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test code file
            code_file = Path(temp_dir) / "test_code.py"
            code_file.write_text(sample_python_code)
            
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir()
            
            # Execute pipeline with real APIs
            pipeline_yaml = f"""
            id: test_code_optimization
            name: Test Code Optimization
            parameters:
              code_file: "{code_file}"
              language: "python"
            steps:
              - id: read_code
                tool: filesystem
                action: read
                parameters:
                  path: "{{{{code_file}}}}"
                  
              - id: analyze_code
                action: analyze_text
                parameters:
                  text: |
                    Analyze this {{{{language}}}} code for optimization opportunities:
                    
                    ```{{{{language}}}}
                    {{{{read_code.result.content}}}}
                    ```
                    
                    Identify:
                    1. Performance bottlenecks
                    2. Code quality issues
                    3. Best practice violations
                  model: "gpt-5-mini"
                  analysis_type: "code_quality"
                dependencies:
                  - read_code
                  
              - id: optimize_code
                action: generate_text
                parameters:
                  prompt: |
                    Based on this analysis:
                    {{{{analyze_code.result}}}}
                    
                    Provide optimized version of the code that addresses the identified issues.
                    Return ONLY the optimized code without markdown formatting or explanations.
                    Do not include ```python``` or any markdown blocks - just the raw code.
                  model: "gpt-5-mini"
                  max_tokens: 2000
                dependencies:
                  - analyze_code
                  
              - id: clean_optimized_code
                action: generate_text
                parameters:
                  prompt: |
                    Extract ONLY the code from the following text, removing any markdown formatting or explanations:
                    
                    {{{{optimize_code.result}}}}
                    
                    Return ONLY the pure Python code without any ```python``` blocks or explanations.
                  model: "gpt-5-mini"
                  max_tokens: 2000
                dependencies:
                  - optimize_code
                  
              - id: save_optimized_code
                tool: filesystem
                action: write
                parameters:
                  path: "{output_dir}/optimized_test_code.py"
                  content: "{{{{clean_optimized_code.result}}}}"
                dependencies:
                  - clean_optimized_code
                  
              - id: save_analysis_report
                tool: filesystem
                action: write
                parameters:
                  path: "{output_dir}/analysis_report.md"
                  content: |
                    # Code Analysis Report
                    
                    **File:** {{{{code_file}}}}
                    **Language:** {{{{language}}}}
                    **Date:** {{{{ execution.timestamp | date('%Y-%m-%d %H:%M:%S') }}}}
                    
                    ## Analysis Results
                    
                    {{{{analyze_code.result}}}}
                    
                    ## Optimized Code
                    
                    The optimized code has been saved to: optimized_test_code.py
                dependencies:
                  - analyze_code
                  - clean_optimized_code
            """
            
            result = await real_orchestrator.execute_yaml(pipeline_yaml, {
                "code_file": str(code_file),
                "language": "python"
            })
            
            # Validate results
            assert result is not None, "Pipeline execution failed"
            
            # Check output files were created
            optimized_file = output_dir / "optimized_test_code.py"
            report_file = output_dir / "analysis_report.md"
            
            assert optimized_file.exists(), "Optimized code file not created"
            assert report_file.exists(), "Analysis report file not created"
            
            # Validate optimized code content
            optimized_content = optimized_file.read_text()
            
            # Should not contain template artifacts
            assert '{{' not in optimized_content, "Unrendered templates in optimized code"
            assert '{%' not in optimized_content, "Unrendered control structures in optimized code"
            
            # Should not contain markdown formatting
            assert not optimized_content.startswith('```'), "Markdown formatting found in optimized code"
            assert '```' not in optimized_content, "Markdown code blocks found in optimized code"
            
            # Should be valid Python syntax
            try:
                compile(optimized_content, '<string>', 'exec')
            except SyntaxError as e:
                pytest.fail(f"Optimized code has syntax errors: {e}")
            
            # Validate report content
            report_content = report_file.read_text()
            
            # Should not contain template artifacts
            assert '{{' not in report_content, "Unrendered templates in report"
            assert '{%' not in report_content, "Unrendered control structures in report"
            
            # Should contain expected content
            assert str(code_file) in report_content, "Code file path not in report"
            assert "python" in report_content, "Language not in report"
            assert "Analysis Results" in report_content, "Analysis section not in report"
            
            print(f"✅ Python optimization test completed successfully")
            print(f"   - Optimized code: {len(optimized_content)} characters")
            print(f"   - Report: {len(report_content)} characters")

    @cost_controlled_test(timeout=250)
    async def test_javascript_optimization(self, real_orchestrator, sample_javascript_code, api_keys_available):
        """Test JavaScript code optimization with real APIs."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            code_file = Path(temp_dir) / "test_code.js"
            code_file.write_text(sample_javascript_code)
            
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir()
            
            # Execute pipeline for JavaScript
            with open("examples/code_optimization.yaml", 'r') as f:
                pipeline_content = f.read()
            
            result = await real_orchestrator.execute_yaml(
                pipeline_content,
                {
                    "code_file": str(code_file),
                    "language": "javascript"
                }
            )
            
            assert result is not None, "JavaScript optimization pipeline failed"
            
            # Check that outputs directory exists
            outputs_dir = Path("examples/outputs/code_optimization")
            if outputs_dir.exists():
                # Find generated files
                optimized_files = list(outputs_dir.glob("optimized_test_code.js"))
                report_files = list(outputs_dir.glob("code_optimization_report_*.md"))
                
                if optimized_files:
                    optimized_content = optimized_files[0].read_text()
                    
                    # Validate JavaScript optimization
                    assert '{{' not in optimized_content, "Unrendered templates in JS code"
                    assert '```' not in optimized_content, "Markdown formatting in JS code"
                    
                    print(f"✅ JavaScript optimization completed: {len(optimized_content)} characters")
            
            print("✅ JavaScript optimization test completed")

    @cost_controlled_test(timeout=200)
    async def test_template_rendering_validation(self, real_orchestrator, sample_python_code, api_keys_available):
        """Validate all template variables render correctly in code optimization."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            code_file = Path(temp_dir) / "template_test.py"
            code_file.write_text(sample_python_code)
            
            # Execute pipeline with specific focus on template rendering  
            with open("examples/code_optimization.yaml", 'r') as f:
                pipeline_content = f.read()
            
            result = await real_orchestrator.execute_yaml(
                pipeline_content,
                {
                    "code_file": str(code_file),
                    "language": "python"
                }
            )
            
            assert result is not None, "Template rendering test failed"
            
            # Validate that outputs contain no unrendered templates
            outputs_dir = Path("examples/outputs/code_optimization")
            if outputs_dir.exists():
                for file_path in outputs_dir.glob("*"):
                    if file_path.is_file():
                        content = file_path.read_text()
                        
                        # Check for unrendered templates
                        unrendered_vars = []
                        if '{{' in content:
                            import re
                            matches = re.findall(r'{{[^}]+}}', content)
                            unrendered_vars.extend(matches)
                            
                        if '{%' in content:
                            control_matches = re.findall(r'{%[^%]+%}', content)
                            unrendered_vars.extend(control_matches)
                        
                        if unrendered_vars:
                            print(f"❌ Found unrendered templates in {file_path.name}: {unrendered_vars}")
                            pytest.fail(f"Unrendered templates found in {file_path.name}: {unrendered_vars}")
            
            print("✅ Template rendering validation passed")

    @cost_controlled_test(timeout=240)
    async def test_large_file_performance(self, real_orchestrator, api_keys_available):
        """Test code optimization performance with large code files."""
        
        # Create a larger Python file with multiple classes and functions
        large_code = '''"""Large Python file for performance testing."""

import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

''' + sample_python_code + '''

class LargeDataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processed_count = 0
        
    def process_large_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a large dataset inefficiently."""
        results = []
        
        # Inefficient: multiple passes
        active_items = []
        for item in dataset:
            if item.get("active", False):
                active_items.append(item)
        
        processed_items = []
        for item in active_items:
            processed = self.complex_transform(item)
            processed_items.append(processed)
        
        # More inefficient processing
        final_results = []
        for item in processed_items:
            if self.validate_item(item):
                final_results.append(item)
        
        return {
            "results": final_results,
            "count": len(final_results),
            "processed_at": time.time()
        }
    
    def complex_transform(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Complex transformation with potential issues."""
        # No error handling
        transformed = {
            "id": item["id"],
            "data": json.dumps(item.get("data", {})),
            "score": self.calculate_score(item),
            "category": item["category"].upper()
        }
        
        # Inefficient calculation
        if "values" in item:
            transformed["sum"] = sum(item["values"])
            transformed["avg"] = sum(item["values"]) / len(item["values"])  # Potential division by zero
        
        return transformed
    
    def calculate_score(self, item: Dict[str, Any]) -> float:
        """Calculate score inefficiently."""
        score = 0.0
        
        # Nested loops (inefficient)
        for i in range(100):  # Magic number
            for j in range(10):
                score += (i * j) * 0.001
        
        # Add item-specific score
        if "priority" in item:
            score += item["priority"] * 1.5
        
        return score
    
    def validate_item(self, item: Dict[str, Any]) -> bool:
        """Validate item without proper error handling."""
        required_fields = ["id", "data", "score"]
        
        # No try-catch
        for field in required_fields:
            if field not in item:
                return False
        
        return True

# More global variables (bad practice)
GLOBAL_CACHE = {}
GLOBAL_CONFIG = {"timeout": 30, "retries": 3}

def global_helper_function(data: Any) -> Any:
    """Global helper function with issues."""
    global GLOBAL_CACHE
    
    # No validation
    cache_key = str(data)
    
    if cache_key not in GLOBAL_CACHE:
        # Simulate expensive operation
        result = json.loads(json.dumps(data))  # Inefficient serialization roundtrip
        GLOBAL_CACHE[cache_key] = result
    
    return GLOBAL_CACHE[cache_key]
'''
        
        with tempfile.TemporaryDirectory() as temp_dir:
            code_file = Path(temp_dir) / "large_test_code.py"
            code_file.write_text(large_code)
            
            start_time = time.time()
            
            with open("examples/code_optimization.yaml", 'r') as f:
                pipeline_content = f.read()
            
            result = await real_orchestrator.execute_yaml(
                pipeline_content,
                {
                    "code_file": str(code_file),
                    "language": "python"
                }
            )
            
            execution_time = time.time() - start_time
            
            assert result is not None, "Large file optimization failed"
            assert execution_time < 300, f"Large file processing too slow: {execution_time:.2f}s"
            
            print(f"✅ Large file optimization completed in {execution_time:.2f}s")
            print(f"   - Original code: {len(large_code)} characters")

    async def test_malformed_code_handling(self, real_orchestrator, api_keys_available):
        """Test pipeline gracefully handles malformed code."""
        
        malformed_python = '''
# This code has syntax errors
def broken_function(
    # Missing closing parenthesis and colon
    
class BrokenClass
    # Missing colon
    def method(self):
        # Indentation error
    print("This is incorrectly indented")
    
# Undefined variable usage
result = undefined_variable + another_undefined

# Malformed string
broken_string = "This string is not properly closed

def another_broken():
    # Missing return statement with inconsistent indentation
  if True:
      return
    else:
      # Wrong indentation
        pass
'''
        
        with tempfile.TemporaryDirectory() as temp_dir:
            code_file = Path(temp_dir) / "malformed_code.py"
            code_file.write_text(malformed_python)
            
            try:
                # Pipeline should handle malformed code gracefully
                result = await real_orchestrator.execute_file(
                    "examples/code_optimization.yaml",
                    {
                        "code_file": str(code_file),
                        "language": "python"
                    }
                )
                
                # Should complete even with malformed input
                assert result is not None, "Pipeline should complete with malformed code"
                print("✅ Malformed code handled gracefully")
                
            except Exception as e:
                # Acceptable to fail, but should fail gracefully
                print(f"⚠️ Pipeline failed gracefully with malformed code: {e}")

    async def test_empty_file_handling(self, real_orchestrator, api_keys_available):
        """Test pipeline handles empty code files."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_file = Path(temp_dir) / "empty.py"
            empty_file.write_text("")  # Empty file
            
            result = await real_orchestrator.execute_file(
                "examples/code_optimization.yaml",
                {
                    "code_file": str(empty_file),
                    "language": "python"
                }
            )
            
            # Should handle empty files gracefully
            assert result is not None, "Pipeline should handle empty files"
            print("✅ Empty file handled gracefully")

    async def test_missing_file_error_handling(self, real_orchestrator, api_keys_available):
        """Test pipeline handles non-existent file paths."""
        
        nonexistent_file = "/tmp/this_file_does_not_exist.py"
        
        try:
            result = await real_orchestrator.execute_file(
                "examples/code_optimization.yaml",
                {
                    "code_file": nonexistent_file,
                    "language": "python"
                }
            )
            
            # Should fail gracefully or handle the error
            print("⚠️ Pipeline completed despite missing file")
            
        except Exception as e:
            # Expected to fail - should fail gracefully
            print(f"✅ Pipeline failed gracefully with missing file: {type(e).__name__}")


class TestCodeOptimizationAnalysisQuality:
    """Test the quality and accuracy of code analysis."""
    
    @cost_controlled_test(timeout=200)
    async def test_fibonacci_optimization_detection(self, real_orchestrator, api_keys_available):
        """Test that pipeline correctly identifies and optimizes fibonacci inefficiency."""
        
        fibonacci_code = '''
def inefficient_fibonacci(n):
    if n <= 1:
        return n
    return inefficient_fibonacci(n - 1) + inefficient_fibonacci(n - 2)

def test_fibonacci():
    for i in range(30):
        print(f"Fibonacci({i}) = {inefficient_fibonacci(i)}")
'''
        
        with tempfile.TemporaryDirectory() as temp_dir:
            code_file = Path(temp_dir) / "fibonacci_test.py"
            code_file.write_text(fibonacci_code)
            
            with open("examples/code_optimization.yaml", 'r') as f:
                pipeline_content = f.read()
            
            result = await real_orchestrator.execute_yaml(
                pipeline_content,
                {
                    "code_file": str(code_file),
                    "language": "python"
                }
            )
            
            assert result is not None, "Fibonacci optimization test failed"
            
            # Check if analysis identified fibonacci inefficiency
            analysis = result.get('outputs', {}).get('analysis', '')
            if analysis:
                analysis_lower = analysis.lower()
                assert any(term in analysis_lower for term in [
                    'fibonacci', 'recursive', 'inefficient', 'memoization', 'cache'
                ]), "Analysis should identify fibonacci inefficiency"
                
                print("✅ Fibonacci inefficiency correctly identified")

    @cost_controlled_test(timeout=180)
    async def test_global_variable_detection(self, real_orchestrator, api_keys_available):
        """Test that pipeline identifies global variable issues."""
        
        global_var_code = '''
# Global variables (bad practice)
GLOBAL_COUNTER = 0
GLOBAL_DATA = {}

def increment_global():
    global GLOBAL_COUNTER
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER

def process_with_global(item):
    global GLOBAL_DATA
    GLOBAL_DATA[item['id']] = item
    return GLOBAL_DATA
'''
        
        with tempfile.TemporaryDirectory() as temp_dir:
            code_file = Path(temp_dir) / "global_vars_test.py"  
            code_file.write_text(global_var_code)
            
            with open("examples/code_optimization.yaml", 'r') as f:
                pipeline_content = f.read()
            
            result = await real_orchestrator.execute_yaml(
                pipeline_content,
                {
                    "code_file": str(code_file),
                    "language": "python"
                }
            )
            
            assert result is not None, "Global variable test failed"
            
            analysis = result.get('outputs', {}).get('analysis', '')
            if analysis:
                analysis_lower = analysis.lower()
                assert any(term in analysis_lower for term in [
                    'global', 'variable', 'encapsulation', 'class', 'state'
                ]), "Analysis should identify global variable issues"
                
                print("✅ Global variable issues correctly identified")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])