#!/usr/bin/env python3
"""Test code optimization pipeline."""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orchestrator.orchestrator import Orchestrator
from orchestrator.core.control_system import MockControlSystem
from orchestrator.core.task import Task
from orchestrator.core.model import Model, ModelCapabilities


# Sample unoptimized code for testing
SAMPLE_CODE = """
def calculate_sum(numbers):
    total = 0
    for i in range(len(numbers)):
        total = total + numbers[i]
    return total

def find_duplicates(items):
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates
"""

OPTIMIZED_CODE = """
def calculate_sum(numbers):
    return sum(numbers)

def find_duplicates(items):
    seen = set()
    duplicates = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return list(duplicates)
"""


# Mock actions
async def mock_analyze_code(task):
    """Mock code analysis."""
    print(f"[Code Analysis] Analyzing {task.parameters.get('language', 'python')} code")
    return {
        "code": SAMPLE_CODE,
        "metrics": {
            "complexity": 8,
            "performance": 3,
            "maintainability": 6
        },
        "line_count": 15,
        "functions": ["calculate_sum", "find_duplicates"]
    }


async def mock_find_issues(task):
    """Mock issue identification."""
    analysis = task.parameters.get("analysis", {})
    print(f"[Find Issues] Identifying optimization opportunities")
    return {
        "issues": [
            {
                "type": "performance",
                "function": "calculate_sum",
                "description": "Using manual loop instead of built-in sum()",
                "severity": "medium"
            },
            {
                "type": "complexity",
                "function": "find_duplicates", 
                "description": "O(n²) algorithm can be optimized to O(n)",
                "severity": "high"
            }
        ],
        "total_issues": 2
    }


async def mock_optimize(task):
    """Mock code optimization."""
    issues = task.parameters.get("issues", {})
    print(f"[Optimize] Generating optimizations for {issues.get('total_issues', 0)} issues")
    return {
        "code": OPTIMIZED_CODE,
        "changes": [
            "Replaced manual loop with sum() function",
            "Optimized duplicate detection using sets"
        ],
        "estimated_improvement": {
            "performance": "+60%",
            "readability": "+40%"
        }
    }


async def mock_validate(task):
    """Mock validation."""
    print(f"[Validate] Validating optimized code")
    return {
        "valid": True,
        "tests_passed": True,
        "functionality_preserved": True,
        "improvements": {
            "complexity": -3,
            "performance": +60,
            "maintainability": +2
        }
    }


async def mock_report(task):
    """Mock report generation."""
    print(f"[Report] Generating optimization report")
    return {
        "document": """# Code Optimization Report

## Summary
Successfully optimized 2 functions with significant performance improvements.

## Issues Found
1. **calculate_sum**: Manual loop replaced with built-in sum()
2. **find_duplicates**: O(n²) algorithm optimized to O(n)

## Performance Improvements
- Overall performance: +60%
- Code complexity: -37.5%
- Maintainability: +33%

## Optimized Code
```python
def calculate_sum(numbers):
    return sum(numbers)

def find_duplicates(items):
    seen = set()
    duplicates = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return list(duplicates)
```

## Validation Results
✅ All tests passed
✅ Functionality preserved
✅ No breaking changes detected
""",
        "summary": "2 functions optimized, +60% performance improvement"
    }


class CodeOptimizationControlSystem(MockControlSystem):
    """Control system for code optimization pipeline."""
    
    def __init__(self):
        super().__init__(
            name="code-optimization-control",
            config={
                "capabilities": {
                    "supported_actions": [
                        "analyze_code", "find_issues", "optimize", 
                        "validate", "report"
                    ],
                }
            }
        )
        self.actions = {
            "analyze_code": mock_analyze_code,
            "find_issues": mock_find_issues,
            "optimize": mock_optimize,
            "validate": mock_validate,
            "report": mock_report
        }
        self._results = {}
    
    async def execute_task(self, task: Task, context: dict = None):
        """Execute a task."""
        # Handle $results references
        for key, value in task.parameters.items():
            if isinstance(value, str) and value.startswith("$results."):
                parts = value.split(".")
                if len(parts) >= 2:
                    task_id = parts[1]
                    if task_id in self._results:
                        if len(parts) > 2:
                            # Handle nested access like $results.task.field
                            result = self._results[task_id]
                            for part in parts[2:]:
                                if isinstance(result, dict) and part in result:
                                    result = result[part]
                                else:
                                    result = None
                                    break
                            task.parameters[key] = result
                        else:
                            task.parameters[key] = self._results[task_id]
        
        # Execute the action
        handler = self.actions.get(task.action)
        if handler:
            result = await handler(task)
            self._results[task.id] = result
            return result
        else:
            return {"status": "completed"}


class MockAutoResolverModel(Model):
    """Mock model for AUTO tag resolution."""
    
    def __init__(self):
        capabilities = ModelCapabilities(
            supported_tasks=["reasoning", "generation"],
            context_window=4096,
            languages=["en"]
        )
        super().__init__(
            name="Mock Auto Resolver",
            provider="mock",
            capabilities=capabilities
        )
    
    async def generate(self, prompt, **kwargs):
        """Generate responses for AUTO tags."""
        if "threshold" in prompt:
            return "medium"
        elif "focus areas" in prompt:
            return "performance,complexity"
        return "default"
    
    async def generate_structured(self, prompt, schema, **kwargs):
        return {"value": await self.generate(prompt, **kwargs)}
    
    async def validate_response(self, response, schema):
        return True
    
    def estimate_tokens(self, text):
        return len(text.split())
    
    def estimate_cost(self, input_tokens, output_tokens):
        return 0.0
    
    async def health_check(self):
        return True


async def test_code_optimization():
    """Test code optimization pipeline."""
    print("Testing Code Optimization Pipeline")
    print("=" * 50)
    
    # Load pipeline
    with open("pipelines/code_optimization.yaml", "r") as f:
        pipeline_yaml = f.read()
    
    # Initialize orchestrator
    control_system = CodeOptimizationControlSystem()
    orchestrator = Orchestrator(control_system=control_system)
    
    # Set up AUTO resolver
    mock_model = MockAutoResolverModel()
    orchestrator.model_registry.register_model(mock_model)
    orchestrator.yaml_compiler.ambiguity_resolver.model = mock_model
    
    try:
        # Execute pipeline
        results = await orchestrator.execute_yaml(
            pipeline_yaml,
            context={
                "code_path": "/path/to/code.py",
                "optimization_level": "balanced",
                "language": "python"
            }
        )
        
        print("\nPipeline execution completed!")
        print("\nResults Summary:")
        
        # Show key results
        if "create_report" in results:
            report = results["create_report"]
            if isinstance(report, dict) and "summary" in report:
                print(f"\nSummary: {report['summary']}")
        
        if "generate_fixes" in results:
            fixes = results["generate_fixes"]
            if isinstance(fixes, dict) and "estimated_improvement" in fixes:
                print(f"\nEstimated Improvements:")
                for metric, value in fixes["estimated_improvement"].items():
                    print(f"  - {metric}: {value}")
        
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_code_optimization())
    
    if success:
        print("\n✅ Code optimization pipeline executed successfully!")
    else:
        print("\n❌ Code optimization pipeline failed!")
        sys.exit(1)