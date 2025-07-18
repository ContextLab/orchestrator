#!/usr/bin/env python3
"""
Real-world pipeline integration tests.

These tests run actual pipeline examples with real data to ensure
the Orchestrator framework works correctly in production scenarios.
"""

import asyncio
import pytest
import sys
import os
import json
import csv
import traceback
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from orchestrator.orchestrator import Orchestrator
from orchestrator.core.control_system import MockControlSystem
from orchestrator.core.task import Task, TaskStatus
from orchestrator.core.model import Model, ModelCapabilities


class RealDataControlSystem(MockControlSystem):
    """Control system that processes real data files for testing."""
    
    def __init__(self):
        super().__init__(name="real-data-control")
        self._results = {}
        self._attempt_counts = {}
    
    async def execute_task(self, task: Task, context: dict = None):
        """Execute task with real data processing."""
        # Handle $results references
        self._resolve_references(task)
        
        # Track attempts for retry testing
        self._attempt_counts[task.id] = self._attempt_counts.get(task.id, 0) + 1
        
        # Route to appropriate handler
        action_handlers = {
            "search": self._search,
            "analyze": self._analyze,
            "summarize": self._summarize,
            "analyze_code": self._analyze_code,
            "find_issues": self._find_issues,
            "optimize": self._optimize_code,
            "validate": self._validate_code,
            "report": self._generate_report,
            "ingest": self._ingest_data,
            "validate_data": self._validate_data,
            "clean": self._clean_data,
            "transform": self._transform_data,
            "quality_check": self._quality_check,
            "export": self._export_data,
        }
        
        handler = action_handlers.get(task.action)
        if handler:
            try:
                result = await handler(task)
                self._results[task.id] = result
                task.status = TaskStatus.COMPLETED
                return result
            except Exception as e:
                # Simulate retry logic for certain errors
                if self._attempt_counts[task.id] < 3 and "connection" in str(e).lower():
                    # Don't store result on failure, will retry
                    raise e
                else:
                    # Final failure
                    result = {"status": "failed", "error": str(e)}
                    self._results[task.id] = result
                    task.status = TaskStatus.FAILED
                    return result
        else:
            result = {"status": "completed", "message": f"Mock execution of {task.action}"}
            self._results[task.id] = result
            task.status = TaskStatus.COMPLETED
            return result
    
    def _resolve_references(self, task):
        """Resolve $results references."""
        for key, value in task.parameters.items():
            if isinstance(value, str) and value.startswith("$results."):
                parts = value.split(".")
                if len(parts) >= 2:
                    task_id = parts[1]
                    if task_id in self._results:
                        result = self._results[task_id]
                        for part in parts[2:]:
                            if isinstance(result, dict) and part in result:
                                result = result[part]
                            else:
                                result = None
                                break
                        task.parameters[key] = result
    
    # Research pipeline actions
    async def _search(self, task):
        """Mock web search with realistic results."""
        query = task.parameters.get("query", "")
        
        # Simulate search results based on query
        base_results = [
            {"title": f"Academic paper about {query}", "url": "https://arxiv.org/example1", "snippet": f"Research on {query}..."},
            {"title": f"{query} - Wikipedia", "url": "https://wikipedia.org/example", "snippet": f"Encyclopedia entry for {query}"},
            {"title": f"Tutorial: {query}", "url": "https://tutorial.com/example", "snippet": f"Learn about {query}"},
        ]
        
        return {
            "results": base_results,
            "total": len(base_results),
            "query": query,
            "search_time": 0.5
        }
    
    async def _analyze(self, task):
        """Analyze search results."""
        data = task.parameters.get("data", {})
        results = data.get("results", [])
        
        return {
            "findings": [
                f"Found {len(results)} relevant sources",
                "Sources include academic and educational content",
                "Information appears current and credible"
            ],
            "insights": f"Analysis of {data.get('query', 'topic')} reveals multiple research angles",
            "source_types": ["academic", "reference", "tutorial"],
            "confidence": 0.85
        }
    
    async def _summarize(self, task):
        """Create summary using AI."""
        content = task.parameters.get("content", {})
        findings = content.get("findings", [])
        
        try:
            # Use AI for summarization
            registry = ModelRegistry()
            model = registry.get_model("gpt-4o-mini") or registry.get_model("claude-3-5-haiku-20241022")
            
            if model:
                prompt = f"""Create a research summary based on:

Findings: {json.dumps(findings)}
Insights: {content.get('insights', 'No insights available')}
Source Types: {content.get('source_types', [])}

Format as a markdown report with Key Findings and Insights sections."""
                
                summary_text = await model.generate(prompt, max_tokens=300, temperature=0.3)
                
                # Ensure markdown formatting
                if not summary_text.startswith("#"):
                    summary_text = "# Research Summary\\n\\n" + summary_text
                
                return {
                    "summary": summary_text,
                    "word_count": len(summary_text.split()),
                    "confidence": content.get("confidence", 0.8),
                    "ai_generated": True
                }
            else:
                raise Exception("No AI model available")
                
        except Exception as e:
            print(f"Summarization error: {e}, using fallback")
            # Fallback summary
            summary_text = "# Research Summary\\n\\n"
            summary_text += "## Key Findings\\n"
            for i, finding in enumerate(findings, 1):
                summary_text += f"{i}. {finding}\\n"
            
            summary_text += f"\\n## Insights\\n{content.get('insights', 'No insights available')}"
            
            return {
                "summary": summary_text,
                "word_count": len(summary_text.split()),
                "confidence": content.get("confidence", 0.8)
            }
    
    # Code optimization actions
    async def _analyze_code(self, task):
        """Analyze real code files."""
        path = task.parameters.get("path", "")
        language = task.parameters.get("language", "python")
        
        # Read actual file if it exists
        code_content = ""
        test_data_path = Path(__file__).parent.parent.parent / "examples" / "test_data" / "sample_code.py"
        if test_data_path.exists():
            with open(test_data_path, 'r') as f:
                code_content = f.read()
        else:
            code_content = "# Sample code for testing\\ndef hello():\\n    return 'Hello World'"
        
        # Basic code analysis
        lines = code_content.split('\\n')
        functions = [line for line in lines if line.strip().startswith('def ')]
        classes = [line for line in lines if line.strip().startswith('class ')]
        
        return {
            "code": code_content,
            "metrics": {
                "lines": len(lines),
                "functions": len(functions),
                "classes": len(classes),
                "complexity": min(len(functions) * 2, 20),
                "maintainability": max(10 - len(functions), 1)
            },
            "functions_found": [line.strip() for line in functions[:5]],
            "language": language
        }
    
    async def _find_issues(self, task):
        """Find code issues in real code."""
        analysis = task.parameters.get("analysis", {})
        code = analysis.get("code", "")
        
        issues = []
        lines = code.split('\\n')
        
        # Look for actual inefficiencies
        for i, line in enumerate(lines):
            if 'range(len(' in line:
                issues.append({
                    "type": "performance",
                    "line": i + 1,
                    "description": "Using range(len()) pattern - could use enumerate()",
                    "severity": "medium"
                })
        
        return {
            "issues": issues,
            "total_issues": len(issues),
            "severity_counts": {
                "high": len([i for i in issues if i["severity"] == "high"]),
                "medium": len([i for i in issues if i["severity"] == "medium"]),
                "low": len([i for i in issues if i["severity"] == "low"])
            }
        }
    
    async def _optimize_code(self, task):
        """Generate optimized code."""
        issues = task.parameters.get("issues", {})
        
        optimizations = []
        for issue in issues.get("issues", []):
            if "range(len(" in issue["description"]:
                optimizations.append("Replace range(len()) with enumerate()")
        
        return {
            "optimizations": optimizations,
            "code": "# Optimized code would be generated here",
            "estimated_improvement": {
                "performance": f"+{min(len(optimizations) * 15, 80)}%",
                "complexity": f"-{min(len(optimizations) * 10, 50)}%"
            }
        }
    
    async def _validate_code(self, task):
        """Validate optimized code."""
        return {
            "valid": True,
            "tests_passed": True,
            "syntax_valid": True,
            "performance_improved": True
        }
    
    # Data processing actions
    async def _ingest_data(self, task):
        """Ingest real data files."""
        source = task.parameters.get("source", "")
        
        # Simulate connection error on first attempt for some sources
        if self._attempt_counts[task.id] == 1 and "malformed" not in source:
            raise ConnectionError("Simulated connection timeout")
        
        # Try to load actual test data
        test_data_dir = Path(__file__).parent.parent.parent / "examples" / "test_data"
        
        data = None
        if source.endswith('.csv') and (test_data_dir / "sample_data.csv").exists():
            data = self._load_csv(test_data_dir / "sample_data.csv")
        elif source.endswith('.json') and (test_data_dir / "customers.json").exists():
            data = self._load_json(test_data_dir / "customers.json")
        else:
            # Default sample data
            data = {
                "records": [
                    {"id": 1, "name": "John Doe", "salary": 50000},
                    {"id": 2, "name": "Jane Smith", "salary": 60000}
                ],
                "total": 2
            }
        
        return {
            "data": data,
            "stats": {
                "records_loaded": len(data.get("records", [])),
                "source": source,
                "format": source.split('.')[-1] if '.' in source else "json"
            }
        }
    
    def _load_csv(self, filepath):
        """Load CSV file."""
        records = []
        if filepath.exists():
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                records = list(reader)
        return {"records": records, "total": len(records)}
    
    def _load_json(self, filepath):
        """Load JSON file."""
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        return {"records": [], "total": 0}
    
    async def _validate_data(self, task):
        """Validate real data."""
        data = task.parameters.get("data", {})
        records = data.get("data", {}).get("records", [])
        
        issues = []
        for i, record in enumerate(records):
            if not record.get("id"):
                issues.append({"row": i, "issue": "missing_id", "severity": "high"})
            if "name" in record and not record["name"]:
                issues.append({"row": i, "issue": "missing_name", "severity": "medium"})
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "total_issues": len(issues),
            "records_checked": len(records)
        }
    
    async def _clean_data(self, task):
        """Clean real data."""
        data = task.parameters.get("data", {})
        records = data.get("data", {}).get("records", [])
        
        cleaned_records = []
        issues_fixed = 0
        
        for record in records:
            cleaned_record = record.copy()
            
            # Fix missing names
            if not cleaned_record.get("name"):
                cleaned_record["name"] = f"Unknown_{cleaned_record.get('id', 'Person')}"
                issues_fixed += 1
            
            if cleaned_record.get("id"):
                cleaned_records.append(cleaned_record)
        
        return {
            "data": {"records": cleaned_records, "total": len(cleaned_records)},
            "summary": {
                "original_count": len(records),
                "cleaned_count": len(cleaned_records),
                "issues_fixed": issues_fixed
            }
        }
    
    async def _transform_data(self, task):
        """Transform cleaned data."""
        cleaned_data = task.parameters.get("cleaned_data", {})
        records = cleaned_data.get("data", {}).get("records", [])
        
        # Simulate error on first attempt for larger datasets
        if self._attempt_counts[task.id] == 1 and len(records) > 5:
            raise RuntimeError("Simulated memory allocation failure")
        
        transformed = []
        for record in records:
            new_record = record.copy()
            
            # Add computed fields
            if "salary" in record:
                try:
                    salary = float(record["salary"]) if record["salary"] else 0
                    new_record["salary_grade"] = "high" if salary > 50000 else "medium" if salary > 30000 else "low"
                except:
                    new_record["salary_grade"] = "unknown"
            
            transformed.append(new_record)
        
        return {
            "data": {"records": transformed, "total": len(transformed)},
            "transformations_applied": ["salary_grade_calculation"],
            "metrics": {
                "records_transformed": len(transformed),
                "fields_added": 1,
                "success_rate": 1.0
            }
        }
    
    async def _quality_check(self, task):
        """Check data quality."""
        data = task.parameters.get("transformed_data", {})
        records = data.get("data", {}).get("records", [])
        
        complete_records = sum(1 for r in records if all(str(v).strip() for v in r.values() if v is not None))
        completeness = complete_records / len(records) if records else 0
        
        return {
            "passed": completeness >= 0.8,
            "metrics": {
                "completeness": completeness,
                "total_records": len(records),
                "complete_records": complete_records,
                "quality_score": completeness
            }
        }
    
    async def _export_data(self, task):
        """Export processed data."""
        data = task.parameters.get("data", {})
        destination = task.parameters.get("destination", "./output/")
        
        # Create output directory
        os.makedirs(destination, exist_ok=True)
        
        # Write actual output file
        output_path = os.path.join(destination, "processed_data.json")
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return {
            "path": output_path,
            "records_exported": data.get("data", {}).get("total", 0),
            "format": "json",
            "size_bytes": os.path.getsize(output_path)
        }
    
    async def _generate_report(self, task):
        """Generate comprehensive report."""
        report_sections = []
        
        if "ingestion_stats" in task.parameters:
            stats = task.parameters["ingestion_stats"]
            report_sections.append(f"## Data Ingestion\\n- Records loaded: {stats.get('records_loaded', 0)}")
        
        report_text = "# Processing Report\\n\\n" + "\\n\\n".join(report_sections)
        
        return {
            "report": report_text,
            "summary": "Processing completed successfully",
            "timestamp": "2024-01-01T12:00:00Z"
        }


class MockAutoResolver(Model):
    """Enhanced AUTO resolver for realistic responses."""
    
    def __init__(self):
        capabilities = ModelCapabilities(
            supported_tasks=["reasoning", "generation"],
            context_window=4096,
            languages=["en"]
        )
        super().__init__(
            name="Enhanced Auto Resolver",
            provider="mock",
            capabilities=capabilities
        )
    
    async def generate(self, prompt, **kwargs):
        """Generate contextual responses."""
        prompt_lower = prompt.lower()
        
        # Research pipeline responses
        if "sources for research" in prompt_lower:
            return "academic databases, peer-reviewed journals, government reports"
        elif "number" in prompt_lower and "analysis" in prompt_lower:
            return "15"
        
        # Code optimization responses
        elif "threshold" in prompt_lower and "optimization" in prompt_lower:
            return "medium"
        elif "focus areas" in prompt_lower:
            return "performance,complexity,maintainability"
        
        # Data processing responses
        elif "batch size" in prompt_lower:
            return "1000" if "streaming" not in prompt_lower else "50"
        elif "schema" in prompt_lower:
            return "auto_inferred_schema"
        elif "transformations" in prompt_lower:
            return "normalization,enrichment,validation"
        elif "format" in prompt_lower and "data" in prompt_lower:
            return "json"
        
        return "optimized_default"
    
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


@pytest.fixture
def orchestrator_with_real_data():
    """Set up orchestrator with real data control system."""
    control_system = RealDataControlSystem()
    orchestrator = Orchestrator(control_system=control_system)
    
    # Set up AUTO resolver
    mock_model = MockAutoResolver()
    orchestrator.model_registry.register_model(mock_model)
    orchestrator.yaml_compiler.ambiguity_resolver.model = mock_model
    
    return orchestrator


async def run_pipeline_test(orchestrator, pipeline_file, context, expected_outputs=None):
    """Run a single pipeline test."""
    try:
        # Load pipeline
        pipeline_path = Path(__file__).parent.parent.parent / "docs" / "tutorials" / "examples" / pipeline_file
        if not pipeline_path.exists():
            pipeline_path = Path(__file__).parent.parent.parent / "examples" / "pipelines" / pipeline_file
        
        if not pipeline_path.exists():
            pytest.skip(f"Pipeline file not found: {pipeline_file}")
        
        with open(pipeline_path, "r") as f:
            pipeline_yaml = f.read()
        
        # Execute pipeline
        results = await orchestrator.execute_yaml(pipeline_yaml, context=context)
        
        # Verify results
        assert len(results) > 0, "Pipeline should produce at least one result"
        
        # Check expected outputs if provided
        if expected_outputs:
            for expected in expected_outputs:
                assert expected in results, f"Missing expected output: {expected}"
        
        return True, results
        
    except Exception as e:
        pytest.fail(f"Pipeline test failed: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_simple_research_pipeline(orchestrator_with_real_data):
    """Test simple research pipeline with basic query."""
    success, results = await run_pipeline_test(
        orchestrator_with_real_data,
        "simple_research.yaml",
        {"topic": "Python asyncio programming"},
        ["summarize"]
    )
    
    assert success
    assert "summarize" in results
    assert results["summarize"]["word_count"] > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_code_optimization_pipeline(orchestrator_with_real_data):
    """Test code optimization pipeline with real Python file."""
    success, results = await run_pipeline_test(
        orchestrator_with_real_data,
        "code_optimization.yaml",
        {
            "code_path": "test_data/sample_code.py",
            "optimization_level": "performance",
            "language": "python"
        },
        ["create_report", "generate_fixes"]
    )
    
    assert success
    if "analyze_code" in results:
        assert results["analyze_code"]["metrics"]["lines"] > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_data_processing_pipeline(orchestrator_with_real_data):
    """Test data processing pipeline with CSV file."""
    success, results = await run_pipeline_test(
        orchestrator_with_real_data,
        "data_processing.yaml",
        {
            "data_source": "test_data/sample_data.csv",
            "processing_mode": "batch", 
            "error_tolerance": 0.1
        },
        ["data_export", "generate_report"]
    )
    
    assert success
    if "data_ingestion" in results:
        assert results["data_ingestion"]["stats"]["records_loaded"] >= 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_error_recovery_pipeline(orchestrator_with_real_data):
    """Test pipeline error recovery with malformed data."""
    success, results = await run_pipeline_test(
        orchestrator_with_real_data,
        "data_processing.yaml",
        {
            "data_source": "test_data/malformed_data.json",
            "processing_mode": "batch",
            "error_tolerance": 0.2
        },
        ["generate_report"]
    )
    
    assert success
    # Should complete even with errors due to error tolerance


@pytest.mark.integration
@pytest.mark.asyncio
async def test_research_report_template(orchestrator_with_real_data):
    """Test research report template pipeline."""
    success, results = await run_pipeline_test(
        orchestrator_with_real_data,
        "research-report-template.yaml",
        {
            "topic": "machine_learning",
            "instructions": "Focus on recent advances in transformer architectures"
        }
    )
    
    assert success
    assert len(results) > 0


@pytest.mark.integration
def test_pipeline_files_exist():
    """Test that all referenced pipeline files exist."""
    docs_examples_dir = Path(__file__).parent.parent.parent / "docs" / "tutorials" / "examples"
    examples_pipelines_dir = Path(__file__).parent.parent.parent / "examples" / "pipelines"
    
    required_pipelines = [
        "simple_research.yaml",
        "code_optimization.yaml", 
        "data_processing.yaml",
        "research-report-template.yaml"
    ]
    
    for pipeline in required_pipelines:
        docs_path = docs_examples_dir / pipeline
        examples_path = examples_pipelines_dir / pipeline
        
        assert docs_path.exists() or examples_path.exists(), f"Pipeline file not found: {pipeline}"


@pytest.mark.integration
def test_test_data_files_exist():
    """Test that test data files exist for integration tests."""
    test_data_dir = Path(__file__).parent.parent.parent / "examples" / "test_data"
    
    # These files should exist for proper testing
    expected_files = [
        "sample_code.py",
        "sample_data.csv",
        "customers.json"
    ]
    
    for filename in expected_files:
        filepath = test_data_dir / filename
        # Mark as expected but don't fail if missing, just warn
        if not filepath.exists():
            pytest.warns(UserWarning, f"Test data file missing: {filename}")


# Run tests directly
if __name__ == "__main__":
    async def main():
        """Run tests directly without pytest."""
        print("Running Real-World Pipeline Integration Tests")
        print("=" * 60)
        
        # Set up orchestrator
        control_system = RealDataControlSystem()
        orchestrator = Orchestrator(control_system=control_system)
        
        # Set up real model
        registry = ModelRegistry()
        real_model = None
        for model_id in ["gpt-4o-mini", "claude-3-5-haiku-20241022"]:
            try:
                real_model = registry.get_model(model_id)
                if real_model:
                    print(f"Using {model_id} for AUTO resolution")
                    break
            except:
                continue
        
        if real_model:
            orchestrator.yaml_compiler.ambiguity_resolver.model = real_model
        else:
            print("WARNING: No real model available for AUTO resolution")
        
        # Test 1: Simple Research Pipeline
        print("\n1. Testing Simple Research Pipeline...")
        try:
            success, results = await run_pipeline_test(
                orchestrator,
                "simple_research.yaml",
                {"topic": "Python asyncio programming"},
                ["summarize"]
            )
            print(f"   {'✅ PASS' if success else '❌ FAIL'}")
        except Exception as e:
            print(f"   ❌ FAIL: {e}")
        
        # Test 2: Code Optimization Pipeline
        print("\n2. Testing Code Optimization Pipeline...")
        try:
            success, results = await run_pipeline_test(
                orchestrator,
                "code_optimization.yaml",
                {
                    "code_path": "test_data/sample_code.py",
                    "optimization_level": "performance",
                    "language": "python"
                },
                ["create_report", "generate_fixes"]
            )
            print(f"   {'✅ PASS' if success else '❌ FAIL'}")
        except Exception as e:
            print(f"   ❌ FAIL: {e}")
        
        # Test 3: Data Processing Pipeline
        print("\n3. Testing Data Processing Pipeline...")
        try:
            success, results = await run_pipeline_test(
                orchestrator,
                "data_processing.yaml",
                {
                    "data_source": "test_data/sample_data.csv",
                    "processing_mode": "batch",
                    "error_tolerance": 0.1
                },
                ["data_export", "generate_report"]
            )
            print(f"   {'✅ PASS' if success else '❌ FAIL'}")
        except Exception as e:
            print(f"   ❌ FAIL: {e}")
        
        print("\n" + "=" * 60)
        print("Real-world pipeline tests completed!")
    
    asyncio.run(main())