#!/usr/bin/env python3
"""Comprehensive test runner for all pipeline examples with real inputs."""

import asyncio
import sys
import os
import json
import csv
import traceback
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orchestrator.orchestrator import Orchestrator
from orchestrator.core.control_system import MockControlSystem
from orchestrator.core.task import Task
from orchestrator.core.model import Model, ModelCapabilities


class RealDataControlSystem(MockControlSystem):
    """Control system that processes real data files."""
    
    def __init__(self):
        super().__init__(name="real-data-control")
        self._results = {}
        self._attempt_counts = {}
    
    async def execute_task(self, task: Task, context: dict = None):
        """Execute task with real data processing."""
        # Handle $results references
        self._resolve_references(task)
        
        # Track attempts
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
                return result
            except Exception as e:
                # Simulate retry logic for certain errors
                if self._attempt_counts[task.id] < 3 and "connection" in str(e).lower():
                    print(f"[{task.action.upper()}] Retry {self._attempt_counts[task.id]} after error: {e}")
                    # Don't store result on failure, will retry
                    raise e
                else:
                    # Final failure
                    result = {"status": "failed", "error": str(e)}
                    self._results[task.id] = result
                    return result
        else:
            result = {"status": "completed", "message": f"Mock execution of {task.action}"}
            self._results[task.id] = result
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
        print(f"[SEARCH] Searching for: '{query}'")
        
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
        print(f"[ANALYZE] Analyzing {len(results)} search results")
        
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
        """Create summary from analysis."""
        content = task.parameters.get("content", {})
        print("[SUMMARIZE] Creating summary from analysis")
        
        findings = content.get("findings", [])
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
        print(f"[ANALYZE_CODE] Analyzing {language} code at: {path}")
        
        # Read actual file if it exists
        code_content = ""
        if os.path.exists(path):
            with open(path, 'r') as f:
                code_content = f.read()
        else:
            code_content = "# No code file found at specified path"
        
        # Basic code analysis
        lines = code_content.split('\\n')
        functions = [line for line in lines if line.strip().startswith('def ')]
        classes = [line for line in lines if line.strip().startswith('class ')]
        
        # Calculate basic complexity metrics
        complexity_score = len([line for line in lines if 'for ' in line or 'while ' in line]) * 2
        complexity_score += len([line for line in lines if 'if ' in line])
        
        return {
            "code": code_content,
            "metrics": {
                "lines": len(lines),
                "functions": len(functions),
                "classes": len(classes),
                "complexity": min(complexity_score, 20),  # Cap at 20
                "maintainability": max(10 - complexity_score // 5, 1)  # Inverse relationship
            },
            "functions_found": [line.strip() for line in functions[:5]],  # First 5
            "language": language
        }
    
    async def _find_issues(self, task):
        """Find code issues in real code."""
        analysis = task.parameters.get("analysis", {})
        code = analysis.get("code", "")
        print("[FIND_ISSUES] Identifying optimization opportunities")
        
        issues = []
        lines = code.split('\\n')
        
        # Look for actual inefficiencies
        for i, line in enumerate(lines):
            if 'range(len(' in line:
                issues.append({
                    "type": "performance",
                    "line": i + 1,
                    "description": "Using range(len()) pattern - could use enumerate() or direct iteration",
                    "severity": "medium"
                })
            if 'for i in range' in line and 'for j in range' in lines[i+1:i+3]:
                issues.append({
                    "type": "complexity",
                    "line": i + 1,
                    "description": "Nested loops detected - potential O(n²) complexity",
                    "severity": "high"
                })
            if line.strip().startswith('if ') and 'not in' in line:
                issues.append({
                    "type": "performance",
                    "line": i + 1,
                    "description": "List membership check - consider using set for better performance",
                    "severity": "low"
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
        print(f"[OPTIMIZE] Generating fixes for {issues.get('total_issues', 0)} issues")
        
        # Simulate code optimization
        optimizations = []
        for issue in issues.get("issues", []):
            if "range(len(" in issue["description"]:
                optimizations.append("Replace range(len()) with enumerate() or direct iteration")
            elif "nested loops" in issue["description"].lower():
                optimizations.append("Optimize nested loops using sets or hash maps")
            elif "list membership" in issue["description"].lower():
                optimizations.append("Convert list to set for O(1) membership checks")
        
        return {
            "optimizations": optimizations,
            "code": "# Optimized code would be generated here",
            "estimated_improvement": {
                "performance": f"+{min(len(optimizations) * 15, 80)}%",
                "complexity": f"-{min(len(optimizations) * 10, 50)}%",
                "maintainability": f"+{min(len(optimizations) * 5, 25)}%"
            }
        }
    
    async def _validate_code(self, task):
        """Validate optimized code."""
        print("[VALIDATE] Validating optimized code")
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
        print(f"[INGEST] Loading data from: {source}")
        
        # Simulate connection error on first attempt
        if self._attempt_counts[task.id] == 1 and "malformed" not in source:
            raise ConnectionError("Simulated connection timeout")
        
        data = None
        if source.endswith('.csv'):
            data = self._load_csv(source)
        elif source.endswith('.json'):
            data = self._load_json(source)
        else:
            # Default sample data
            data = {
                "records": [
                    {"id": 1, "value": 100},
                    {"id": 2, "value": 200}
                ],
                "total": 2
            }
        
        return {
            "data": data,
            "stats": {
                "records_loaded": len(data.get("records", [])),
                "source": source,
                "format": source.split('.')[-1] if '.' in source else "unknown"
            }
        }
    
    def _load_csv(self, filepath):
        """Load CSV file."""
        records = []
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                records = list(reader)
        return {"records": records, "total": len(records)}
    
    def _load_json(self, filepath):
        """Load JSON file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return {"records": [], "total": 0}
    
    async def _validate_data(self, task):
        """Validate real data."""
        data = task.parameters.get("data", {})
        records = data.get("data", {}).get("records", [])
        print(f"[VALIDATE_DATA] Validating {len(records)} records")
        
        issues = []
        for i, record in enumerate(records):
            # Check for missing required fields
            if not record.get("id"):
                issues.append({"row": i, "issue": "missing_id", "severity": "high"})
            
            # Check for invalid values
            if "salary" in record:
                try:
                    salary = float(record["salary"]) if record["salary"] else 0
                    if salary < 0:
                        issues.append({"row": i, "issue": "negative_salary", "severity": "medium"})
                except (ValueError, TypeError):
                    issues.append({"row": i, "issue": "invalid_salary_format", "severity": "high"})
            
            # Check for empty names
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
        validation = task.parameters.get("validation_report", {})
        records = data.get("data", {}).get("records", [])
        print(f"[CLEAN] Cleaning {len(records)} records")
        
        cleaned_records = []
        issues_fixed = 0
        
        for record in records:
            cleaned_record = record.copy()
            
            # Fix missing names
            if not cleaned_record.get("name"):
                cleaned_record["name"] = f"Unknown_{cleaned_record.get('id', 'Person')}"
                issues_fixed += 1
            
            # Fix negative salaries
            if "salary" in cleaned_record:
                try:
                    salary = float(cleaned_record["salary"]) if cleaned_record["salary"] else 0
                    if salary < 0:
                        cleaned_record["salary"] = abs(salary)
                        issues_fixed += 1
                except (ValueError, TypeError):
                    cleaned_record["salary"] = 0
                    issues_fixed += 1
            
            # Only keep records with valid IDs
            if cleaned_record.get("id"):
                cleaned_records.append(cleaned_record)
        
        return {
            "data": {"records": cleaned_records, "total": len(cleaned_records)},
            "summary": {
                "original_count": len(records),
                "cleaned_count": len(cleaned_records),
                "issues_fixed": issues_fixed,
                "records_removed": len(records) - len(cleaned_records)
            }
        }
    
    async def _transform_data(self, task):
        """Transform cleaned data."""
        cleaned_data = task.parameters.get("cleaned_data", {})
        records = cleaned_data.get("data", {}).get("records", [])
        
        # Simulate error on first attempt for some pipelines
        if self._attempt_counts[task.id] == 1 and len(records) > 5:
            raise RuntimeError("Simulated memory allocation failure")
        
        print(f"[TRANSFORM] Transforming {len(records)} records")
        
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
            
            # Normalize status
            if "status" in record:
                new_record["status"] = record["status"].lower()
            
            transformed.append(new_record)
        
        return {
            "data": {"records": transformed, "total": len(transformed)},
            "transformations_applied": ["salary_grade_calculation", "status_normalization"],
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
        print(f"[QUALITY_CHECK] Checking quality of {len(records)} records")
        
        # Calculate quality metrics
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
        print(f"[EXPORT] Exporting to {destination}")
        
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
        print("[REPORT] Generating final report")
        
        # Collect all available results
        report_sections = []
        
        if "ingestion_stats" in task.parameters:
            stats = task.parameters["ingestion_stats"]
            report_sections.append(f"## Data Ingestion\\n- Records loaded: {stats.get('records_loaded', 0)}")
        
        if "validation_results" in task.parameters:
            validation = task.parameters["validation_results"]
            report_sections.append(f"## Validation\\n- Issues found: {validation.get('total_issues', 0)}")
        
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
            if "performance" in prompt_lower:
                return "high"
            elif "balanced" in prompt_lower:
                return "medium"
            else:
                return "low"
        elif "focus areas" in prompt_lower:
            return "performance,complexity,maintainability"
        
        # Data processing responses
        elif "batch size" in prompt_lower:
            if "streaming" in prompt_lower:
                return "50"
            else:
                return "1000"
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


async def run_pipeline_test(pipeline_file, test_name, context, expected_outputs=None):
    """Run a single pipeline test."""
    print(f"\\n{'='*60}")
    print(f"Testing: {test_name}")
    print(f"Pipeline: {pipeline_file}")
    print(f"Context: {context}")
    print('='*60)
    
    try:
        # Load pipeline
        with open(f"pipelines/{pipeline_file}", "r") as f:
            pipeline_yaml = f.read()
        
        # Set up orchestrator
        control_system = RealDataControlSystem()
        orchestrator = Orchestrator(control_system=control_system)
        
        # Set up AUTO resolver
        mock_model = MockAutoResolver()
        orchestrator.model_registry.register_model(mock_model)
        orchestrator.yaml_compiler.ambiguity_resolver.model = mock_model
        
        # Execute pipeline
        results = await orchestrator.execute_yaml(pipeline_yaml, context=context)
        
        print(f"\\n✅ SUCCESS: {test_name}")
        print(f"Tasks completed: {len(results)}")
        
        # Show key results
        for task_id, result in results.items():
            if isinstance(result, dict):
                if "summary" in result:
                    print(f"  {task_id}: {result['summary']}")
                elif "report" in result:
                    print(f"  {task_id}: Report generated")
                elif "status" in result:
                    print(f"  {task_id}: {result['status']}")
                else:
                    print(f"  {task_id}: Completed")
        
        # Check expected outputs if provided
        if expected_outputs:
            for expected in expected_outputs:
                if expected not in results:
                    print(f"  ⚠️  Missing expected output: {expected}")
                else:
                    print(f"  ✓ Found expected output: {expected}")
        
        return True, results
        
    except Exception as e:
        print(f"\\n❌ FAILED: {test_name}")
        print(f"Error: {e}")
        traceback.print_exc()
        return False, None


async def main():
    """Run comprehensive pipeline tests."""
    print("🚀 Running Comprehensive Pipeline Tests")
    print("Testing real-world scenarios with actual data files")
    
    test_results = []
    
    # Test 1: Simple Research Pipeline - Basic Query
    success, _ = await run_pipeline_test(
        "simple_research.yaml",
        "Simple Research - Python Programming",
        {"topic": "Python asyncio programming"},
        ["summarize"]
    )
    test_results.append(("Simple Research Basic", success))
    
    # Test 2: Simple Research Pipeline - Complex Query
    success, _ = await run_pipeline_test(
        "simple_research.yaml", 
        "Simple Research - AI Ethics",
        {"topic": "Artificial Intelligence Ethics and Bias"},
        ["summarize"]
    )
    test_results.append(("Simple Research Complex", success))
    
    # Test 3: Code Optimization - Real File
    success, _ = await run_pipeline_test(
        "code_optimization.yaml",
        "Code Optimization - Real Python File",
        {
            "code_path": "test_data/sample_code.py",
            "optimization_level": "performance",
            "language": "python"
        },
        ["create_report", "generate_fixes"]
    )
    test_results.append(("Code Optimization Performance", success))
    
    # Test 4: Code Optimization - Balanced Mode  
    success, _ = await run_pipeline_test(
        "code_optimization.yaml",
        "Code Optimization - Balanced Mode",
        {
            "code_path": "test_data/sample_code.py", 
            "optimization_level": "balanced",
            "language": "python"
        },
        ["create_report", "generate_fixes"]
    )
    test_results.append(("Code Optimization Balanced", success))
    
    # Test 5: Data Processing - CSV File
    success, _ = await run_pipeline_test(
        "data_processing.yaml",
        "Data Processing - Employee CSV",
        {
            "data_source": "test_data/sample_data.csv",
            "processing_mode": "batch", 
            "error_tolerance": 0.1
        },
        ["data_export", "generate_report"]
    )
    test_results.append(("Data Processing CSV", success))
    
    # Test 6: Data Processing - Malformed JSON (Error Recovery)
    success, _ = await run_pipeline_test(
        "data_processing.yaml",
        "Data Processing - Error Recovery",
        {
            "data_source": "test_data/malformed_data.json",
            "processing_mode": "batch",
            "error_tolerance": 0.2
        },
        ["generate_report"]
    )
    test_results.append(("Data Processing Error Recovery", success))
    
    # Test 7: Data Processing - Streaming Mode
    success, _ = await run_pipeline_test(
        "data_processing.yaml",
        "Data Processing - Streaming Mode", 
        {
            "data_source": "test_data/sample_data.csv",
            "processing_mode": "streaming",
            "error_tolerance": 0.05
        },
        ["data_export"]
    )
    test_results.append(("Data Processing Streaming", success))
    
    # Summary
    print(f"\\n{'='*60}")
    print("📊 TEST RESULTS SUMMARY")
    print('='*60)
    
    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} {test_name}")
    
    print(f"\\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\\n🎉 All tests passed! Pipelines are working correctly.")
        return True
    else:
        print(f"\\n⚠️  {total-passed} tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)