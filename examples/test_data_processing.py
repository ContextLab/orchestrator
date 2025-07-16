#!/usr/bin/env python3
"""Test data processing pipeline with error recovery."""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orchestrator.orchestrator import Orchestrator
from orchestrator.core.control_system import MockControlSystem
from orchestrator.core.task import Task
from orchestrator.core.model import Model, ModelCapabilities


# Simulated data for testing
SAMPLE_DATA = {
    "records": [
        {"id": 1, "value": 100, "status": "active"},
        {"id": 2, "value": None, "status": "active"},  # Missing value
        {"id": 3, "value": 150, "status": "inactive"},
        {"id": 1, "value": 100, "status": "active"},  # Duplicate
        {"id": 4, "value": -50, "status": "active"},   # Invalid value
        {"id": 5, "value": 200, "status": "pending"},
    ],
    "total": 6
}


class DataProcessingControlSystem(MockControlSystem):
    """Control system for data processing with simulated errors."""
    
    def __init__(self):
        super().__init__(name="data-processing-control")
        self._results = {}
        self._attempt_counts = {}
        self._checkpoints = {}
    
    async def execute_task(self, task: Task, context: dict = None):
        """Execute task with error simulation."""
        # Handle $results references
        self._resolve_references(task)
        
        # Track attempts for retry testing
        self._attempt_counts[task.id] = self._attempt_counts.get(task.id, 0) + 1
        
        # Simulate different actions
        if task.action == "ingest":
            return await self._ingest_data(task)
        elif task.action == "validate_data":
            return await self._validate_data(task)
        elif task.action == "clean":
            return await self._clean_data(task)
        elif task.action == "transform":
            return await self._transform_data(task)
        elif task.action == "quality_check":
            return await self._quality_check(task)
        elif task.action == "export":
            return await self._export_data(task)
        elif task.action == "report":
            return await self._generate_report(task)
        else:
            return {"status": "completed"}
    
    def _resolve_references(self, task):
        """Resolve $results references in parameters."""
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
    
    async def _ingest_data(self, task):
        """Simulate data ingestion with potential errors."""
        print(f"[Ingest] Attempt {self._attempt_counts[task.id]}: Loading data")
        
        # Simulate error on first attempt
        if self._attempt_counts[task.id] == 1:
            print("[Ingest] Simulating connection error...")
            raise ConnectionError("Failed to connect to data source")
        
        result = {
            "data": SAMPLE_DATA,
            "stats": {
                "records_loaded": 6,
                "load_time": 1.2,
                "source": task.parameters.get("source")
            }
        }
        
        self._results[task.id] = result
        self._checkpoints[task.id] = result
        print(f"[Ingest] Successfully loaded {result['stats']['records_loaded']} records")
        return result
    
    async def _validate_data(self, task):
        """Validate data and find issues."""
        print("[Validate] Checking data quality")
        
        data = task.parameters.get("data", {})
        records = data.get("data", {}).get("records", [])
        
        issues = []
        if len(records) > 0:
            # Check for missing values
            for i, record in enumerate(records):
                if record.get("value") is None:
                    issues.append({"row": i, "issue": "missing_value", "field": "value"})
                elif record.get("value", 0) < 0:
                    issues.append({"row": i, "issue": "invalid_value", "field": "value"})
        
        result = {
            "valid": len(issues) == 0,
            "issues": issues,
            "total_issues": len(issues),
            "schema": {"fields": ["id", "value", "status"]}
        }
        
        self._results[task.id] = result
        print(f"[Validate] Found {len(issues)} data quality issues")
        return result
    
    async def _clean_data(self, task):
        """Clean data with error handling."""
        print("[Clean] Cleaning data")
        
        data = task.parameters.get("data", {})
        validation = task.parameters.get("validation_report", {})
        
        # Get original records
        records = data.get("data", {}).get("records", []).copy()
        
        # Remove duplicates
        seen = set()
        cleaned = []
        duplicates_removed = 0
        
        for record in records:
            record_id = record.get("id")
            if record_id not in seen:
                seen.add(record_id)
                # Fix missing values
                if record.get("value") is None:
                    record["value"] = 0  # Default value
                # Fix negative values
                elif record.get("value", 0) < 0:
                    record["value"] = abs(record["value"])
                cleaned.append(record)
            else:
                duplicates_removed += 1
        
        result = {
            "data": {"records": cleaned, "total": len(cleaned)},
            "original": data,
            "summary": {
                "original_count": len(records),
                "cleaned_count": len(cleaned),
                "duplicates_removed": duplicates_removed,
                "values_fixed": validation.get("total_issues", 0)
            }
        }
        
        self._results[task.id] = result
        self._checkpoints[task.id] = result
        print(f"[Clean] Cleaned {len(cleaned)} records, removed {duplicates_removed} duplicates")
        return result
    
    async def _transform_data(self, task):
        """Transform data with retry logic."""
        attempt = self._attempt_counts[task.id]
        print(f"[Transform] Attempt {attempt}: Transforming data")
        
        # Simulate failure on first attempt
        if attempt == 1:
            print("[Transform] Simulating processing error...")
            raise RuntimeError("Memory allocation failed")
        
        cleaned_data = task.parameters.get("cleaned_data", {})
        records = cleaned_data.get("data", {}).get("records", [])
        
        # Apply transformations
        transformed = []
        for record in records:
            transformed_record = record.copy()
            # Add computed field
            transformed_record["score"] = record.get("value", 0) * 1.5
            # Normalize status
            transformed_record["status"] = record.get("status", "").upper()
            transformed.append(transformed_record)
        
        result = {
            "data": {"records": transformed, "total": len(transformed)},
            "metrics": {
                "records_transformed": len(transformed),
                "transformations_applied": ["score_calculation", "status_normalization"],
                "success_rate": 1.0
            }
        }
        
        self._results[task.id] = result
        self._checkpoints[task.id] = result
        print(f"[Transform] Successfully transformed {len(transformed)} records")
        return result
    
    async def _quality_check(self, task):
        """Check data quality."""
        print("[Quality Check] Verifying data quality")
        
        transformed = task.parameters.get("transformed_data", {})
        records = transformed.get("data", {}).get("records", [])
        
        # Calculate quality metrics
        completeness = sum(1 for r in records if all(v is not None for v in r.values())) / len(records)
        accuracy = 1.0  # Simulated
        consistency = 1.0  # Simulated
        
        result = {
            "passed": completeness >= 0.95,
            "metrics": {
                "completeness": completeness,
                "accuracy": accuracy,
                "consistency": consistency,
                "overall_score": (completeness + accuracy + consistency) / 3
            }
        }
        
        self._results[task.id] = result
        print(f"[Quality Check] Quality score: {result['metrics']['overall_score']:.2%}")
        return result
    
    async def _export_data(self, task):
        """Export processed data."""
        print("[Export] Exporting data")
        
        data = task.parameters.get("data", {})
        destination = task.parameters.get("destination", "./output/")
        
        result = {
            "path": f"{destination}processed_data.json",
            "format": "json",
            "size_mb": 0.1,
            "records_exported": data.get("data", {}).get("total", 0),
            "compressed": True
        }
        
        self._results[task.id] = result
        print(f"[Export] Exported to {result['path']}")
        return result
    
    async def _generate_report(self, task):
        """Generate processing report."""
        print("[Report] Generating report")
        
        result = {
            "report": """# Data Processing Report

## Summary
Successfully processed data with error recovery and quality checks.

## Pipeline Statistics
- Records ingested: 6
- Duplicates removed: 1
- Values fixed: 2
- Final record count: 5
- Quality score: 100%

## Error Recovery
- Ingestion: Recovered from connection error (1 retry)
- Transformation: Recovered from memory error (1 retry)

## Quality Metrics
- Completeness: 100%
- Accuracy: 100%
- Consistency: 100%

## Performance
- Total processing time: 5.2s
- Checkpoints created: 3
- Error recovery time: 2.1s
""",
            "summary": "Processed 5 records with 100% quality score"
        }
        
        self._results[task.id] = result
        return result


class MockAutoModel(Model):
    """Mock model for AUTO resolution."""
    
    def __init__(self):
        capabilities = ModelCapabilities(
            supported_tasks=["reasoning"],
            context_window=4096,
            languages=["en"]
        )
        super().__init__(
            name="Mock Auto Model",
            provider="mock",
            capabilities=capabilities
        )
    
    async def generate(self, prompt, **kwargs):
        if "batch size" in prompt:
            return "100"
        elif "schema" in prompt:
            return "auto_inferred"
        elif "transformations" in prompt:
            return "normalization,enrichment"
        elif "format" in prompt:
            return "json"
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


async def test_data_processing():
    """Test data processing pipeline with error recovery."""
    print("Testing Data Processing Pipeline with Error Recovery")
    print("=" * 50)
    
    # Load pipeline
    with open("pipelines/data_processing.yaml", "r") as f:
        pipeline_yaml = f.read()
    
    # Initialize orchestrator
    control_system = DataProcessingControlSystem()
    orchestrator = Orchestrator(control_system=control_system)
    
    # Set up AUTO resolver
    mock_model = MockAutoModel()
    orchestrator.model_registry.register_model(mock_model)
    orchestrator.yaml_compiler.ambiguity_resolver.model = mock_model
    
    try:
        # Execute pipeline
        print("\nExecuting pipeline (errors will be simulated and recovered)...")
        print("-" * 50)
        
        results = await orchestrator.execute_yaml(
            pipeline_yaml,
            context={
                "data_source": "/path/to/data.csv",
                "processing_mode": "batch",
                "error_tolerance": 0.05
            }
        )
        
        print("\n" + "-" * 50)
        print("Pipeline execution completed with error recovery!")
        
        # Show checkpoint recovery
        print(f"\nCheckpoints used: {len(control_system._checkpoints)}")
        print(f"Total retry attempts: {sum(v for v in control_system._attempt_counts.values())}")
        
        # Show results
        if "generate_report" in results:
            report = results["generate_report"]
            if isinstance(report, dict) and "summary" in report:
                print(f"\nFinal Summary: {report['summary']}")
        
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_data_processing())
    
    if success:
        print("\n✅ Data processing pipeline with error recovery succeeded!")
    else:
        print("\n❌ Data processing pipeline failed!")
        sys.exit(1)