#!/usr/bin/env python3
"""Test successful data processing without errors."""

import asyncio
import sys
import os
import json
import csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orchestrator.orchestrator import Orchestrator
from orchestrator.core.control_system import MockControlSystem
from orchestrator.core.task import Task
from orchestrator.core.model import Model, ModelCapabilities


class SuccessfulDataControlSystem(MockControlSystem):
    """Control system that processes data successfully without errors."""
    
    def __init__(self):
        super().__init__(name="successful-data-control")
        self._results = {}
    
    async def execute_task(self, task: Task, context: dict = None):
        """Execute task successfully."""
        # Handle $results references
        self._resolve_references(task)
        
        # Route to appropriate handler
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
    
    async def _ingest_data(self, task):
        """Successfully ingest data."""
        source = task.parameters.get("source", "")
        print(f"[INGEST] Loading data from: {source}")
        
        # Load actual CSV data
        records = []
        if os.path.exists(source):
            with open(source, 'r') as f:
                reader = csv.DictReader(f)
                records = list(reader)
        
        result = {
            "data": {"records": records, "total": len(records)},
            "stats": {
                "records_loaded": len(records),
                "source": source,
                "format": "csv"
            }
        }
        self._results[task.id] = result
        return result
    
    async def _validate_data(self, task):
        """Validate data."""
        data = task.parameters.get("data", {})
        records = data.get("data", {}).get("records", [])
        print(f"[VALIDATE] Checking {len(records)} records")
        
        # No issues with small clean dataset
        result = {
            "valid": True,
            "issues": [],
            "total_issues": 0,
            "records_checked": len(records)
        }
        self._results[task.id] = result
        return result
    
    async def _clean_data(self, task):
        """Clean data (no changes needed)."""
        data = task.parameters.get("data", {})
        records = data.get("data", {}).get("records", [])
        print(f"[CLEAN] Processing {len(records)} records")
        
        result = {
            "data": {"records": records, "total": len(records)},
            "summary": {
                "original_count": len(records),
                "cleaned_count": len(records),
                "issues_fixed": 0,
                "records_removed": 0
            }
        }
        self._results[task.id] = result
        return result
    
    async def _transform_data(self, task):
        """Transform data successfully."""
        cleaned_data = task.parameters.get("cleaned_data", {})
        records = cleaned_data.get("data", {}).get("records", [])
        print(f"[TRANSFORM] Transforming {len(records)} records")
        
        # Apply transformations
        transformed = []
        for record in records:
            new_record = record.copy()
            
            # Add value category
            try:
                value = int(record.get("value", 0))
                new_record["value_category"] = "high" if value > 150 else "medium" if value > 100 else "low"
            except:
                new_record["value_category"] = "unknown"
            
            # Normalize status
            new_record["status"] = record.get("status", "").upper()
            
            transformed.append(new_record)
        
        result = {
            "data": {"records": transformed, "total": len(transformed)},
            "metrics": {
                "records_transformed": len(transformed),
                "transformations_applied": ["value_categorization", "status_normalization"],
                "success_rate": 1.0
            }
        }
        self._results[task.id] = result
        return result
    
    async def _quality_check(self, task):
        """Check quality."""
        data = task.parameters.get("transformed_data", {})
        records = data.get("data", {}).get("records", [])
        print(f"[QUALITY_CHECK] Checking {len(records)} records")
        
        result = {
            "passed": True,
            "metrics": {
                "completeness": 1.0,
                "accuracy": 1.0,
                "consistency": 1.0,
                "total_records": len(records),
                "quality_score": 1.0
            }
        }
        self._results[task.id] = result
        return result
    
    async def _export_data(self, task):
        """Export successfully."""
        data = task.parameters.get("data", {})
        destination = task.parameters.get("destination", "./output/")
        print(f"[EXPORT] Exporting to {destination}")
        
        os.makedirs(destination, exist_ok=True)
        output_path = os.path.join(destination, "successful_output.json")
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        result = {
            "path": output_path,
            "records_exported": data.get("data", {}).get("total", 0),
            "format": "json",
            "size_bytes": os.path.getsize(output_path)
        }
        self._results[task.id] = result
        return result
    
    async def _generate_report(self, task):
        """Generate success report."""
        print("[REPORT] Generating success report")
        
        report = """# Data Processing Success Report

## Summary
All processing steps completed successfully with no errors.

## Pipeline Steps
1. ✅ Data Ingestion: 3 records loaded
2. ✅ Data Validation: No issues found  
3. ✅ Data Cleaning: No cleaning required
4. ✅ Data Transformation: Added value categories and normalized status
5. ✅ Quality Check: 100% quality score
6. ✅ Data Export: Successfully exported processed data

## Results
- Records processed: 3
- Quality score: 100%
- Processing time: ~2 seconds
- Output format: JSON

## Transformations Applied
- Value categorization (high/medium/low)
- Status normalization (uppercase)

## Quality Metrics
- Completeness: 100%
- Accuracy: 100%  
- Consistency: 100%
"""
        
        result = {
            "report": report,
            "summary": "Successfully processed 3 records with 100% quality",
            "status": "success"
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
            return "10"
        elif "schema" in prompt:
            return "auto_inferred"
        elif "transformations" in prompt:
            return "categorization,normalization"
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


async def test_successful_processing():
    """Test complete successful data processing."""
    print("Testing Successful Data Processing Pipeline")
    print("=" * 50)
    
    # Load pipeline
    with open("pipelines/data_processing.yaml", "r") as f:
        pipeline_yaml = f.read()
    
    # Initialize orchestrator
    control_system = SuccessfulDataControlSystem()
    orchestrator = Orchestrator(control_system=control_system)
    
    # Set up AUTO resolver
    mock_model = MockAutoModel()
    orchestrator.model_registry.register_model(mock_model)
    orchestrator.yaml_compiler.ambiguity_resolver.model = mock_model
    
    try:
        # Execute pipeline
        results = await orchestrator.execute_yaml(
            pipeline_yaml,
            context={
                "data_source": "test_data/small_dataset.csv",
                "processing_mode": "batch",
                "error_tolerance": 0.05
            }
        )
        
        print("\nPipeline execution completed successfully!")
        print("\nTask Results:")
        for task_id, result in results.items():
            if isinstance(result, dict):
                if "summary" in result:
                    print(f"  {task_id}: {result['summary']}")
                elif "status" in result:
                    print(f"  {task_id}: {result['status']}")
                elif "metrics" in result and "records_transformed" in result["metrics"]:
                    print(f"  {task_id}: Transformed {result['metrics']['records_transformed']} records")
                elif "records_exported" in result:
                    print(f"  {task_id}: Exported {result['records_exported']} records to {result['path']}")
                else:
                    print(f"  {task_id}: Completed")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_successful_processing())
    
    if success:
        print("\n✅ Successful data processing pipeline completed!")
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)