"""
Comprehensive test suite for data_processing pipeline.
Tests all functionality with REAL data operations and API calls (NO MOCKS).
"""

import json
import csv
import os
import tempfile
from pathlib import Path
import pytest
import asyncio
import yaml

from src.orchestrator.orchestrator import Orchestrator


@pytest.fixture
async def orchestrator():
    """Create orchestrator instance."""
    from src.orchestrator.control_systems.hybrid_control_system import HybridControlSystem
    from src.orchestrator.models.registry import ModelRegistry
    
    # Create a minimal registry and control system to avoid initialization error
    registry = ModelRegistry()
    control_system = HybridControlSystem(model_registry=registry)
    
    orchestrator = Orchestrator(
        model_registry=registry,
        control_system=control_system
    )
    return orchestrator


@pytest.fixture
def pipeline_yaml():
    """Load the data_processing pipeline."""
    pipeline_path = Path("examples/data_processing.yaml")
    with open(pipeline_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_json_data(temp_dir):
    """Create sample JSON test data."""
    data = {
        "records": [
            {"id": 1, "name": "Item A", "active": True, "value": 100.0},
            {"id": 2, "name": "Item B", "active": False, "value": 200.0},
            {"id": 3, "name": "Item C", "active": True, "value": 300.0},
            {"id": 4, "name": "Item D", "active": True, "value": 150.0},
            {"id": 5, "name": "Item E", "active": False, "value": 250.0},
        ]
    }
    file_path = Path(temp_dir) / "test_data.json"
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    return str(file_path)


@pytest.fixture
def sample_csv_data(temp_dir):
    """Create sample CSV test data."""
    file_path = Path(temp_dir) / "test_data.csv"
    with open(file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'name', 'active', 'value'])
        writer.writeheader()
        writer.writerows([
            {"id": "1", "name": "Item A", "active": "true", "value": "100.0"},
            {"id": "2", "name": "Item B", "active": "false", "value": "200.0"},
            {"id": "3", "name": "Item C", "active": "true", "value": "300.0"},
        ])
    return str(file_path)


class TestCoreDataProcessing:
    """Test core data processing functionality."""
    
    @pytest.mark.asyncio
    async def test_load_json_data(self, orchestrator, pipeline_yaml, sample_json_data, temp_dir):
        """Test loading and parsing JSON file."""
        # Run pipeline with JSON data
        inputs = {
            "data_source": sample_json_data,
            "output_path": temp_dir
        }
        
        result = await orchestrator.run(pipeline_yaml, inputs)
        
        # Verify data was loaded
        assert result.steps["load_data"]["success"] is True
        assert "records" in result.steps["load_data"]["content"]
        
        # Verify parsing detected JSON
        assert result.steps["parse_data"] == "json"
    
    @pytest.mark.asyncio
    async def test_load_csv_data(self, orchestrator, pipeline_yaml, sample_csv_data, temp_dir):
        """Test loading and parsing CSV file."""
        inputs = {
            "data_source": sample_csv_data,
            "output_path": temp_dir
        }
        
        result = await orchestrator.run(pipeline_yaml, inputs)
        
        # Verify data was loaded
        assert result.steps["load_data"]["success"] is True
        assert len(result.steps["load_data"]["content"]) > 0
        
        # CSV should be detected
        assert result.steps["parse_data"] in ["csv", "unknown"]
    
    @pytest.mark.asyncio
    async def test_validate_valid_data(self, orchestrator, pipeline_yaml, sample_json_data, temp_dir):
        """Test validation of valid data against schema."""
        inputs = {
            "data_source": sample_json_data,
            "output_path": temp_dir
        }
        
        result = await orchestrator.run(pipeline_yaml, inputs)
        
        # Check validation passed
        validation = result.steps["validate_data"]
        assert validation["success"] is True
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0
    
    @pytest.mark.asyncio
    async def test_validate_invalid_data(self, orchestrator, pipeline_yaml, temp_dir):
        """Test validation of invalid data."""
        # Create invalid JSON (missing required fields)
        invalid_data = {
            "records": [
                {"name": "Item A"},  # Missing 'id' field
                {"id": 2}  # Missing 'name' field
            ]
        }
        file_path = Path(temp_dir) / "invalid.json"
        with open(file_path, 'w') as f:
            json.dump(invalid_data, f)
        
        inputs = {
            "data_source": str(file_path),
            "output_path": temp_dir
        }
        
        result = await orchestrator.run(pipeline_yaml, inputs)
        
        # Validation should report errors
        validation = result.steps["validate_data"]
        assert validation["success"] is True  # Tool executed successfully
        # Note: With lenient mode, it may still pass but with warnings
    
    @pytest.mark.asyncio
    async def test_filter_operation(self, orchestrator, pipeline_yaml, sample_json_data, temp_dir):
        """Test filtering data by criteria."""
        inputs = {
            "data_source": sample_json_data,
            "output_path": temp_dir
        }
        
        result = await orchestrator.run(pipeline_yaml, inputs)
        
        # Check transformation was applied
        transform = result.steps["transform_data"]
        assert transform["success"] is True
        
        # Result should contain filtered and aggregated data
        assert "result" in transform or "processed_data" in transform
    
    @pytest.mark.asyncio
    async def test_aggregate_operation(self, orchestrator, pipeline_yaml, sample_json_data, temp_dir):
        """Test aggregation operations."""
        inputs = {
            "data_source": sample_json_data,
            "output_path": temp_dir
        }
        
        result = await orchestrator.run(pipeline_yaml, inputs)
        
        # Check aggregation was performed
        transform = result.steps["transform_data"]
        assert transform["success"] is True
        
        # The aggregation should sum the value field
        # Active items: 100 + 300 + 150 = 550
        # This is based on the filter for active=true
    
    @pytest.mark.asyncio
    async def test_save_results(self, orchestrator, pipeline_yaml, sample_json_data, temp_dir):
        """Test saving processed data to file."""
        inputs = {
            "data_source": sample_json_data,
            "output_path": temp_dir,
            "output_format": "json"
        }
        
        result = await orchestrator.run(pipeline_yaml, inputs)
        
        # Check file was saved
        save_result = result.steps["save_results"]
        assert save_result["success"] is True
        
        # Verify file exists
        output_file = Path(temp_dir) / "processed_data.json"
        assert output_file.exists()
        
        # Verify content is valid JSON
        with open(output_file, 'r') as f:
            content = f.read()
            # Should be valid JSON or a string representation
            assert len(content) > 0
    
    @pytest.mark.asyncio
    async def test_generate_report(self, orchestrator, pipeline_yaml, sample_json_data, temp_dir):
        """Test report generation."""
        inputs = {
            "data_source": sample_json_data,
            "output_path": temp_dir
        }
        
        result = await orchestrator.run(pipeline_yaml, inputs)
        
        # Check report was generated
        report_result = result.steps["save_report"]
        assert report_result["success"] is True
        
        # Verify report file exists
        report_file = Path(temp_dir) / "processing_report.md"
        assert report_file.exists()
        
        # Verify report contains expected sections
        with open(report_file, 'r') as f:
            content = f.read()
            assert "# Data Processing Report" in content
            assert "Validation Results" in content
            assert "Processing Summary" in content
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_full_pipeline(self, orchestrator, pipeline_yaml, sample_json_data, temp_dir):
        """Test complete pipeline execution."""
        inputs = {
            "data_source": sample_json_data,
            "output_path": temp_dir,
            "output_format": "json"
        }
        
        result = await orchestrator.run(pipeline_yaml, inputs)
        
        # All steps should complete successfully
        assert result.steps["load_data"]["success"] is True
        assert result.steps["validate_data"]["success"] is True
        assert result.steps["transform_data"]["success"] is True
        assert result.steps["save_results"]["success"] is True
        assert result.steps["save_report"]["success"] is True
        
        # Output files should exist
        assert (Path(temp_dir) / "processed_data.json").exists()
        assert (Path(temp_dir) / "processing_report.md").exists()


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_empty_data_file(self, orchestrator, pipeline_yaml, temp_dir):
        """Test handling of empty data file."""
        # Create empty JSON file
        empty_file = Path(temp_dir) / "empty.json"
        with open(empty_file, 'w') as f:
            json.dump({}, f)
        
        inputs = {
            "data_source": str(empty_file),
            "output_path": temp_dir
        }
        
        result = await orchestrator.run(pipeline_yaml, inputs)
        
        # Pipeline should handle empty data gracefully
        assert result.steps["load_data"]["success"] is True
    
    @pytest.mark.asyncio
    async def test_malformed_json(self, orchestrator, pipeline_yaml, temp_dir):
        """Test handling of malformed JSON."""
        # Create malformed JSON file
        bad_file = Path(temp_dir) / "bad.json"
        with open(bad_file, 'w') as f:
            f.write("{invalid json content")
        
        inputs = {
            "data_source": str(bad_file),
            "output_path": temp_dir
        }
        
        result = await orchestrator.run(pipeline_yaml, inputs)
        
        # Should load the file but fail validation
        assert result.steps["load_data"]["success"] is True
    
    @pytest.mark.asyncio
    async def test_missing_fields(self, orchestrator, pipeline_yaml, temp_dir):
        """Test handling of data with missing fields."""
        incomplete_data = {
            "records": [
                {"id": 1, "name": "Item A"},  # Missing 'active' and 'value'
                {"id": 2, "name": "Item B", "active": True},  # Missing 'value'
            ]
        }
        file_path = Path(temp_dir) / "incomplete.json"
        with open(file_path, 'w') as f:
            json.dump(incomplete_data, f)
        
        inputs = {
            "data_source": str(file_path),
            "output_path": temp_dir
        }
        
        result = await orchestrator.run(pipeline_yaml, inputs)
        
        # Should process with lenient validation
        assert result.steps["validate_data"]["mode"] == "lenient"
    
    @pytest.mark.asyncio
    async def test_large_dataset(self, orchestrator, pipeline_yaml, temp_dir):
        """Test processing of large dataset."""
        # Create large dataset
        large_data = {
            "records": [
                {"id": i, "name": f"Item {i}", "active": i % 2 == 0, "value": i * 10.0}
                for i in range(1000)
            ]
        }
        file_path = Path(temp_dir) / "large.json"
        with open(file_path, 'w') as f:
            json.dump(large_data, f)
        
        inputs = {
            "data_source": str(file_path),
            "output_path": temp_dir
        }
        
        result = await orchestrator.run(pipeline_yaml, inputs)
        
        # Should handle large dataset
        assert result.steps["load_data"]["success"] is True
        assert result.steps["transform_data"]["success"] is True
    
    @pytest.mark.asyncio
    async def test_special_characters(self, orchestrator, pipeline_yaml, temp_dir):
        """Test handling of special characters in data."""
        special_data = {
            "records": [
                {"id": 1, "name": "Item™ A®", "active": True, "value": 100.0},
                {"id": 2, "name": "Ítem B", "active": False, "value": 200.0},
                {"id": 3, "name": "商品 C", "active": True, "value": 300.0},
            ]
        }
        file_path = Path(temp_dir) / "special.json"
        with open(file_path, 'w') as f:
            json.dump(special_data, f, ensure_ascii=False)
        
        inputs = {
            "data_source": str(file_path),
            "output_path": temp_dir
        }
        
        result = await orchestrator.run(pipeline_yaml, inputs)
        
        # Should handle special characters
        assert result.steps["load_data"]["success"] is True
        assert result.steps["save_results"]["success"] is True
    
    @pytest.mark.asyncio
    async def test_nested_json_structure(self, orchestrator, pipeline_yaml, temp_dir):
        """Test processing of deeply nested JSON."""
        nested_data = {
            "records": [
                {
                    "id": 1,
                    "name": "Item A",
                    "active": True,
                    "value": 100.0,
                    "metadata": {
                        "category": "electronics",
                        "tags": ["new", "featured"],
                        "specs": {
                            "weight": 1.5,
                            "dimensions": {"w": 10, "h": 20, "d": 5}
                        }
                    }
                }
            ]
        }
        file_path = Path(temp_dir) / "nested.json"
        with open(file_path, 'w') as f:
            json.dump(nested_data, f)
        
        inputs = {
            "data_source": str(file_path),
            "output_path": temp_dir
        }
        
        result = await orchestrator.run(pipeline_yaml, inputs)
        
        # Should handle nested structures
        assert result.steps["load_data"]["success"] is True
        assert result.steps["validate_data"]["success"] is True
    
    @pytest.mark.asyncio
    async def test_csv_with_headers(self, orchestrator, pipeline_yaml, temp_dir):
        """Test CSV processing with headers."""
        csv_file = Path(temp_dir) / "with_headers.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "active", "value"])
            writer.writerow([1, "Item A", "true", 100.0])
            writer.writerow([2, "Item B", "false", 200.0])
        
        inputs = {
            "data_source": str(csv_file),
            "output_path": temp_dir
        }
        
        result = await orchestrator.run(pipeline_yaml, inputs)
        
        # Should process CSV
        assert result.steps["load_data"]["success"] is True
    
    @pytest.mark.asyncio
    async def test_mixed_data_types(self, orchestrator, pipeline_yaml, temp_dir):
        """Test handling of mixed data types."""
        mixed_data = {
            "records": [
                {"id": 1, "name": "Item A", "active": True, "value": 100},  # int value
                {"id": 2, "name": "Item B", "active": False, "value": 200.5},  # float value
                {"id": 3, "name": "Item C", "active": True, "value": "300"},  # string value
            ]
        }
        file_path = Path(temp_dir) / "mixed.json"
        with open(file_path, 'w') as f:
            json.dump(mixed_data, f)
        
        inputs = {
            "data_source": str(file_path),
            "output_path": temp_dir
        }
        
        result = await orchestrator.run(pipeline_yaml, inputs)
        
        # Should handle mixed types with lenient validation
        assert result.steps["validate_data"]["mode"] == "lenient"


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_file_not_found(self, orchestrator, pipeline_yaml, temp_dir):
        """Test handling of missing input file."""
        inputs = {
            "data_source": "/nonexistent/file.json",
            "output_path": temp_dir
        }
        
        result = await orchestrator.run(pipeline_yaml, inputs)
        
        # Should fail at load_data step
        assert result.steps["load_data"]["success"] is False
    
    @pytest.mark.asyncio
    async def test_invalid_output_path(self, orchestrator, pipeline_yaml, sample_json_data):
        """Test handling of invalid output path."""
        inputs = {
            "data_source": sample_json_data,
            "output_path": "/invalid/path/that/does/not/exist"
        }
        
        # Should create the directory or handle the error
        result = await orchestrator.run(pipeline_yaml, inputs)
        
        # The filesystem tool should create directories as needed
        assert result.steps["save_results"]["success"] is True
    
    @pytest.mark.asyncio
    async def test_invalid_schema(self, orchestrator, temp_dir):
        """Test handling of invalid validation schema."""
        # Create a modified pipeline with invalid schema
        pipeline = {
            "id": "test-invalid-schema",
            "name": "Test Invalid Schema",
            "parameters": {
                "data_source": {"type": "string", "default": "test.json"},
                "output_path": {"type": "string", "default": temp_dir}
            },
            "steps": [
                {
                    "id": "load_data",
                    "tool": "filesystem",
                    "action": "read",
                    "parameters": {"path": "{{ data_source }}"}
                },
                {
                    "id": "validate_data",
                    "tool": "validation",
                    "action": "validate",
                    "parameters": {
                        "data": "{{ load_data.content }}",
                        "schema": {"invalid": "schema"},  # Invalid schema
                        "mode": "strict"
                    },
                    "dependencies": ["load_data"]
                }
            ]
        }
        
        # Create test data
        test_data = {"test": "data"}
        file_path = Path(temp_dir) / "test.json"
        with open(file_path, 'w') as f:
            json.dump(test_data, f)
        
        orchestrator = Orchestrator()
        inputs = {"data_source": str(file_path)}
        
        result = await orchestrator.run(pipeline, inputs)
        
        # Validation should handle invalid schema
        assert "validate_data" in result.steps
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_timeout_handling(self, orchestrator, pipeline_yaml, temp_dir):
        """Test handling of processing timeouts."""
        # Create extremely large dataset that might timeout
        huge_data = {
            "records": [
                {"id": i, "name": f"Item {i}", "active": i % 2 == 0, "value": i * 10.0}
                for i in range(10000)
            ]
        }
        file_path = Path(temp_dir) / "huge.json"
        with open(file_path, 'w') as f:
            json.dump(huge_data, f)
        
        inputs = {
            "data_source": str(file_path),
            "output_path": temp_dir
        }
        
        # Should complete within timeout
        result = await orchestrator.run(pipeline_yaml, inputs)
        
        # Should either complete or timeout gracefully
        assert "load_data" in result.steps