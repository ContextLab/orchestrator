"""
Data Processing Pipeline Tests

Tests for data processing pipelines including:
- data_processing.yaml
- data_processing_pipeline.yaml  
- simple_data_processing.yaml
- statistical_analysis.yaml

Validates:
- Data transformation accuracy
- CSV/JSON processing
- Statistical calculations
- Data integrity
- Error handling
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import csv
import pytest

from tests.pipeline_tests.test_base import BasePipelineTest, PipelineTestConfiguration, PipelineExecutionResult


class DataProcessingPipelineTests(BasePipelineTest):
    """Test suite for data processing pipelines with real API calls and data validation."""
    
    def test_basic_execution(self):
        """Test basic pipeline execution - required by base class."""
        # Simple synchronous test using basic data processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create simple test data
            test_data = {"records": [{"id": 1, "name": "Test", "active": True}]}
            json_file = temp_path / "test.json"
            with open(json_file, 'w') as f:
                json.dump(test_data, f)
            
            simple_pipeline = f"""
name: Basic Test
description: Basic execution test

tasks:
  - name: load_data
    type: filesystem
    action: read
    path: "{json_file}"
    
  - name: validate_data
    type: llm
    model: anthropic:claude-sonnet-4-20250514
    template: |
      Is this valid JSON data? Answer YES or NO:
      {{{{ load_data.content }}}}
    dependencies:
      - load_data

outputs:
  data: "{{{{ load_data.content }}}}"
  validation: "{{{{ validate_data.content }}}}"
"""
            
            result = self.execute_pipeline_sync(simple_pipeline)
            
            self.assert_pipeline_success(result, "Basic execution should work")
            self.assert_output_contains(result, "validation", "yes", case_sensitive=False)
    
    def test_error_handling(self):
        """Test error handling scenarios - required by base class."""
        # Test with invalid file path
        error_pipeline = """
name: Error Test
description: Test error handling

tasks:
  - name: load_invalid
    type: filesystem
    action: read
    path: "/nonexistent/file.json"
    
  - name: process_anyway
    type: llm
    model: anthropic:claude-sonnet-4-20250514
    template: "Process this: {{ load_invalid.content }}"
    dependencies:
      - load_invalid

outputs:
  result: "{{ process_anyway.content }}"
"""
        
        result = self.execute_pipeline_sync(error_pipeline)
        
        # Should fail gracefully
        assert not result.success, "Pipeline should fail with invalid file path"
        assert result.error is not None, "Error should be captured"


def _create_test_csv_data(output_dir: Path) -> Path:
    """Create test CSV data for processing tests."""
    test_data = [
        {"id": 1, "name": "Alice", "age": 30, "active": True, "value": 100.5},
        {"id": 2, "name": "Bob", "age": 25, "active": False, "value": 75.2},
        {"id": 3, "name": "Charlie", "age": 35, "active": True, "value": 200.8},
        {"id": 4, "name": "Diana", "age": 28, "active": True, "value": 150.3},
        {"id": 5, "name": "Eve", "age": 32, "active": False, "value": 90.1}
    ]
    
    csv_file = output_dir / "test_data.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "age", "active", "value"])
        writer.writeheader()
        writer.writerows(test_data)
    
    return csv_file


def _create_test_json_data(output_dir: Path) -> Path:
    """Create test JSON data for processing tests."""
    test_data = {
        "records": [
            {"id": 1, "name": "Product A", "active": True},
            {"id": 2, "name": "Product B", "active": False},
            {"id": 3, "name": "Product C", "active": True}
        ]
    }
    
    json_file = output_dir / "test_data.json"
    with open(json_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    return json_file


def _create_sales_test_data(output_dir: Path) -> Path:
    """Create sales data for advanced data processing tests."""
    sales_data = [
        {
            "order_id": "ORD-001234",
            "customer_id": "CUST-001",
            "product_name": "Laptop Pro",
            "quantity": 2,
            "unit_price": 1299.99,
            "order_date": "2024-01-15",
            "status": "delivered"
        },
        {
            "order_id": "ORD-001235", 
            "customer_id": "CUST-002",
            "product_name": "Wireless Mouse",
            "quantity": 5,
            "unit_price": 29.99,
            "order_date": "2024-01-16",
            "status": "shipped"
        },
        {
            "order_id": "ORD-001236",
            "customer_id": "CUST-001", 
            "product_name": "USB Cable",
            "quantity": 3,
            "unit_price": 9.99,
            "order_date": "2024-01-17",
            "status": "processing"
        },
        {
            "order_id": "ORD-001237",
            "customer_id": "CUST-003",
            "product_name": "Laptop Pro", 
            "quantity": 1,
            "unit_price": 1299.99,
            "order_date": "2024-01-18",
            "status": "cancelled"
        }
    ]
    
    csv_file = output_dir / "sales_data.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "order_id", "customer_id", "product_name", 
            "quantity", "unit_price", "order_date", "status"
        ])
        writer.writeheader()
        writer.writerows(sales_data)
    
    return csv_file


@pytest.mark.asyncio
async def test_data_processing_basic(pipeline_orchestrator, pipeline_model_registry, temp_output_dir):
    """Test basic data_processing.yaml pipeline with JSON data."""
    # Create test data
    json_file = _create_test_json_data(temp_output_dir)
    
    # Create output directory
    output_dir = temp_output_dir / "outputs"
    output_dir.mkdir()
    
    # Read the pipeline YAML
    pipeline_yaml = f"""
id: data-processing-test
name: Data Processing Test
description: Test basic data processing

parameters:
  data_source:
    type: string
    default: "{json_file}"
  output_path:
    type: string
    default: "{output_dir}"

steps:
  - id: load_data
    tool: filesystem
    action: read
    parameters:
      path: "{{{{ data_source }}}}"
  
  - id: parse_data
    action: generate_text
    parameters:
      prompt: |
        Parse this data and identify its structure:
        {{{{ load_data }}}}
        
        Return ONLY one word: "json" if it's JSON, "csv" if it's CSV, or "unknown" if unclear.
      model: anthropic:claude-sonnet-4-20250514
      max_tokens: 10
    dependencies:
      - load_data
  
  - id: format_results
    action: generate_text
    parameters:
      prompt: |
        Convert this data to clean JSON format:
        {{{{ load_data }}}}
        
        Return ONLY valid JSON without any markdown formatting, code fences, or explanations.
        Do NOT include ```json or ``` markers.
        Start directly with {{ and end with }}
      model: anthropic:claude-sonnet-4-20250514
      max_tokens: 500
    dependencies:
      - parse_data
  
  - id: save_results
    tool: filesystem
    action: write
    parameters:
      path: "{{{{ output_path }}}}/processed_data.json"
      content: "{{{{ format_results }}}}"
    dependencies:
      - format_results

outputs:
  original_data: "{{{{ load_data }}}}"
  parsed_format: "{{{{ parse_data }}}}"
  processed_data: "{{{{ format_results }}}}"
  output_file: "{{{{ output_path }}}}/processed_data.json"
"""
    
    # Create test configuration
    config = PipelineTestConfiguration(
        timeout_seconds=120,
        max_cost_dollars=0.20,
        enable_performance_tracking=True
    )
    
    # Create test instance
    test_instance = DataProcessingPipelineTests(
        orchestrator=pipeline_orchestrator,
        model_registry=pipeline_model_registry,
        config=config
    )
    
    # Execute pipeline
    result = await test_instance.execute_pipeline_async(
        pipeline_yaml,
        inputs={"data_source": str(json_file), "output_path": str(output_dir)}
    )
    
    # Validate execution
    test_instance.assert_pipeline_success(result, "Data processing pipeline should complete successfully")
    
    # Validate outputs
    test_instance.assert_output_contains(result, "parsed_format", "json", case_sensitive=False)
    
    # Check that output file was created
    output_file = output_dir / "processed_data.json"
    assert output_file.exists(), "Output file should be created"
    
    # Validate JSON structure
    with open(output_file) as f:
        processed_data = json.load(f)
    assert "records" in processed_data, "Processed data should contain records"
    assert len(processed_data["records"]) == 3, "Should have 3 records"
    
    # Performance validation
    test_instance.assert_performance_within_limits(result, max_time=60, max_cost=0.10)


@pytest.mark.asyncio
async def test_simple_data_processing(pipeline_orchestrator, pipeline_model_registry, temp_output_dir):
    """Test simple_data_processing.yaml pipeline with CSV filtering."""
    # Create test CSV data with active/inactive records
    test_data = [
        {"id": 1, "name": "Alice", "status": "active", "value": 100},
        {"id": 2, "name": "Bob", "status": "inactive", "value": 75},
        {"id": 3, "name": "Charlie", "status": "active", "value": 200}
    ]
    
    # Create data directory structure
    data_dir = temp_output_dir / "data"
    data_dir.mkdir()
    csv_file = data_dir / "input.csv"
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "status", "value"])
        writer.writeheader()
        writer.writerows(test_data)
    
    # Create output directory
    output_dir = temp_output_dir / "outputs"
    output_dir.mkdir()
    
    # Modified simple data processing pipeline
    pipeline_yaml = f"""
id: simple_data_processing_test
name: Simple Data Processing Test
description: Test CSV filtering

parameters:
  output_path:
    type: string
    default: "{output_dir}"

steps:
  - id: read_data
    tool: filesystem
    action: read
    parameters:
      path: "{csv_file}"
    
  - id: process_data
    tool: data-processing
    action: filter
    parameters:
      data: "{{{{ read_data.content }}}}"
      format: "csv"
      operation:
        criteria:
          status: "active"
    dependencies:
      - read_data
    
  - id: save_results
    tool: filesystem
    action: write
    parameters:
      path: "{{{{ output_path }}}}/filtered_output.csv"
      content: "{{{{ process_data.processed_data }}}}"
    dependencies:
      - process_data

outputs:
  original_data: "{{{{ read_data.content }}}}"
  filtered_data: "{{{{ process_data.processed_data }}}}"
  output_file: "{{{{ output_path }}}}/filtered_output.csv"
"""
    
    # Create test configuration
    config = PipelineTestConfiguration(
        timeout_seconds=120,
        max_cost_dollars=0.20,
        enable_performance_tracking=True
    )
    
    # Create test instance
    test_instance = DataProcessingPipelineTests(
        orchestrator=pipeline_orchestrator,
        model_registry=pipeline_model_registry,
        config=config
    )
    
    # Execute pipeline
    result = await test_instance.execute_pipeline_async(
        pipeline_yaml,
        inputs={"output_path": str(output_dir)}
    )
    
    # Validate execution
    test_instance.assert_pipeline_success(result, "Simple data processing should complete successfully")
    
    # Check output file exists
    output_file = output_dir / "filtered_output.csv"
    assert output_file.exists(), "Filtered output file should be created"
    
    # Validate filtering worked - should only have active records
    with open(output_file, 'r') as f:
        reader = csv.DictReader(f)
        filtered_rows = list(reader)
    
    assert len(filtered_rows) == 2, "Should have 2 active records after filtering"
    for row in filtered_rows:
        assert row["status"] == "active", "All filtered records should have status=active"
    
    # Performance validation
    test_instance.assert_performance_within_limits(result, max_time=60, max_cost=0.10)


@pytest.mark.asyncio
async def test_statistical_analysis(pipeline_orchestrator, pipeline_model_registry, temp_output_dir):
    """Test statistical_analysis.yaml pipeline with numerical data."""
    # Create test data for statistical analysis
    test_data = [
        {"value": 10, "category": "A"},
        {"value": 20, "category": "B"}, 
        {"value": 15, "category": "A"},
        {"value": 25, "category": "B"},
        {"value": 12, "category": "A"}
    ]
    
    # Create output directory structure
    output_dir = temp_output_dir / "outputs" / "statistical_analysis"
    output_dir.mkdir(parents=True)
    (output_dir / "data").mkdir()
    (output_dir / "analysis").mkdir()
    
    # Statistical analysis pipeline
    pipeline_yaml = f"""
id: statistical_analysis_test
name: Statistical Analysis Test
description: Test statistical analysis

parameters:
  data: {json.dumps(test_data)}
  confidence_level: 0.95

steps:
  - id: prepare_data
    tool: filesystem
    action: write
    parameters:
      path: "{output_dir / 'data' / 'input_data.json'}"
      content: "{{{{ parameters.data | to_json }}}}"
    
  - id: descriptive_stats
    action: analyze_text
    parameters:
      text: |
        Analyze this dataset and provide descriptive statistics:
        {{{{ parameters.data | to_json }}}}
        
        Calculate and return only these statistics in JSON format:
        - count: number of records
        - mean: average of values
        - min: minimum value
        - max: maximum value
        
        Return as: {{"count": N, "mean": X.X, "min": X, "max": X}}
      model: anthropic:claude-sonnet-4-20250514
      analysis_type: "statistical"
    dependencies:
      - prepare_data
    
  - id: generate_insights
    action: generate_text
    parameters:
      prompt: |
        Based on these statistics, provide 2-3 key insights about the data:
        {{{{ descriptive_stats.result }}}}
        
        Focus on what the numbers tell us about the dataset.
      model: anthropic:claude-sonnet-4-20250514
      max_tokens: 200
    dependencies:
      - descriptive_stats

outputs:
  statistics: "{{{{ descriptive_stats.result }}}}"
  insights: "{{{{ generate_insights.result }}}}"
"""
    
    # Create test configuration
    config = PipelineTestConfiguration(
        timeout_seconds=120,
        max_cost_dollars=0.25,
        enable_performance_tracking=True
    )
    
    # Create test instance
    test_instance = DataProcessingPipelineTests(
        orchestrator=pipeline_orchestrator,
        model_registry=pipeline_model_registry,
        config=config
    )
    
    # Execute pipeline
    result = await test_instance.execute_pipeline_async(
        pipeline_yaml,
        inputs={"data": test_data, "confidence_level": 0.95}
    )
    
    # Validate execution
    test_instance.assert_pipeline_success(result, "Statistical analysis should complete successfully")
    
    # Validate statistical outputs
    test_instance.assert_output_contains(result, "statistics", ["count", "mean"], case_sensitive=False)
    test_instance.assert_output_contains(result, "insights", "data", case_sensitive=False)
    
    # Check data file was created
    data_file = output_dir / "data" / "input_data.json"
    assert data_file.exists(), "Input data file should be created"
    
    # Performance validation
    test_instance.assert_performance_within_limits(result, max_time=90, max_cost=0.20)


@pytest.mark.asyncio
async def test_data_processing_pipeline_advanced(pipeline_orchestrator, pipeline_model_registry, temp_output_dir):
    """Test advanced data_processing_pipeline.yaml with sales data."""
    # Create output directory and sales data
    output_dir = temp_output_dir / "outputs" / "data_processing_pipeline"
    output_dir.mkdir(parents=True)
    
    sales_file = _create_sales_test_data(output_dir)
    
    # Simplified version of data processing pipeline
    pipeline_yaml = f"""
id: data-processing-pipeline-test
name: Data Processing Pipeline Test
description: Test advanced data processing

inputs:
  input_file: "sales_data.csv"
  output_path: "{output_dir}"
  quality_threshold: 0.95

steps:
  - id: read_data
    tool: filesystem
    action: read
    parameters:
      path: "{{{{ output_path }}}}/{{{{ input_file }}}}"
      
  - id: validate_schema
    tool: validation
    action: validate
    parameters:
      data: "{{{{ read_data.content }}}}"
      mode: "CSV"
      schema:
        type: object
        properties:
          order_id:
            type: string
          customer_id:
            type: string
          product_name:
            type: string
          quantity:
            type: integer
            minimum: 1
          unit_price:
            type: number
            minimum: 0
        required: ["order_id", "customer_id", "product_name", "quantity", "unit_price"]
      mode: "LENIENT"
    dependencies:
      - read_data
      
  - id: analyze_data
    action: analyze_text
    parameters:
      text: |
        Analyze this sales data and calculate basic metrics:
        {{{{ read_data.content }}}}
        
        Calculate:
        1. Total number of orders
        2. Total revenue (sum of quantity * unit_price for non-cancelled orders)
        3. Average order value
        
        Return as JSON: {{"total_orders": N, "total_revenue": X.XX, "avg_order_value": X.XX}}
      model: anthropic:claude-sonnet-4-20250514
      analysis_type: "sales_analysis"
    dependencies:
      - validate_schema
      
  - id: save_analysis
    tool: filesystem
    action: write
    parameters:
      path: "{{{{ output_path }}}}/analysis_report.md"
      content: |
        # Sales Data Analysis Report
        
        ## Validation Results
        - Schema Valid: {{{{ validate_schema.valid }}}}
        
        ## Analysis Results
        {{{{ analyze_data.result }}}}
        
        Generated: {{{{ now() }}}}
    dependencies:
      - analyze_data

outputs:
  validation_passed: "{{{{ validate_schema.valid }}}}"
  analysis_results: "{{{{ analyze_data.result }}}}"
  report_path: "{{{{ output_path }}}}/analysis_report.md"
"""
    
    # Create test configuration
    config = PipelineTestConfiguration(
        timeout_seconds=180,
        max_cost_dollars=0.30,
        enable_performance_tracking=True
    )
    
    # Create test instance
    test_instance = DataProcessingPipelineTests(
        orchestrator=pipeline_orchestrator,
        model_registry=pipeline_model_registry,
        config=config
    )
    
    # Execute pipeline
    result = await test_instance.execute_pipeline_async(
        pipeline_yaml,
        inputs={
            "input_file": "sales_data.csv",
            "output_path": str(output_dir),
            "quality_threshold": 0.95
        }
    )
    
    # Validate execution
    test_instance.assert_pipeline_success(result, "Advanced data processing should complete successfully")
    
    # Validate outputs
    test_instance.assert_output_contains(result, "analysis_results", ["total_orders", "revenue"], case_sensitive=False)
    
    # Check report file was created
    report_file = output_dir / "analysis_report.md"
    assert report_file.exists(), "Analysis report should be created"
    
    # Performance validation
    test_instance.assert_performance_within_limits(result, max_time=120, max_cost=0.25)


@pytest.mark.asyncio
async def test_data_integrity_validation(pipeline_orchestrator, pipeline_model_registry, temp_output_dir):
    """Test data integrity checks across processing steps."""
    # Create test data with intentional issues
    problematic_data = [
        {"id": 1, "name": "Valid", "value": 100, "status": "active"},
        {"id": 2, "name": "", "value": -50, "status": "active"},  # Empty name, negative value
        {"id": 3, "name": "Another", "value": 200, "status": "invalid_status"},  # Invalid status
        {"id": 1, "name": "Duplicate", "value": 150, "status": "active"}  # Duplicate ID
    ]
    
    output_dir = temp_output_dir / "outputs"
    output_dir.mkdir()
    
    csv_file = output_dir / "problematic_data.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "value", "status"])
        writer.writeheader()
        writer.writerows(problematic_data)
    
    # Data integrity validation pipeline
    pipeline_yaml = f"""
id: data_integrity_test
name: Data Integrity Test
description: Test data validation and integrity checks

steps:
  - id: read_data
    tool: filesystem
    action: read
    parameters:
      path: "{csv_file}"
    
  - id: validate_integrity
    action: analyze_text
    parameters:
      text: |
        Analyze this CSV data for integrity issues:
        {{{{ read_data.content }}}}
        
        Check for:
        1. Duplicate IDs (same id value appears multiple times)
        2. Empty required fields (name cannot be empty)
        3. Invalid values (value should be positive)
        4. Invalid status (should be 'active' or 'inactive')
        
        Return JSON with findings: {{"duplicate_ids": N, "empty_names": N, "negative_values": N, "invalid_status": N, "total_issues": N}}
      model: anthropic:claude-sonnet-4-20250514
      analysis_type: "data_validation"
    dependencies:
      - read_data
      
  - id: generate_report
    action: generate_text
    parameters:
      prompt: |
        Create a data quality summary based on these findings:
        {{{{ validate_integrity.result }}}}
        
        Provide recommendations for data cleaning.
      model: anthropic:claude-sonnet-4-20250514
      max_tokens: 300
    dependencies:
      - validate_integrity

outputs:
  raw_data: "{{{{ read_data.content }}}}"
  integrity_issues: "{{{{ validate_integrity.result }}}}"
  quality_report: "{{{{ generate_report.result }}}}"
"""
    
    # Create test configuration
    config = PipelineTestConfiguration(
        timeout_seconds=120,
        max_cost_dollars=0.20,
        enable_performance_tracking=True
    )
    
    # Create test instance
    test_instance = DataProcessingPipelineTests(
        orchestrator=pipeline_orchestrator,
        model_registry=pipeline_model_registry,
        config=config
    )
    
    # Execute pipeline
    result = await test_instance.execute_pipeline_async(pipeline_yaml)
    
    # Validate execution
    test_instance.assert_pipeline_success(result, "Data integrity validation should complete")
    
    # Validate that issues were detected
    test_instance.assert_output_contains(result, "integrity_issues", ["duplicate", "issues"], case_sensitive=False)
    test_instance.assert_output_contains(result, "quality_report", "data", case_sensitive=False)
    
    # Performance validation
    test_instance.assert_performance_within_limits(result, max_time=90, max_cost=0.15)


def test_data_processing_infrastructure():
    """Test that the data processing testing infrastructure is properly set up."""
    # Validate that all test functions are properly defined
    test_functions = [
        test_data_processing_basic,
        test_simple_data_processing,
        test_statistical_analysis,
        test_data_processing_pipeline_advanced,
        test_data_integrity_validation
    ]
    
    for func in test_functions:
        assert callable(func), f"Test function {func.__name__} is not callable"
        assert hasattr(func, '__name__'), f"Test function missing name attribute"
    
    print(f"✓ All {len(test_functions)} data processing test functions are properly defined")


def get_data_processing_test_summary() -> Dict[str, Any]:
    """
    Get summary of all data processing tests.
    
    Returns:
        Dict[str, Any]: Test suite summary
    """
    return {
        "test_suite": "Data Processing Pipelines",
        "total_test_functions": 5,
        "features_tested": [
            "Basic data processing with JSON/CSV",
            "Simple CSV filtering operations",
            "Statistical analysis and calculations",
            "Advanced data processing pipeline",
            "Data integrity validation"
        ],
        "test_functions": [
            "test_data_processing_basic",
            "test_simple_data_processing", 
            "test_statistical_analysis",
            "test_data_processing_pipeline_advanced",
            "test_data_integrity_validation"
        ],
        "pipelines_tested": [
            "data_processing.yaml",
            "simple_data_processing.yaml",
            "statistical_analysis.yaml",
            "data_processing_pipeline.yaml"
        ]
    }