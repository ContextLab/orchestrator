#!/usr/bin/env python3
"""
Test POML data integration components (documents, tables, images).
"""

import sys
import os
import tempfile
import json
import csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from orchestrator.core.template_resolver import (
    TemplateResolver, 
    TemplateFormat,
    POML_AVAILABLE
)
from orchestrator.core.output_tracker import OutputTracker

def create_test_data_files():
    """Create temporary test files for data integration."""
    files = {}
    
    # Create a CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Age', 'Score'])
        writer.writerow(['Alice', 25, 95])
        writer.writerow(['Bob', 30, 87])
        writer.writerow(['Carol', 28, 92])
        files['csv'] = f.name
    
    # Create a JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        data = {
            "project": "Data Analysis Pipeline",
            "status": "completed",
            "metrics": {
                "records_processed": 1000,
                "accuracy": 95.2,
                "processing_time": 45.3
            }
        }
        json.dump(data, f, indent=2)
        files['json'] = f.name
    
    # Create a text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Sample Analysis Report\n")
        f.write("===================\n\n")
        f.write("This report contains the analysis of customer data.\n")
        f.write("Key findings:\n")
        f.write("- Customer satisfaction: 85%\n")
        f.write("- Average response time: 2.3 hours\n")
        f.write("- Top issue category: Technical Support\n")
        files['txt'] = f.name
    
    return files

def test_poml_document_integration():
    """Test POML document integration with real files."""
    print("=== POML Document Integration Test ===")
    
    if not POML_AVAILABLE:
        print("POML SDK not available, skipping document integration tests")
        return
    
    # Create test files
    test_files = create_test_data_files()
    
    try:
        output_tracker = OutputTracker()
        output_tracker.register_output("data_processor", test_files['csv'])
        output_tracker.register_output("metrics", {"file_path": test_files['json']})
        
        resolver = TemplateResolver(output_tracker, enable_poml=True)
        
        if not resolver.is_poml_available():
            print("POML processor not available")
            return
        
        # Template with document integration
        document_template = f"""
        <role>Data Integration Specialist</role>
        <task>Process multiple data sources and create unified analysis</task>
        
        <document src="{test_files['csv']}" type="csv">
          Customer scores data
        </document>
        
        <document src="{test_files['txt']}" type="txt">
          Analysis report summary
        </document>
        
        <output-format>Provide integrated analysis combining all data sources</output-format>
        """
        
        # Test data component extraction
        data_components = resolver.extract_poml_data_components(document_template)
        print(f"Found {len(data_components)} data components:")
        for i, component in enumerate(data_components):
            print(f"  {i+1}. Type: {component['type']}")
            print(f"     Attributes: {component['attributes']}")
            print()
        
        # Test template resolution
        resolved = resolver.resolve_template(document_template)
        print("Document Integration Template Resolved:")
        print(resolved)
        print()
        
    finally:
        # Cleanup test files
        for file_path in test_files.values():
            if os.path.exists(file_path):
                os.unlink(file_path)

def test_poml_hybrid_data_templates():
    """Test hybrid templates with dynamic data paths."""
    print("=== Hybrid Data Templates Test ===")
    
    if not POML_AVAILABLE:
        print("POML SDK not available, skipping hybrid data tests")
        return
    
    test_files = create_test_data_files()
    
    try:
        output_tracker = OutputTracker()
        output_tracker.register_output("file_locator", {
            "primary_data": test_files['csv'],
            "report_data": test_files['txt'],
            "metrics_file": test_files['json']
        })
        output_tracker.register_output("analysis_config", {
            "focus_area": "customer satisfaction",
            "output_format": "structured_report"
        })
        
        resolver = TemplateResolver(output_tracker, enable_poml=True)
        
        if not resolver.is_poml_available():
            print("POML processor not available")
            return
        
        # Hybrid template with dynamic file paths
        hybrid_template = """
        <role>{{ analysis_config.focus_area | title }} Analyst</role>
        <task>Create comprehensive analysis using available data sources</task>
        
        <document src="{{ file_locator.primary_data }}" type="csv">
          Primary dataset for {{ analysis_config.focus_area }}
        </document>
        
        <document src="{{ file_locator.report_data }}" type="txt">
          Supporting analysis documentation
        </document>
        
        <hint>Focus on {{ analysis_config.focus_area }} trends and patterns</hint>
        <output-format>Generate {{ analysis_config.output_format }} with key insights</output-format>
        """
        
        print("Template format:", resolver.get_template_format(hybrid_template).value)
        
        resolved = resolver.resolve_template(hybrid_template)
        print("Hybrid Data Template Resolved:")
        print(resolved)
        print()
        
    finally:
        # Cleanup
        for file_path in test_files.values():
            if os.path.exists(file_path):
                os.unlink(file_path)

def test_data_component_validation():
    """Test validation of data components in templates."""
    print("=== Data Component Validation Test ===")
    
    output_tracker = OutputTracker()
    resolver = TemplateResolver(output_tracker, enable_poml=True)
    
    # Template with various data components
    complex_template = """
    <role>Multi-modal Data Analyst</role>
    <task>Process multiple data types and create comprehensive report</task>
    
    <document src="/data/sales.csv" type="csv">Sales data</document>
    <table src="{{ results.table_path }}">Dynamic table</table>
    <document src="{{ report.file_path }}" parser="pdf">Report document</document>
    
    <hint>Validate data quality before analysis</hint>
    """
    
    # Extract and validate data components
    components = resolver.extract_poml_data_components(complex_template)
    print(f"Found {len(components)} data components:")
    
    for i, component in enumerate(components):
        print(f"  {i+1}. {component['type'].upper()}")
        print(f"     Attributes: {component['attributes']}")
        
        # Check if paths are template variables or literal paths
        if 'src' in component['attributes']:
            src_path = component['attributes']['src']
            has_variables = '{{' in src_path and '}}' in src_path
            print(f"     Path type: {'Template variable' if has_variables else 'Literal path'}")
            print(f"     Path: {src_path}")
        print()
    
    # Test template validation
    validation_issues = resolver.validate_poml_template(complex_template)
    print("Validation issues:", validation_issues if validation_issues else "None")
    print()

def test_template_migration_helpers():
    """Test utilities for migrating from Jinja2 to POML."""
    print("=== Template Migration Helpers Test ===")
    
    output_tracker = OutputTracker()
    resolver = TemplateResolver(output_tracker, enable_poml=True)
    
    # Create POML template programmatically from components
    migrated_template = resolver.create_poml_template_from_components(
        role="{{ user_type | default('Data Analyst') }}",
        task="Process {{ data_type }} data and generate {{ output_type }} report",
        examples=[
            {
                "input": "Analyze customer survey responses",
                "output": "Customer satisfaction: 85%, Top issues: {{ top_issues }}"
            },
            {
                "input": "Process sales data for Q{{ quarter }}",
                "output": "Revenue: {{ revenue }}, Growth: {{ growth_rate }}%"
            }
        ],
        hints=[
            "Focus on {{ analysis_focus }} metrics",
            "Include confidence intervals for statistical measures"
        ],
        output_format="Generate {{ output_format }} with structured insights"
    )
    
    print("Migrated POML template:")
    print(migrated_template)
    print()
    
    print("Detected format:", resolver.get_template_format(migrated_template).value)
    
    # This template contains Jinja2 variables, so it should be detected as hybrid
    validation_issues = resolver.validate_poml_template(migrated_template)
    print("Validation issues:", validation_issues if validation_issues else "None")

def run_data_integration_tests():
    """Run all data integration tests."""
    print("Testing POML Data Integration Features")
    print("=" * 60)
    print()
    
    test_poml_document_integration()
    test_poml_hybrid_data_templates()
    test_data_component_validation()
    test_template_migration_helpers()
    
    print("=" * 60)
    print("POML Data Integration Tests Complete")

if __name__ == "__main__":
    run_data_integration_tests()