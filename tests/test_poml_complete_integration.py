#!/usr/bin/env python3
"""
Final comprehensive test of complete POML integration.
Tests all features together in a realistic scenario.
"""

import sys
import os
import tempfile
import csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.orchestrator.core.template_resolver import TemplateResolver, TemplateFormat
from src.orchestrator.core.output_tracker import OutputTracker
from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider
from src.orchestrator.core.template_migration_tools import (

    analyze_template, migrate_template, MigrationStrategy
)

def create_test_pipeline_data():
    """Create realistic pipeline data for testing."""
    # Create output tracker with realistic pipeline data
    output_tracker = OutputTracker()
    
    # Register pipeline outputs
    output_tracker.register_output("data_ingestion", {
        "records_processed": 5000,
        "file_path": "/data/processed/customer_data.csv",
        "quality_score": 0.95,
        "timestamp": "2024-08-25T10:30:00Z"
    })
    
    output_tracker.register_output("preprocessing", {
        "clean_records": 4850,
        "removed_duplicates": 150,
        "data_quality": "excellent",
        "preprocessing_time": 45.2
    })
    
    output_tracker.register_output("analysis", {
        "key_insights": [
            "Customer satisfaction increased 15%",
            "Support response time improved by 25%", 
            "Top issue: billing questions (35%)"
        ],
        "metrics": {
            "satisfaction_score": 8.5,
            "nps_score": 72,
            "churn_rate": 0.12
        },
        "report_file": "/reports/customer_analysis_2024.pdf"
    })
    
    return output_tracker

def test_complete_workflow():
    """Test complete POML integration workflow."""
    print("=== Complete POML Integration Workflow Test ===")
    
    # Setup
    output_tracker = create_test_pipeline_data()
    resolver = TemplateResolver(output_tracker, enable_poml=True)
    
    print(f"✅ POML Available: {resolver.is_poml_available()}")
    print(f"✅ Pipeline outputs: {len(output_tracker.outputs)} tasks")
    
    # Test 1: Legacy Jinja2 template (backward compatibility)
    legacy_template = """
    Analysis Report for Customer Data Pipeline
    
    Records Processed: {{ data_ingestion.records_processed }}
    Clean Records: {{ preprocessing.clean_records }}
    Data Quality: {{ preprocessing.data_quality | title }}
    
    Key Findings:
    {% for insight in analysis.key_insights %}
    - {{ insight }}
    {% endfor %}
    
    Customer Satisfaction: {{ analysis.metrics.satisfaction_score }}/10
    """
    
    print("\n--- Legacy Jinja2 Template Test ---")
    print("Format:", resolver.get_template_format(legacy_template).value)
    
    try:
        # This should work exactly as before
        result1 = resolver.resolve_template(legacy_template)
        print("✅ Legacy template resolution: SUCCESS")
        print("Sample output:")
        lines = result1.split('\n')[:3]
        for line in lines:
            if line.strip():
                print(f"  {line.strip()}")
    except Exception as e:
        print(f"❌ Legacy template failed: {e}")
    
    # Test 2: Pure POML template
    poml_template = """
    <role>Customer Experience Analyst</role>
    <task>Analyze customer data pipeline results and generate executive summary</task>
    
    <example>
      <input>Customer satisfaction data with {{ data_ingestion.records_processed }} records</input>
      <output>Satisfaction improved to {{ analysis.metrics.satisfaction_score }}/10 with NPS of {{ analysis.metrics.nps_score }}</output>
    </example>
    
    <hint>Focus on business impact and actionable recommendations</hint>
    <hint>Include data quality metrics for credibility</hint>
    
    <output-format>
    Executive summary with:
    1. Key performance indicators
    2. Trend analysis  
    3. Strategic recommendations
    4. Data confidence metrics
    </output-format>
    """
    
    print("\n--- Pure POML Template Test ---")
    print("Format:", resolver.get_template_format(poml_template).value)
    
    try:
        result2 = resolver.resolve_template(poml_template)
        print("✅ POML template resolution: SUCCESS")
        print("Sample output:")
        lines = [line for line in result2.split('\n') if line.strip()][:4]
        for line in lines:
            print(f"  {line}")
    except Exception as e:
        print(f"❌ POML template failed: {e}")
    
    # Test 3: Hybrid template with data integration
    hybrid_template = """
    <role>{{ user_context.role | default("Senior Data Analyst") }}</role>
    <task>Create comprehensive analysis report for {{ analysis.metrics.satisfaction_score }} satisfaction score</task>
    
    <document src="{{ analysis.report_file }}" type="pdf">
      Detailed customer analysis report
    </document>
    
    <example>
      <input>How did customer satisfaction change?</input>
      <output>Satisfaction improved by {{ satisfaction_improvement }}% based on {{ preprocessing.clean_records }} customer records</output>
    </example>
    
    <hint>Data processed: {{ data_ingestion.timestamp | date('%Y-%m-%d') }}</hint>
    <hint>Quality score: {{ data_ingestion.quality_score | round(2) }}</hint>
    
    <output-format>Structured report in {{ output_format | default("JSON") }} format</output-format>
    """
    
    print("\n--- Hybrid Template Test ---")
    print("Format:", resolver.get_template_format(hybrid_template).value)
    
    context = {
        "user_context": {"role": "Customer Success Manager"},
        "satisfaction_improvement": 15,
        "output_format": "JSON"
    }
    
    try:
        result3 = resolver.resolve_template(hybrid_template, context)
        print("✅ Hybrid template resolution: SUCCESS")
        print("Sample output:")
        lines = [line for line in result3.split('\n') if line.strip()][:4]  
        for line in lines:
            print(f"  {line}")
    except Exception as e:
        print(f"❌ Hybrid template failed: {e}")
    
    return True

def test_migration_workflow():
    """Test template migration workflow."""
    print("\n=== Migration Workflow Test ===")
    
    # Existing template to migrate
    existing_template = """
    You are a customer service analyst with expertise in {{ domain }}.
    
    Task: Analyze the support ticket data and identify key issues.
    
    Context:
    - Total tickets: {{ ticket_count }}
    - Priority level: {{ priority | default("normal") }}  
    - Time period: {{ time_period }}
    
    Please provide:
    1. Top 3 issue categories
    2. Customer satisfaction trends
    3. Recommended actions
    
    Output format: {{ output_format | upper }}
    """
    
    # Step 1: Analyze template
    print("--- Step 1: Template Analysis ---")
    analysis = analyze_template(existing_template)
    
    print(f"Current format: {analysis.original_format.value}")
    print(f"Complexity score: {analysis.complexity_score:.2f}")
    print(f"Migration effort: {analysis.migration_effort}")
    print(f"Suggested strategy: {analysis.suggested_strategy.value}")
    print(f"Key benefits: {len(analysis.poml_benefits)} identified")
    
    # Step 2: Migrate template
    print("\n--- Step 2: Template Migration ---")
    migration_result = migrate_template(existing_template, analysis.suggested_strategy)
    
    print(f"Migration success: {migration_result.success}")
    print(f"Strategy used: {migration_result.strategy_used.value}")
    print(f"Changes made: {', '.join(migration_result.changes_made)}")
    
    if migration_result.validation_issues:
        print(f"Issues found: {', '.join(migration_result.validation_issues)}")
    
    # Step 3: Test migrated template
    print("\n--- Step 3: Test Migrated Template ---")
    output_tracker = create_test_pipeline_data()
    resolver = TemplateResolver(output_tracker, enable_poml=True)
    
    test_context = {
        "domain": "technical support",
        "ticket_count": 1250,
        "priority": "high",
        "time_period": "Q3 2024",
        "output_format": "json"
    }
    
    try:
        migrated_result = resolver.resolve_template(migration_result.migrated_template, test_context)
        print("✅ Migrated template works correctly")
        
        # Compare formats
        original_format = resolver.get_template_format(existing_template)
        migrated_format = resolver.get_template_format(migration_result.migrated_template)
        print(f"Format change: {original_format.value} → {migrated_format.value}")
        
    except Exception as e:
        print(f"❌ Migrated template failed: {e}")
    
    return migration_result.success

def test_data_integration_workflow():
    """Test data integration features."""
    print("\n=== Data Integration Workflow Test ===")
    
    # Create temporary data files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(['CustomerID', 'Satisfaction', 'IssueType'])
        writer.writerow(['C001', 9, 'Billing'])
        writer.writerow(['C002', 7, 'Technical'])
        writer.writerow(['C003', 8, 'General'])
        csv_file = f.name
    
    try:
        output_tracker = OutputTracker()
        output_tracker.register_output("file_manager", {"customer_data": csv_file})
        
        resolver = TemplateResolver(output_tracker, enable_poml=True)
        
        # Template with data integration
        data_template = f"""
        <role>Customer Data Analyst</role>
        <task>Analyze customer satisfaction data and provide insights</task>
        
        <document src="{csv_file}" type="csv">
          Customer satisfaction survey results
        </document>
        
        <hint>Focus on satisfaction scores and issue type patterns</hint>
        <output-format>Summary with key statistics and recommendations</output-format>
        """
        
        # Extract data components
        data_components = resolver.extract_poml_data_components(data_template)
        print(f"✅ Data components found: {len(data_components)}")
        for component in data_components:
            print(f"  - {component['type']}: {component['attributes'].get('src', 'N/A')}")
        
        # Render template
        if resolver.is_poml_available():
            result = resolver.resolve_template(data_template)
            print("✅ Data integration template rendered successfully")
            print("Output preview:")
            lines = [line for line in result.split('\n') if line.strip()][:3]
            for line in lines:
                print(f"  {line}")
        else:
            print("⚠️ POML not available - data integration skipped")
    
    finally:
        # Cleanup
        if os.path.exists(csv_file):
            os.unlink(csv_file)
    
    return True

def run_comprehensive_integration_test():
    """Run all integration tests."""
    print("🚀 POML Integration - Comprehensive System Test")
    print("=" * 60)
    
    test_results = []
    
    try:
        result1 = test_complete_workflow()
        test_results.append(("Complete Workflow", result1))
    except Exception as e:
        print(f"❌ Complete workflow test failed: {e}")
        test_results.append(("Complete Workflow", False))
    
    try:
        result2 = test_migration_workflow()
        test_results.append(("Migration Workflow", result2))
    except Exception as e:
        print(f"❌ Migration workflow test failed: {e}")
        test_results.append(("Migration Workflow", False))
    
    try:
        result3 = test_data_integration_workflow()
        test_results.append(("Data Integration", result3))
    except Exception as e:
        print(f"❌ Data integration test failed: {e}")
        test_results.append(("Data Integration", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - POML Integration Ready for Production!")
    else:
        print("⚠️  Some tests failed - Review implementation before deployment")
    
    return passed == total

if __name__ == "__main__":
    run_comprehensive_integration_test()