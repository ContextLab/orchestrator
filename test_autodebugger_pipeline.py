#!/usr/bin/env python3
"""
Comprehensive AutoDebugger pipeline test demonstrating real-world debugging scenarios.
"""

import asyncio
import json
import sys
import tempfile
import os
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestrator.tools.auto_debugger import AutoDebuggerTool

async def test_comprehensive_debugging_scenarios():
    """Test AutoDebugger on various real-world debugging scenarios."""
    
    debugger = AutoDebuggerTool()
    print("=== AutoDebugger Phase 4 Comprehensive Testing ===\n")
    
    # Scenario 1: Python Syntax Error Debugging
    print("🐍 Scenario 1: Python Syntax Error Debugging")
    python_code_with_errors = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    else  # Missing colon
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

# Function name typo and missing parenthesis
results = [fibonacci_calc(i for i in range(5)]
print(f"Fibonacci results: {results}")
"""
    
    result1 = await debugger._arun(
        task_description="Fix Python syntax errors in Fibonacci calculator",
        content_to_debug=python_code_with_errors,
        error_context="SyntaxError: missing colon after 'else', NameError: 'fibonacci_calc' not defined, SyntaxError: unmatched parenthesis",
        expected_outcome="Working Python code that calculates first 5 Fibonacci numbers"
    )
    
    result1_parsed = json.loads(result1)
    print(f"✅ Success: {result1_parsed['success']}")
    print(f"📊 Iterations: {result1_parsed['total_iterations']}")
    print(f"🎯 Quality Score: {result1_parsed.get('validation', {}).get('quality_score', 'N/A')}")
    if result1_parsed['success']:
        print(f"🔧 Fixed Code Preview: {result1_parsed['final_content'][:100]}...")
    print()
    
    # Scenario 2: JavaScript Runtime Error Debugging  
    print("🟨 Scenario 2: JavaScript Runtime Error Debugging")
    js_code_with_errors = """
const data = {
    users: [
        {name: "John", age: 30},
        {name: "Jane", age: 25},
        {name: "Bob"}  // Missing age property
    ]
};

function calculateAverageAge(userData) {
    let total = 0;
    let count = 0;
    
    // Will fail when age is undefined
    for (let user of userData.users) {
        total += user.age;  // TypeError when age is undefined
        count++;
    }
    
    // Division by zero possible
    return total / count;
}

console.log("Average age:", calculateAverageAge(data));
"""
    
    result2 = await debugger._arun(
        task_description="Fix JavaScript runtime errors in age calculator",
        content_to_debug=js_code_with_errors,
        error_context="TypeError: Cannot read property 'age' of undefined, NaN result from undefined values",
        expected_outcome="Robust JavaScript that handles missing age data gracefully"
    )
    
    result2_parsed = json.loads(result2)
    print(f"✅ Success: {result2_parsed['success']}")
    print(f"📊 Iterations: {result2_parsed['total_iterations']}")
    print(f"🎯 Quality Score: {result2_parsed.get('validation', {}).get('quality_score', 'N/A')}")
    if result2_parsed['success']:
        print(f"🔧 Fixed Code Preview: {result2_parsed['final_content'][:150]}...")
    print()
    
    # Scenario 3: YAML Configuration Error Debugging
    print("📋 Scenario 3: YAML Configuration Debugging")
    yaml_with_errors = """
# Pipeline configuration with errors
name: data-processing-pipeline
version: 1.0

stages:
  - name: extraction
    type: data_extraction
    config:
      source: database
      query: SELECT * FROM users
    timeout: 300
    # Missing dash for next item
    name: transformation
    type: data_transformation
    config:
      operations:
        - clean_nulls
        - normalize_names
        - calculate_metrics
      # Indentation error
    output_format: json
    
  # Invalid boolean value
  - name: validation
    type: data_validation
    enabled: yes_maybe  # Should be true/false
    config:
      rules: [required_fields, data_types]
      
# Missing colon
outputs
  processed_data:
    format: parquet
    location: /data/processed/
"""
    
    result3 = await debugger._arun(
        task_description="Fix YAML configuration syntax and structure errors",
        content_to_debug=yaml_with_errors,
        error_context="YAML parsing errors: missing colons, indentation issues, invalid boolean values, inconsistent list structure",
        expected_outcome="Valid YAML configuration that parses correctly"
    )
    
    result3_parsed = json.loads(result3)
    print(f"✅ Success: {result3_parsed['success']}")
    print(f"📊 Iterations: {result3_parsed['total_iterations']}")
    print(f"🎯 Quality Score: {result3_parsed.get('validation', {}).get('quality_score', 'N/A')}")
    if result3_parsed['success']:
        print(f"🔧 Fixed Code Preview: {result3_parsed['final_content'][:150]}...")
    print()
    
    # Scenario 4: Data Processing Logic Error Debugging
    print("📊 Scenario 4: Data Processing Logic Error Debugging")
    data_processing_code = """
import json

def process_sales_data(sales_records):
    results = {
        "total_sales": 0,
        "average_sale": 0,
        "top_products": [],
        "monthly_breakdown": {}
    }
    
    # Logic errors in data processing
    for record in sales_records:
        # Missing validation - will fail on malformed data
        results["total_sales"] += record["amount"]
        
        # Wrong field name
        product = record["product_name"]  # Should be "product"
        if product not in results["top_products"]:
            results["top_products"].append(product)
    
    # Division by zero risk
    results["average_sale"] = results["total_sales"] / len(sales_records)
    
    # Logic error - not actually calculating monthly breakdown
    results["monthly_breakdown"]["current_month"] = results["total_sales"]
    
    return results

# Test data with intentional issues
test_data = [
    {"product": "Widget A", "amount": 100.50, "date": "2024-01"},
    {"product": "Widget B", "amount": 75.25, "date": "2024-01"},
    {"product": "Widget C"},  # Missing amount
    {"amount": 200.00, "date": "2024-02"}  # Missing product
]

result = process_sales_data(test_data)
print(json.dumps(result, indent=2))
"""
    
    result4 = await debugger._arun(
        task_description="Fix data processing logic errors and handle malformed data",
        content_to_debug=data_processing_code,
        error_context="KeyError: 'amount' missing, KeyError: 'product_name' vs 'product', division by zero risk, incomplete monthly breakdown logic",
        expected_outcome="Robust data processor that handles missing fields and calculates accurate statistics"
    )
    
    result4_parsed = json.loads(result4)
    print(f"✅ Success: {result4_parsed['success']}")
    print(f"📊 Iterations: {result4_parsed['total_iterations']}")
    print(f"🎯 Quality Score: {result4_parsed.get('validation', {}).get('quality_score', 'N/A')}")
    if result4_parsed['success']:
        print(f"🔧 Fixed Code Preview: {result4_parsed['final_content'][:200]}...")
    print()
    
    # Summary
    print("=== AutoDebugger Phase 4 Test Summary ===")
    
    all_results = [result1_parsed, result2_parsed, result3_parsed, result4_parsed]
    successful_tests = sum(1 for r in all_results if r['success'])
    
    print(f"🎯 Success Rate: {successful_tests}/{len(all_results)} ({successful_tests/len(all_results)*100:.1f}%)")
    print(f"📊 Average Iterations: {sum(r['total_iterations'] for r in all_results) / len(all_results):.1f}")
    
    quality_scores = [r.get('validation', {}).get('quality_score', 0) for r in all_results if r['success']]
    if quality_scores:
        print(f"🏆 Average Quality Score: {sum(quality_scores) / len(quality_scores):.2f}")
    
    print(f"\n✨ AutoDebugger Phase 4 demonstrates:")
    print(f"  • Real LLM-powered error analysis and correction")
    print(f"  • Multi-iteration debugging with learning from failures")
    print(f"  • Cross-language debugging (Python, JavaScript, YAML)")
    print(f"  • Complex logic error detection and fixing")
    print(f"  • Comprehensive validation of solutions")
    print(f"  • NO MOCKS - all debugging uses real systems and models")
    
    return successful_tests == len(all_results)

if __name__ == "__main__":
    success = asyncio.run(test_comprehensive_debugging_scenarios())
    sys.exit(0 if success else 1)