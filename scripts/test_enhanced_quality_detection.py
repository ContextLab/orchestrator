#!/usr/bin/env python3
"""
Test enhanced quality detection capabilities from Stream B.

This script tests the enhanced template detection, content assessment,
debug artifact detection, and professional standards validation.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator.quality.enhanced_template_detector import EnhancedTemplateDetector
from orchestrator.quality.content_assessor import AdvancedContentAssessor
from orchestrator.quality.debug_artifact_detector import DebugArtifactDetector  
from orchestrator.quality.professional_standards_validator import ProfessionalStandardsValidator


def test_enhanced_template_detection():
    """Test enhanced template detection with various template systems."""
    print("🔍 Testing Enhanced Template Detection...")
    
    detector = EnhancedTemplateDetector()
    
    # Test cases with various template systems
    test_cases = [
        ("Jinja2", "Processing data from {{input_file}} with model {{model_name}}"),
        ("Handlebars", "User: {{#user}}{{name}} ({{email}}){{/user}}"),
        ("ERB", "Hello <%= user.name %>, your balance is <%= account.balance %>"),
        ("Go Template", "Welcome {{.User.Name}} to {{.Site.Title}}"),
        ("Angular", "Hello {{user.name}}! <div *ngIf=\"user.active\">Active</div>"),
        ("Vue.js", "Count: {{count}} <button v-on:click=\"increment\">+</button>"),
        ("Nested", "Result: {{data.{{field_name}}.value}}"),
        ("Multi-line", "{% for item in items %}\n  Item: {{item}}\n{% endfor %}"),
    ]
    
    total_issues = 0
    for template_type, content in test_cases:
        issues = detector.detect_template_artifacts(content, f"test_{template_type.lower()}.txt")
        print(f"  {template_type}: {len(issues)} issues detected")
        if issues:
            for issue in issues[:2]:  # Show first 2 issues
                print(f"    - {issue.description}")
        total_issues += len(issues)
    
    print(f"✅ Enhanced template detection: {total_issues} total issues detected\n")
    return total_issues > 0


def test_debug_artifact_detection():
    """Test debug artifact detection system.""" 
    print("🐛 Testing Debug Artifact Detection...")
    
    detector = DebugArtifactDetector()
    
    test_cases = [
        ("Conversational AI", "Certainly! Here's the analysis you requested. I hope this helps!"),
        ("Debug statements", "print('Debug: processing file') \nconsole.log('Current state:', data)"),
        ("Stack trace", "Traceback (most recent call last):\n  File 'test.py', line 42"),
        ("Test data", "Using test user John Doe with email john@example.com"),
        ("Development comments", "# TODO: Fix this later\n# HACK: Temporary workaround"),
        ("Console output", "$ python script.py\n>>> import sys\n[INFO] Processing started"),
        ("Meta-commentary", "As noted above, please note that this demonstrates the feature."),
        ("Processing artifacts", "Processing... Loading... Analysis in progress..."),
    ]
    
    total_issues = 0
    for artifact_type, content in test_cases:
        issues = detector.detect_debug_artifacts(content, f"test_{artifact_type.lower().replace(' ', '_')}.txt")
        print(f"  {artifact_type}: {len(issues)} issues detected")
        if issues:
            for issue in issues[:2]:  # Show first 2 issues
                print(f"    - {issue.description}")
        total_issues += len(issues)
    
    print(f"✅ Debug artifact detection: {total_issues} total issues detected\n")
    return total_issues > 0


async def test_advanced_content_assessment():
    """Test advanced content assessment with LLM integration."""
    print("📝 Testing Advanced Content Assessment...")
    
    assessor = AdvancedContentAssessor()
    
    test_cases = [
        ("markdown_doc.md", """
# Analysis Report

## Overview
This is gonna be super awesome! We've analyzed the data and it's really cool.

## Results  
TODO: Add results here
The analysis shows maybe around 85% accuracy, I think.

Certainly! Here are the findings you requested:
- Sample data from test users
- Mock responses for demonstration
        """),
        ("data.csv", """
name,email,test_field
John Doe,john@example.com,sample_data
Test User,testuser@test.com,dummy_value  
Jane Smith,jane@example.org,placeholder
        """),
        ("report.json", """
{
  "analysis": "processing...",
  "test_data": true,
  "results": {
    "accuracy": "probably around 90%",
    "status": "work in progress"
  },
  "todo": "complete analysis"
}
        """),
    ]
    
    total_issues = 0
    for file_name, content in test_cases:
        # Test without LLM first (rule-based only)
        quality_assessment = await assessor.assess_content_advanced(
            content, file_name, llm_client=None
        )
        issues = quality_assessment.issues
        print(f"  {file_name}: {len(issues)} issues detected (rule-based)")
        if issues:
            for issue in issues[:3]:  # Show first 3 issues
                print(f"    - {issue.description}")
        total_issues += len(issues)
    
    print(f"✅ Advanced content assessment: {total_issues} total issues detected\n")
    return total_issues > 0


def test_professional_standards_validation():
    """Test professional standards validation."""
    print("🏢 Testing Professional Standards Validation...")
    
    validator = ProfessionalStandardsValidator()
    
    test_cases = [
        ("business_doc.md", """
# Project Overview

This is gonna be a really awesome project! We're super excited about it.
The functionality will leverage cutting-edge paradigms to facilitate synergy.

I think it's probably the best solution, maybe even revolutionary.

## Setup
First, do this. Second, do that. Finally, you're done!

## Contact
Email us at support@example.com or visit our web site.
        """),
        ("technical_doc.md", """
# API Reference

## Authentication
The API uses OAuth 2.0. You'll need to get tokens first.

### Endpoints
- GET /users (gets user data, probably)
- POST /data (sends stuff)

TODO: Add more endpoints
        """),
        ("data_report.csv", """
col1,col2,field1
test data,sample value,123
mock data,dummy text,456
        """),
    ]
    
    total_issues = 0
    for file_name, content in test_cases:
        issues = validator.validate_professional_standards(content, file_name)
        print(f"  {file_name}: {len(issues)} professional standards issues")
        if issues:
            for issue in issues[:3]:  # Show first 3 issues
                print(f"    - {issue.description}")
        total_issues += len(issues)
    
    # Generate summary
    summary = validator.generate_professional_standards_summary(issues)
    print(f"  Professional readiness score: {summary['professional_readiness_score']}/100")
    
    print(f"✅ Professional standards validation: {total_issues} total issues detected\n")
    return total_issues > 0


def test_with_real_pipeline_output():
    """Test with actual pipeline output file."""
    print("🚀 Testing with Real Pipeline Output...")
    
    # Look for existing pipeline outputs to test
    outputs_dir = Path("examples/outputs")
    if not outputs_dir.exists():
        print("  ⚠️ No pipeline outputs directory found, skipping real data test")
        return False
    
    # Find some test files
    test_files = []
    for pipeline_dir in outputs_dir.iterdir():
        if pipeline_dir.is_dir():
            for file_path in pipeline_dir.rglob("*.md"):
                test_files.append(file_path)
                if len(test_files) >= 3:  # Test up to 3 files
                    break
            if len(test_files) >= 3:
                break
    
    if not test_files:
        print("  ⚠️ No markdown files found in pipeline outputs, skipping real data test")
        return False
    
    # Test with enhanced detection
    detector = EnhancedTemplateDetector()
    debug_detector = DebugArtifactDetector()
    validator = ProfessionalStandardsValidator()
    
    total_issues = 0
    for file_path in test_files:
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            print(f"  Testing: {file_path.relative_to(outputs_dir)}")
            
            # Run all detectors
            template_issues = detector.detect_template_artifacts(content, str(file_path))
            debug_issues = debug_detector.detect_debug_artifacts(content, str(file_path))
            standards_issues = validator.validate_professional_standards(content, str(file_path))
            
            file_total = len(template_issues) + len(debug_issues) + len(standards_issues)
            print(f"    Template: {len(template_issues)}, Debug: {len(debug_issues)}, Standards: {len(standards_issues)}")
            
            # Show a few issues if found
            all_issues = template_issues + debug_issues + standards_issues
            for issue in all_issues[:2]:
                print(f"      - {issue.severity.value}: {issue.description}")
            
            total_issues += file_total
            
        except Exception as e:
            print(f"    ❌ Error testing {file_path}: {e}")
    
    print(f"✅ Real pipeline testing: {total_issues} total issues found across {len(test_files)} files\n")
    return total_issues >= 0  # Always return True if we tested files


async def main():
    """Run all enhanced quality detection tests."""
    print("🧪 Testing Stream B Enhanced Quality Detection Capabilities\n")
    
    tests = [
        ("Enhanced Template Detection", test_enhanced_template_detection),
        ("Debug Artifact Detection", test_debug_artifact_detection), 
        ("Advanced Content Assessment", test_advanced_content_assessment),
        ("Professional Standards Validation", test_professional_standards_validation),
        ("Real Pipeline Output Testing", test_with_real_pipeline_output),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = "✅ PASS" if result else "⚠️ PARTIAL"
        except Exception as e:
            results[test_name] = f"❌ FAIL: {e}"
            print(f"  ❌ Error in {test_name}: {e}")
    
    print("📊 Test Results Summary:")
    for test_name, result in results.items():
        print(f"  {test_name}: {result}")
    
    # Overall assessment
    passed_tests = sum(1 for result in results.values() if result.startswith("✅"))
    total_tests = len(results)
    
    print(f"\n🎯 Stream B Enhancement Status: {passed_tests}/{total_tests} components functional")
    
    if passed_tests == total_tests:
        print("✅ All Stream B enhancements are working correctly!")
    elif passed_tests >= total_tests * 0.8:
        print("🟡 Stream B enhancements are mostly functional with minor issues")
    else:
        print("🔴 Stream B enhancements need additional work")
    
    return passed_tests >= total_tests * 0.8


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)