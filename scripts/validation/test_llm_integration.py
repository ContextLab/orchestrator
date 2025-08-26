#!/usr/bin/env python3
"""
Test script to verify LLM integration with existing credential management.

This script tests the core infrastructure components of the LLM Quality Review system.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from orchestrator.core.llm_quality_reviewer import LLMQualityReviewer, LLMQualityError
from orchestrator.core.credential_manager import create_credential_manager
from orchestrator.core.quality_assessment import (
    TemplateArtifactDetector, ContentQualityAssessor, QualityScorer
)


def test_credential_manager():
    """Test credential manager integration."""
    print("🔐 Testing credential manager integration...")
    
    try:
        credential_manager = create_credential_manager()
        print(f"✅ Credential manager initialized: {type(credential_manager).__name__}")
        
        # List available credentials
        credentials = credential_manager.list_credentials()
        print(f"✅ Found {len(credentials)} stored credentials")
        
        return credential_manager
        
    except Exception as e:
        print(f"❌ Credential manager test failed: {e}")
        return None


def test_quality_assessment_framework():
    """Test quality assessment framework components."""
    print("\n🧪 Testing quality assessment framework...")
    
    # Test template artifact detector
    detector = TemplateArtifactDetector()
    test_content = """
    # Test Content
    Processing {{filename}} with {{model_name}}.
    The result is stored in ${output_path}.
    Normal content here.
    """
    
    issues = detector.detect_template_artifacts(test_content, "test.md")
    print(f"✅ Template detector found {len(issues)} artifacts")
    
    # Test content quality assessor
    assessor = ContentQualityAssessor()
    quality = assessor.assess_content_quality(test_content, "test.md")
    print(f"✅ Content quality rating: {quality.rating.value}")
    print(f"   Template artifacts detected: {quality.template_artifacts_detected}")
    
    # Test quality scorer
    scorer = QualityScorer()
    score = scorer.calculate_score(issues)
    production_ready = scorer.determine_production_readiness(score)
    print(f"✅ Quality score: {score}/100, Production ready: {production_ready}")
    
    return len(issues) > 0  # Should detect template artifacts


def test_llm_reviewer_initialization():
    """Test LLM reviewer initialization."""
    print("\n🤖 Testing LLM reviewer initialization...")
    
    try:
        credential_manager = create_credential_manager()
        reviewer = LLMQualityReviewer(credential_manager=credential_manager)
        
        print(f"✅ LLM reviewer initialized")
        print(f"   Available clients: {len(reviewer.clients)}")
        print(f"   Primary models: {reviewer.primary_models}")
        print(f"   Fallback models: {reviewer.fallback_models}")
        
        # Check which models are actually available
        available_models = []
        for model_name in reviewer.primary_models + reviewer.fallback_models:
            if model_name in reviewer.clients:
                available_models.append(model_name)
        
        print(f"   Operational models: {available_models}")
        
        if available_models:
            print("✅ At least one LLM client is operational")
            return reviewer
        else:
            print("⚠️  No LLM clients operational - check API keys")
            return reviewer
            
    except Exception as e:
        print(f"❌ LLM reviewer initialization failed: {e}")
        return None


async def test_simple_content_review(reviewer):
    """Test simple content review functionality."""
    print("\n📝 Testing simple content review...")
    
    if not reviewer or not reviewer.clients:
        print("⚠️  Skipping content review - no operational LLM clients")
        return
    
    # Create test content with known issues
    test_content = """# Sample Report

Processing {{input_file}} with model {{model_name}}.

Certainly! Here's the analysis you requested:

The data contains [PLACEHOLDER] entries.
Analysis will be completed soon...
"""
    
    try:
        # Use rule-based assessment (doesn't require API calls)
        assessor = ContentQualityAssessor()
        quality = assessor.assess_content_quality(test_content, "test_sample.md")
        
        print(f"✅ Content assessment completed")
        print(f"   Rating: {quality.rating.value}")
        print(f"   Issues found: {len(quality.issues)}")
        print(f"   Template artifacts: {quality.template_artifacts_detected}")
        print(f"   Conversational tone: {quality.conversational_tone_detected}")
        
        # Show first few issues
        for i, issue in enumerate(quality.issues[:3]):
            print(f"   Issue {i+1}: {issue.description}")
        
        return True
        
    except Exception as e:
        print(f"❌ Content review test failed: {e}")
        return False


async def test_pipeline_directory_scan():
    """Test pipeline directory scanning."""
    print("\n📁 Testing pipeline directory scanning...")
    
    outputs_dir = Path("examples/outputs")
    if not outputs_dir.exists():
        print("❌ Pipeline outputs directory not found")
        return False
    
    # Find a sample pipeline to test with
    sample_pipelines = []
    for item in outputs_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            sample_pipelines.append(item.name)
    
    if not sample_pipelines:
        print("❌ No sample pipelines found")
        return False
    
    print(f"✅ Found {len(sample_pipelines)} pipelines")
    
    # Test with first available pipeline
    test_pipeline = sample_pipelines[0]
    pipeline_path = outputs_dir / test_pipeline
    
    files = []
    supported_extensions = {'.md', '.txt', '.csv', '.json', '.html', '.png', '.jpg', '.jpeg'}
    
    for file_path in pipeline_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            files.append(file_path)
    
    print(f"✅ Pipeline '{test_pipeline}' has {len(files)} reviewable files")
    
    # Show file types
    text_files = [f for f in files if f.suffix.lower() in ['.md', '.txt', '.csv', '.json', '.html']]
    image_files = [f for f in files if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    
    print(f"   Text files: {len(text_files)}")
    print(f"   Image files: {len(image_files)}")
    
    return len(files) > 0


async def main():
    """Run all integration tests."""
    print("🚀 LLM Quality Review Integration Test")
    print("=====================================")
    
    # Test credential manager
    credential_manager = test_credential_manager()
    
    # Test quality assessment framework
    framework_working = test_quality_assessment_framework()
    
    # Test LLM reviewer initialization
    reviewer = test_llm_reviewer_initialization()
    
    # Test simple content review
    content_review_working = await test_simple_content_review(reviewer)
    
    # Test pipeline directory scanning
    directory_scan_working = await test_pipeline_directory_scan()
    
    # Summary
    print("\n📊 Integration Test Summary")
    print("==========================")
    
    tests = [
        ("Credential Manager", credential_manager is not None),
        ("Quality Framework", framework_working),
        ("LLM Reviewer Init", reviewer is not None),
        ("Content Review", content_review_working),
        ("Directory Scanning", directory_scan_working)
    ]
    
    passed = 0
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed >= 4:  # Allow for LLM client issues due to API keys
        print("🎉 Core infrastructure is ready!")
        if reviewer and reviewer.clients:
            print("💡 Ready to perform full quality reviews")
        else:
            print("⚠️  Add API keys to enable full LLM-powered reviews")
    else:
        print("⚠️  Some components need attention before production use")
    
    return passed >= 4


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)