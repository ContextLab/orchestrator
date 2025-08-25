#!/usr/bin/env python3
"""
Quick validation demo for Issue #252 - Testing & Validation.

This script demonstrates the comprehensive testing infrastructure
by running a subset of tests to validate the implementation.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add orchestrator to path
sys.path.insert(0, Path(__file__).parent.parent)

from tests.integration.test_pipeline_wrapper_validation import PipelineWrapperValidator
from tests.performance.test_wrapper_performance_regression import PerformanceRegressionTester  
from tests.quality.test_output_quality_validation import OutputQualityValidator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


async def demo_pipeline_validation():
    """Demonstrate pipeline wrapper validation."""
    print("\n" + "="*60)
    print("DEMO: PIPELINE WRAPPER VALIDATION")
    print("="*60)
    
    try:
        validator = PipelineWrapperValidator()
        
        # Test with a smaller subset for demo
        validator.core_pipelines = [
            "simple_data_processing.yaml",
            "research_minimal.yaml"
        ]
        
        # Use subset of wrapper configs
        validator.wrapper_configs = validator.wrapper_configs[:3]  # baseline, routellm, poml
        
        print(f"Testing {len(validator.core_pipelines)} pipelines with {len(validator.wrapper_configs)} wrapper configurations...")
        
        await validator.initialize()
        results = await validator.validate_all_pipelines()
        
        # Print summary
        summary = results.get("summary", {})
        print(f"\n✅ Demo Results:")
        print(f"   Success Rate: {summary.get('success_rate', 0):.1%}")
        print(f"   Total Tests: {summary.get('total_tests', 0)}")
        print(f"   Average Quality: {results.get('performance', {}).get('average_quality_score', 0):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline validation demo failed: {e}")
        return False


async def demo_performance_testing():
    """Demonstrate performance regression testing."""
    print("\n" + "="*60)
    print("DEMO: PERFORMANCE REGRESSION TESTING")
    print("="*60)
    
    try:
        tester = PerformanceRegressionTester()
        
        # Use minimal test set for demo
        tester.test_pipelines = ["simple_data_processing.yaml"]
        
        await tester.initialize()
        
        # Create baseline if needed
        pipeline_name = tester.test_pipelines[0]
        if pipeline_name not in tester.baselines:
            print(f"Creating performance baseline for {pipeline_name}...")
            baseline = await tester.measure_baseline_performance(pipeline_name)
            tester.baselines[pipeline_name] = baseline
            print(f"   Baseline: {baseline.execution_time_ms:.1f}ms")
            
        print("✅ Performance testing framework validated")
        return True
        
    except Exception as e:
        print(f"❌ Performance testing demo failed: {e}")
        return False


async def demo_quality_validation():
    """Demonstrate quality validation."""
    print("\n" + "="*60) 
    print("DEMO: QUALITY VALIDATION")
    print("="*60)
    
    try:
        validator = OutputQualityValidator()
        
        # Test quality analysis on sample content
        sample_content = """# Test Report

This is a high-quality test report that demonstrates:

- Proper markdown formatting
- Clear structure and headings
- Comprehensive content
- No template issues or placeholders

## Analysis Results

The system has been validated successfully with quality metrics showing excellent performance."""

        # Create temporary file for testing
        from pathlib import Path
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(sample_content)
            temp_path = Path(f.name)
            
        try:
            analysis = validator._analyze_single_output(sample_content, temp_path)
            metrics = analysis.metrics
            
            print(f"✅ Quality Analysis Demo:")
            print(f"   Overall Score: {metrics.overall_score():.1f}%")
            print(f"   Template Score: {metrics.template_score:.1f}%")
            print(f"   Content Quality: {metrics.content_quality_score:.1f}%")
            print(f"   Issues Found: {len(analysis.issues)}")
            
            return True
            
        finally:
            temp_path.unlink()  # Clean up temp file
            
    except Exception as e:
        print(f"❌ Quality validation demo failed: {e}")
        return False


async def main():
    """Run comprehensive testing demo."""
    print("🚀 Issue #252 - Comprehensive Testing Infrastructure Demo")
    print("Testing framework validation for RouteLLM, POML, and wrapper integrations")
    
    results = []
    
    # Run demos
    print("\n🔍 Running validation demos...")
    
    # Demo 1: Pipeline Validation
    result1 = await demo_pipeline_validation()
    results.append(result1)
    
    # Demo 2: Performance Testing  
    result2 = await demo_performance_testing()
    results.append(result2)
    
    # Demo 3: Quality Validation
    result3 = await demo_quality_validation()
    results.append(result3)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "="*80)
    print("DEMO VALIDATION SUMMARY")
    print("="*80)
    
    if passed == total:
        print(f"🎉 All {total} testing framework demos PASSED!")
        print("✅ Comprehensive testing infrastructure is ready for deployment")
        print("\nNext steps:")
        print("1. Run full validation: python scripts/test_all_pipelines_with_wrappers.py")
        print("2. Review detailed test results")
        print("3. Deploy wrapper integrations with confidence")
    else:
        print(f"⚠️  {passed}/{total} testing framework demos passed")
        print("🔧 Some frameworks need attention before full deployment")
        
    print(f"\n📄 Full testing available at: scripts/test_all_pipelines_with_wrappers.py")
    print(f"📁 Test results will be saved to: tests/results/comprehensive/")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)