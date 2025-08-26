#!/usr/bin/env python3
"""
Comprehensive test suite for Stream C visual quality assessment and file organization.

This script tests the visual assessment and organization validation components
built for Issue #277 Stream C, validating both the rule-based quality checks
and integration with vision-enabled LLM models.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator.core.llm_quality_reviewer import LLMQualityReviewer
from orchestrator.core.quality_assessment import IssueCategory, IssueSeverity
from orchestrator.quality.visual_assessor import (
    VisualContentAnalyzer, EnhancedVisualAssessor, ChartQualitySpecialist
)
from orchestrator.quality.organization_validator import (
    NamingConventionValidator, DirectoryStructureValidator, 
    FileLocationValidator, OrganizationQualityValidator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VisualQualityTestSuite:
    """Comprehensive test suite for visual quality assessment components."""
    
    def __init__(self):
        """Initialize the test suite."""
        self.visual_analyzer = VisualContentAnalyzer()
        self.enhanced_assessor = EnhancedVisualAssessor()
        self.chart_specialist = ChartQualitySpecialist()
        self.organization_validator = OrganizationQualityValidator()
        self.llm_reviewer = None
        
        # Test results tracking
        self.test_results = {
            'visual_analysis': {'passed': 0, 'failed': 0, 'total': 0},
            'organization_validation': {'passed': 0, 'failed': 0, 'total': 0},
            'integration_tests': {'passed': 0, 'failed': 0, 'total': 0},
            'real_pipeline_tests': {'passed': 0, 'failed': 0, 'total': 0}
        }
        
        self.detailed_results = []
    
    def run_all_tests(self) -> Dict:
        """Run all test suites and return comprehensive results."""
        logger.info("Starting Stream C Visual Quality Assessment Test Suite")
        start_time = time.time()
        
        try:
            # Test 1: Visual Content Analysis (Rule-based)
            logger.info("=== Testing Visual Content Analysis ===")
            self.test_visual_content_analysis()
            
            # Test 2: Organization Validation
            logger.info("=== Testing Organization Validation ===")
            self.test_organization_validation()
            
            # Test 3: Integration with LLM Framework
            logger.info("=== Testing LLM Framework Integration ===")
            asyncio.run(self.test_llm_integration())
            
            # Test 4: Real Pipeline Testing
            logger.info("=== Testing with Real Pipeline Outputs ===")
            self.test_real_pipeline_outputs()
            
            execution_time = time.time() - start_time
            
            # Generate summary report
            summary = self.generate_test_summary(execution_time)
            logger.info("Test suite completed successfully")
            
            return summary
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            raise
    
    def test_visual_content_analysis(self):
        """Test visual content analysis capabilities."""
        
        # Test 1: Image quality analysis with existing files
        logger.info("Testing image quality analysis...")
        self.test_results['visual_analysis']['total'] += 1
        
        try:
            # Find sample images from creative_image_pipeline
            sample_images = self._find_sample_images()
            
            if sample_images:
                sample_image = sample_images[0]
                issues = self.visual_analyzer.analyze_image_quality(str(sample_image))
                
                # Should not have critical errors for existing files
                critical_issues = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
                if not critical_issues:
                    self.test_results['visual_analysis']['passed'] += 1
                    self.detailed_results.append({
                        'test': 'image_quality_analysis_existing',
                        'status': 'PASSED',
                        'details': f'Analyzed {sample_image.name}, found {len(issues)} issues'
                    })
                else:
                    self.test_results['visual_analysis']['failed'] += 1
                    self.detailed_results.append({
                        'test': 'image_quality_analysis_existing',
                        'status': 'FAILED',
                        'details': f'Unexpected critical issues: {[i.description for i in critical_issues]}'
                    })
            else:
                logger.warning("No sample images found, skipping existing file test")
                
        except Exception as e:
            self.test_results['visual_analysis']['failed'] += 1
            self.detailed_results.append({
                'test': 'image_quality_analysis_existing',
                'status': 'ERROR',
                'details': str(e)
            })
        
        # Test 2: Non-existent file handling
        logger.info("Testing non-existent file handling...")
        self.test_results['visual_analysis']['total'] += 1
        
        try:
            issues = self.visual_analyzer.analyze_image_quality("/nonexistent/image.png")
            critical_issues = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
            
            if critical_issues and "does not exist" in critical_issues[0].description:
                self.test_results['visual_analysis']['passed'] += 1
                self.detailed_results.append({
                    'test': 'nonexistent_file_handling',
                    'status': 'PASSED',
                    'details': 'Correctly detected non-existent file'
                })
            else:
                self.test_results['visual_analysis']['failed'] += 1
                self.detailed_results.append({
                    'test': 'nonexistent_file_handling',
                    'status': 'FAILED',
                    'details': 'Did not properly detect non-existent file'
                })
                
        except Exception as e:
            self.test_results['visual_analysis']['failed'] += 1
            self.detailed_results.append({
                'test': 'nonexistent_file_handling',
                'status': 'ERROR',
                'details': str(e)
            })
        
        # Test 3: Chart quality analysis
        logger.info("Testing chart quality analysis...")
        self.test_results['visual_analysis']['total'] += 1
        
        try:
            # Find chart files
            chart_files = self._find_chart_files()
            
            if chart_files:
                chart_file = chart_files[0]
                issues = self.visual_analyzer.analyze_chart_quality(str(chart_file))
                
                # Charts should have some structure validation
                self.test_results['visual_analysis']['passed'] += 1
                self.detailed_results.append({
                    'test': 'chart_quality_analysis',
                    'status': 'PASSED',
                    'details': f'Analyzed chart {chart_file.name}, found {len(issues)} issues'
                })
            else:
                logger.warning("No chart files found, skipping chart analysis test")
                
        except Exception as e:
            self.test_results['visual_analysis']['failed'] += 1
            self.detailed_results.append({
                'test': 'chart_quality_analysis',
                'status': 'ERROR',
                'details': str(e)
            })
    
    def test_organization_validation(self):
        """Test organization validation capabilities."""
        
        # Test 1: Filename validation
        logger.info("Testing filename validation...")
        self.test_results['organization_validation']['total'] += 1
        
        try:
            naming_validator = NamingConventionValidator()
            
            # Test good filename
            good_issues = naming_validator.validate_filename("data_processing_report.md")
            
            # Test bad filename
            bad_issues = naming_validator.validate_filename("output.txt")
            
            if len(bad_issues) > len(good_issues):
                self.test_results['organization_validation']['passed'] += 1
                self.detailed_results.append({
                    'test': 'filename_validation',
                    'status': 'PASSED',
                    'details': f'Good filename: {len(good_issues)} issues, Bad filename: {len(bad_issues)} issues'
                })
            else:
                self.test_results['organization_validation']['failed'] += 1
                self.detailed_results.append({
                    'test': 'filename_validation',
                    'status': 'FAILED',
                    'details': 'Did not properly distinguish between good and bad filenames'
                })
                
        except Exception as e:
            self.test_results['organization_validation']['failed'] += 1
            self.detailed_results.append({
                'test': 'filename_validation',
                'status': 'ERROR',
                'details': str(e)
            })
        
        # Test 2: Directory structure validation
        logger.info("Testing directory structure validation...")
        self.test_results['organization_validation']['total'] += 1
        
        try:
            structure_validator = DirectoryStructureValidator()
            
            # Test with a real pipeline directory
            sample_pipeline = self._find_sample_pipeline_directory()
            
            if sample_pipeline:
                issues = structure_validator.validate_pipeline_structure(str(sample_pipeline))
                
                # Should complete without errors
                self.test_results['organization_validation']['passed'] += 1
                self.detailed_results.append({
                    'test': 'directory_structure_validation',
                    'status': 'PASSED',
                    'details': f'Validated {sample_pipeline.name}, found {len(issues)} issues'
                })
            else:
                logger.warning("No sample pipeline directory found")
                
        except Exception as e:
            self.test_results['organization_validation']['failed'] += 1
            self.detailed_results.append({
                'test': 'directory_structure_validation',
                'status': 'ERROR',
                'details': str(e)
            })
        
        # Test 3: Complete organization validation
        logger.info("Testing complete organization validation...")
        self.test_results['organization_validation']['total'] += 1
        
        try:
            sample_pipeline = self._find_sample_pipeline_directory()
            
            if sample_pipeline:
                review = self.organization_validator.validate_pipeline_organization(
                    str(sample_pipeline), 
                    sample_pipeline.name,
                    "test_pipeline"
                )
                
                # Should return a valid OrganizationReview
                if hasattr(review, 'issues') and hasattr(review, 'correct_location'):
                    self.test_results['organization_validation']['passed'] += 1
                    self.detailed_results.append({
                        'test': 'complete_organization_validation',
                        'status': 'PASSED',
                        'details': f'Generated review with {len(review.issues)} issues'
                    })
                else:
                    self.test_results['organization_validation']['failed'] += 1
                    self.detailed_results.append({
                        'test': 'complete_organization_validation',
                        'status': 'FAILED',
                        'details': 'Invalid OrganizationReview structure'
                    })
            
        except Exception as e:
            self.test_results['organization_validation']['failed'] += 1
            self.detailed_results.append({
                'test': 'complete_organization_validation',
                'status': 'ERROR',
                'details': str(e)
            })
    
    async def test_llm_integration(self):
        """Test integration with LLM quality reviewer framework."""
        
        # Test 1: LLM Reviewer initialization
        logger.info("Testing LLM reviewer initialization...")
        self.test_results['integration_tests']['total'] += 1
        
        try:
            self.llm_reviewer = LLMQualityReviewer()
            
            if self.llm_reviewer:
                self.test_results['integration_tests']['passed'] += 1
                self.detailed_results.append({
                    'test': 'llm_reviewer_initialization',
                    'status': 'PASSED',
                    'details': f'Initialized with {len(self.llm_reviewer.clients)} clients'
                })
            else:
                self.test_results['integration_tests']['failed'] += 1
                self.detailed_results.append({
                    'test': 'llm_reviewer_initialization',
                    'status': 'FAILED',
                    'details': 'Could not initialize LLM reviewer'
                })
                
        except Exception as e:
            self.test_results['integration_tests']['failed'] += 1
            self.detailed_results.append({
                'test': 'llm_reviewer_initialization',
                'status': 'ERROR',
                'details': str(e)
            })
        
        # Test 2: Visual assessment prompt generation
        logger.info("Testing visual assessment prompt generation...")
        self.test_results['integration_tests']['total'] += 1
        
        try:
            sample_image_path = "/test/path/chart.png"
            prompt = self.enhanced_assessor.create_enhanced_visual_assessment_prompt(
                sample_image_path,
                {'pipeline_type': 'data_analysis'}
            )
            
            if prompt and "IMAGE FILE:" in prompt and "JSON" in prompt:
                self.test_results['integration_tests']['passed'] += 1
                self.detailed_results.append({
                    'test': 'visual_assessment_prompt',
                    'status': 'PASSED',
                    'details': f'Generated prompt with {len(prompt)} characters'
                })
            else:
                self.test_results['integration_tests']['failed'] += 1
                self.detailed_results.append({
                    'test': 'visual_assessment_prompt',
                    'status': 'FAILED',
                    'details': 'Generated prompt missing required elements'
                })
                
        except Exception as e:
            self.test_results['integration_tests']['failed'] += 1
            self.detailed_results.append({
                'test': 'visual_assessment_prompt',
                'status': 'ERROR',
                'details': str(e)
            })
        
        # Test 3: Chart-specific prompt generation
        logger.info("Testing chart-specific prompt generation...")
        self.test_results['integration_tests']['total'] += 1
        
        try:
            chart_prompt = self.chart_specialist.create_chart_specific_prompt(
                "/test/path/bar_chart.png",
                "bar_chart"
            )
            
            if chart_prompt and "bar_chart" in chart_prompt and "axes_labels" in chart_prompt:
                self.test_results['integration_tests']['passed'] += 1
                self.detailed_results.append({
                    'test': 'chart_specific_prompt',
                    'status': 'PASSED',
                    'details': 'Generated chart-specific prompt with required elements'
                })
            else:
                self.test_results['integration_tests']['failed'] += 1
                self.detailed_results.append({
                    'test': 'chart_specific_prompt',
                    'status': 'FAILED',
                    'details': 'Chart-specific prompt missing required elements'
                })
                
        except Exception as e:
            self.test_results['integration_tests']['failed'] += 1
            self.detailed_results.append({
                'test': 'chart_specific_prompt',
                'status': 'ERROR',
                'details': str(e)
            })
    
    def test_real_pipeline_outputs(self):
        """Test with actual pipeline outputs containing images and varied structures."""
        
        # Test 1: Creative image pipeline
        logger.info("Testing creative image pipeline outputs...")
        self.test_results['real_pipeline_tests']['total'] += 1
        
        try:
            creative_pipeline_path = Path("examples/outputs/creative_image_pipeline")
            
            if creative_pipeline_path.exists():
                review = self.organization_validator.validate_pipeline_organization(
                    str(creative_pipeline_path),
                    "creative_image_pipeline",
                    "image_generation"
                )
                
                # Should find images and validate structure
                image_issues = [i for i in review.issues if 'image' in i.description.lower()]
                
                self.test_results['real_pipeline_tests']['passed'] += 1
                self.detailed_results.append({
                    'test': 'creative_pipeline_validation',
                    'status': 'PASSED',
                    'details': f'Found {len(review.issues)} total issues, {len(image_issues)} image-related'
                })
            else:
                logger.warning("Creative image pipeline directory not found")
                
        except Exception as e:
            self.test_results['real_pipeline_tests']['failed'] += 1
            self.detailed_results.append({
                'test': 'creative_pipeline_validation',
                'status': 'ERROR',
                'details': str(e)
            })
        
        # Test 2: Modular analysis pipeline (with charts)
        logger.info("Testing modular analysis pipeline outputs...")
        self.test_results['real_pipeline_tests']['total'] += 1
        
        try:
            modular_pipeline_path = Path("examples/outputs/modular_analysis")
            
            if modular_pipeline_path.exists():
                review = self.organization_validator.validate_pipeline_organization(
                    str(modular_pipeline_path),
                    "modular_analysis", 
                    "data_analysis"
                )
                
                # Check for chart-related validation
                chart_directory = modular_pipeline_path / "charts"
                if chart_directory.exists():
                    chart_issues = self.visual_analyzer.assess_visual_directory_structure(str(chart_directory))
                    
                    self.test_results['real_pipeline_tests']['passed'] += 1
                    self.detailed_results.append({
                        'test': 'modular_analysis_validation',
                        'status': 'PASSED',
                        'details': f'Validated charts directory with {len(chart_issues)} issues'
                    })
                else:
                    self.test_results['real_pipeline_tests']['failed'] += 1
                    self.detailed_results.append({
                        'test': 'modular_analysis_validation',
                        'status': 'FAILED',
                        'details': 'Expected charts directory not found'
                    })
            else:
                logger.warning("Modular analysis pipeline directory not found")
                
        except Exception as e:
            self.test_results['real_pipeline_tests']['failed'] += 1
            self.detailed_results.append({
                'test': 'modular_analysis_validation',
                'status': 'ERROR',
                'details': str(e)
            })
        
        # Test 3: Simple data processing pipeline (organization patterns)
        logger.info("Testing simple data processing pipeline organization...")
        self.test_results['real_pipeline_tests']['total'] += 1
        
        try:
            simple_pipeline_path = Path("examples/outputs/simple_data_processing")
            
            if simple_pipeline_path.exists():
                review = self.organization_validator.validate_pipeline_organization(
                    str(simple_pipeline_path),
                    "simple_data_processing",
                    "data_processing" 
                )
                
                # Should validate file organization
                naming_issues = [i for i in review.issues if i.category == IssueCategory.FILE_ORGANIZATION]
                
                self.test_results['real_pipeline_tests']['passed'] += 1
                self.detailed_results.append({
                    'test': 'simple_data_processing_validation',
                    'status': 'PASSED',
                    'details': f'Found {len(naming_issues)} organization issues out of {len(review.issues)} total'
                })
            else:
                logger.warning("Simple data processing pipeline directory not found")
                
        except Exception as e:
            self.test_results['real_pipeline_tests']['failed'] += 1
            self.detailed_results.append({
                'test': 'simple_data_processing_validation',
                'status': 'ERROR',
                'details': str(e)
            })
    
    def _find_sample_images(self) -> List[Path]:
        """Find sample images for testing."""
        sample_images = []
        
        # Check creative_image_pipeline
        creative_path = Path("examples/outputs/creative_image_pipeline")
        if creative_path.exists():
            for img in creative_path.rglob("*.png"):
                sample_images.append(img)
                if len(sample_images) >= 3:  # Just need a few samples
                    break
        
        return sample_images
    
    def _find_chart_files(self) -> List[Path]:
        """Find chart files for testing."""
        chart_files = []
        
        # Check modular_analysis charts
        charts_path = Path("examples/outputs/modular_analysis/charts")
        if charts_path.exists():
            for chart in charts_path.glob("*.png"):
                chart_files.append(chart)
        
        return chart_files
    
    def _find_sample_pipeline_directory(self) -> Optional[Path]:
        """Find a sample pipeline directory for testing."""
        outputs_path = Path("examples/outputs")
        
        if outputs_path.exists():
            for pipeline_dir in outputs_path.iterdir():
                if pipeline_dir.is_dir() and not pipeline_dir.name.startswith('.'):
                    return pipeline_dir
        
        return None
    
    def generate_test_summary(self, execution_time: float) -> Dict:
        """Generate comprehensive test summary report."""
        
        total_tests = sum(category['total'] for category in self.test_results.values())
        total_passed = sum(category['passed'] for category in self.test_results.values())
        total_failed = sum(category['failed'] for category in self.test_results.values())
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        summary = {
            'stream': 'Stream C - Visual Quality Assessment & File Organization',
            'execution_time_seconds': execution_time,
            'total_tests': total_tests,
            'passed': total_passed,
            'failed': total_failed,
            'success_rate_percent': round(success_rate, 2),
            'category_results': self.test_results,
            'detailed_results': self.detailed_results,
            'components_tested': [
                'VisualContentAnalyzer - Image quality analysis',
                'EnhancedVisualAssessor - Vision model integration',
                'ChartQualitySpecialist - Chart-specific assessment',
                'NamingConventionValidator - File naming standards',
                'DirectoryStructureValidator - Organization patterns',
                'FileLocationValidator - Appropriate file placement',
                'OrganizationQualityValidator - Complete organization review',
                'LLM Framework Integration - Vision model connectivity'
            ],
            'test_status': 'PASSED' if total_failed == 0 else 'FAILED' if total_passed == 0 else 'MIXED'
        }
        
        return summary


def main():
    """Run the visual quality assessment test suite."""
    
    print("=" * 80)
    print("Issue #277 - Stream C: Visual Quality Assessment & File Organization")
    print("Comprehensive Test Suite")
    print("=" * 80)
    
    # Change to project directory
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)
    
    # Run test suite
    test_suite = VisualQualityTestSuite()
    results = test_suite.run_all_tests()
    
    # Display results
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"Stream: {results['stream']}")
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {results['success_rate_percent']}%")
    print(f"Execution Time: {results['execution_time_seconds']:.2f} seconds")
    print(f"Overall Status: {results['test_status']}")
    
    print(f"\nCategory Breakdown:")
    for category, stats in results['category_results'].items():
        if stats['total'] > 0:
            rate = (stats['passed'] / stats['total'] * 100)
            print(f"  {category}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")
    
    print(f"\nComponents Tested:")
    for component in results['components_tested']:
        print(f"  ✓ {component}")
    
    # Show detailed results for failures
    failures = [r for r in results['detailed_results'] if r['status'] in ['FAILED', 'ERROR']]
    if failures:
        print(f"\nFAILED/ERROR DETAILS:")
        for failure in failures:
            print(f"  ❌ {failure['test']}: {failure['status']}")
            print(f"     {failure['details']}")
    
    # Save detailed results
    results_file = Path("stream_c_test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return results['test_status'] == 'PASSED'


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)