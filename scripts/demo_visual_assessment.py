#!/usr/bin/env python3
"""
Demo script showing Stream C visual quality assessment capabilities.

This script demonstrates the visual quality assessment and file organization
validation capabilities implemented for Issue #277 Stream C.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator.core.llm_quality_reviewer import LLMQualityReviewer
from orchestrator.quality.visual_assessor import (
    VisualContentAnalyzer, EnhancedVisualAssessor, ChartQualitySpecialist
)
from orchestrator.quality.organization_validator import OrganizationQualityValidator


async def demo_visual_assessment():
    """Demonstrate visual assessment capabilities."""
    
    print("=" * 80)
    print("Stream C Visual Quality Assessment Demo")
    print("=" * 80)
    
    # Initialize components
    visual_analyzer = VisualContentAnalyzer()
    enhanced_assessor = EnhancedVisualAssessor()
    chart_specialist = ChartQualitySpecialist()
    organization_validator = OrganizationQualityValidator()
    
    # Demo 1: Analyze creative image pipeline
    print("\n1. Creative Image Pipeline Analysis")
    print("-" * 40)
    
    creative_path = Path("examples/outputs/creative_image_pipeline")
    if creative_path.exists():
        # Organization assessment
        org_review = organization_validator.validate_pipeline_organization(
            str(creative_path),
            "creative_image_pipeline",
            "image_generation"
        )
        
        print(f"Pipeline: {creative_path.name}")
        print(f"Organization Issues Found: {len(org_review.issues)}")
        print(f"Correct Location: {org_review.correct_location}")
        print(f"Appropriate Naming: {org_review.appropriate_naming}")
        print(f"Expected Files Present: {org_review.expected_files_present}")
        
        # Show some specific issues
        if org_review.issues:
            print("\nTop Issues:")
            for i, issue in enumerate(org_review.issues[:3], 1):
                print(f"  {i}. {issue.severity.value.upper()}: {issue.description}")
        
        # Find and analyze some images
        image_files = list(creative_path.rglob("*.png"))[:3]  # First 3 images
        if image_files:
            print(f"\nVisual Content Analysis ({len(image_files)} sample images):")
            for img_path in image_files:
                issues = visual_analyzer.analyze_image_quality(str(img_path))
                print(f"  📸 {img_path.name}: {len(issues)} issues")
                if issues:
                    for issue in issues[:1]:  # Show first issue
                        print(f"     - {issue.description}")
    
    # Demo 2: Analyze modular analysis pipeline (charts)
    print("\n2. Modular Analysis Pipeline (Charts)")
    print("-" * 40)
    
    modular_path = Path("examples/outputs/modular_analysis")
    if modular_path.exists():
        charts_path = modular_path / "charts"
        if charts_path.exists():
            # Organization assessment for charts directory
            chart_issues = visual_analyzer.assess_visual_directory_structure(str(charts_path))
            print(f"Charts Directory: {charts_path.name}")
            print(f"Visual Organization Issues: {len(chart_issues)}")
            
            # Analyze individual charts
            chart_files = list(charts_path.glob("*.png"))
            print(f"Chart Files Found: {len(chart_files)}")
            
            for chart_file in chart_files[:2]:  # First 2 charts
                issues = visual_analyzer.analyze_chart_quality(str(chart_file))
                print(f"  📊 {chart_file.name}: {len(issues)} issues")
                
                # Generate chart-specific assessment prompt
                prompt = chart_specialist.create_chart_specific_prompt(str(chart_file))
                print(f"     Generated assessment prompt: {len(prompt)} chars")
    
    # Demo 3: File organization analysis across different pipeline types
    print("\n3. Cross-Pipeline Organization Analysis")
    print("-" * 40)
    
    outputs_path = Path("examples/outputs")
    if outputs_path.exists():
        pipeline_summaries = []
        
        for pipeline_dir in list(outputs_path.iterdir())[:5]:  # First 5 pipelines
            if pipeline_dir.is_dir() and not pipeline_dir.name.startswith('.'):
                try:
                    review = organization_validator.validate_pipeline_organization(
                        str(pipeline_dir),
                        pipeline_dir.name
                    )
                    
                    pipeline_summaries.append({
                        'name': pipeline_dir.name,
                        'issues': len(review.issues),
                        'organization_score': 'GOOD' if len(review.issues) < 3 else 'NEEDS_WORK'
                    })
                except Exception as e:
                    print(f"Error analyzing {pipeline_dir.name}: {e}")
        
        print("Pipeline Organization Summary:")
        for summary in pipeline_summaries:
            status_emoji = "✅" if summary['organization_score'] == 'GOOD' else "⚠️"
            print(f"  {status_emoji} {summary['name']}: {summary['issues']} issues ({summary['organization_score']})")
    
    # Demo 4: Enhanced visual assessment prompt generation
    print("\n4. Enhanced Assessment Capabilities")
    print("-" * 40)
    
    # Show prompt generation for different contexts
    sample_contexts = [
        {'pipeline_type': 'data_analysis', 'content_type': 'chart'},
        {'pipeline_type': 'image_generation', 'content_type': 'artistic'},
        {'pipeline_type': 'research', 'content_type': 'diagram'}
    ]
    
    for context in sample_contexts:
        prompt = enhanced_assessor.create_enhanced_visual_assessment_prompt(
            f"/sample/{context['content_type']}_image.png",
            context
        )
        
        print(f"Context: {context['pipeline_type']} -> {len(prompt)} char prompt")
        
        # Show key assessment criteria mentioned
        criteria_keywords = ['professional', 'chart', 'legend', 'accessibility', 'resolution']
        found_criteria = [keyword for keyword in criteria_keywords if keyword in prompt.lower()]
        print(f"  Assessment criteria: {', '.join(found_criteria)}")
    
    print("\n" + "=" * 80)
    print("Demo completed! Stream C visual quality assessment is operational.")
    print("=" * 80)


def main():
    """Run the demo."""
    # Change to project directory
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)
    
    # Run demo
    asyncio.run(demo_visual_assessment())


if __name__ == "__main__":
    main()