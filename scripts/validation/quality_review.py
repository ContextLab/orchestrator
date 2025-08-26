#!/usr/bin/env python3
"""
LLM Quality Review Script

This script provides comprehensive quality assessment of pipeline outputs using
LLM-powered analysis with Claude Sonnet 4 and ChatGPT-5 vision capabilities.

Usage:
    python scripts/validation/quality_review.py <pipeline_name>
    python scripts/validation/quality_review.py --all
    python scripts/validation/quality_review.py --batch pipeline1,pipeline2,pipeline3
    python scripts/validation/quality_review.py --report-only
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from orchestrator.core.llm_quality_reviewer import LLMQualityReviewer, LLMQualityError
from orchestrator.core.credential_manager import create_credential_manager


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('quality_review.log')
        ]
    )


def get_available_pipelines() -> List[str]:
    """Get list of available pipelines from examples/outputs directory."""
    outputs_dir = Path("examples/outputs")
    if not outputs_dir.exists():
        return []
    
    pipelines = []
    for item in outputs_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            pipelines.append(item.name)
    
    return sorted(pipelines)


def save_review_report(review, output_path: Optional[Path] = None):
    """Save quality review report to JSON and markdown files."""
    if output_path is None:
        output_path = Path("quality_reports")
    
    output_path.mkdir(exist_ok=True)
    
    # Save JSON report
    json_path = output_path / f"{review.pipeline_name}_quality_report.json"
    with open(json_path, 'w') as f:
        json.dump(review.to_dict(), f, indent=2)
    
    # Save Markdown summary
    md_path = output_path / f"{review.pipeline_name}_quality_summary.md"
    with open(md_path, 'w') as f:
        f.write(generate_markdown_report(review))
    
    print(f"Reports saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")


def generate_markdown_report(review) -> str:
    """Generate a markdown quality report."""
    md_lines = []
    
    # Header
    md_lines.extend([
        f"# Quality Review: {review.pipeline_name}",
        "",
        f"**Overall Score:** {review.overall_score}/100",
        f"**Production Ready:** {'✅ Yes' if review.production_ready else '❌ No'}",
        f"**Review Date:** {review.reviewed_at}",
        f"**Model Used:** {review.reviewer_model}",
        f"**Duration:** {review.review_duration_seconds:.2f} seconds",
        f"**Files Reviewed:** {len(review.files_reviewed)}",
        ""
    ])
    
    # Score interpretation
    if review.overall_score >= 90:
        status = "🟢 Excellent - Production ready, no issues"
    elif review.overall_score >= 80:
        status = "🟡 Good - Minor issues, acceptable for showcase"
    elif review.overall_score >= 70:
        status = "🟠 Fair - Some issues, needs improvement before release"
    elif review.overall_score >= 60:
        status = "🔴 Poor - Major issues, significant work needed"
    else:
        status = "🚫 Critical - Not suitable for production"
    
    md_lines.extend([
        "## Quality Assessment",
        "",
        f"**Status:** {status}",
        ""
    ])
    
    # Issues summary
    if review.total_issues > 0:
        md_lines.extend([
            "## Issues Found",
            "",
            f"- **Critical Issues:** {len(review.critical_issues)}",
            f"- **Major Issues:** {len(review.major_issues)}", 
            f"- **Minor Issues:** {len(review.minor_issues)}",
            ""
        ])
        
        # Critical issues detail
        if review.critical_issues:
            md_lines.extend([
                "### 🚨 Critical Issues (Must Fix)",
                ""
            ])
            for issue in review.critical_issues:
                md_lines.extend([
                    f"**File:** `{issue.file_path}`",
                    f"**Issue:** {issue.description}",
                    f"**Suggestion:** {issue.suggestion}",
                    ""
                ])
        
        # Major issues detail
        if review.major_issues:
            md_lines.extend([
                "### ⚠️ Major Issues (Should Fix)",
                ""
            ])
            for issue in review.major_issues[:5]:  # Limit to first 5
                md_lines.extend([
                    f"**File:** `{issue.file_path}`",
                    f"**Issue:** {issue.description}",
                    f"**Suggestion:** {issue.suggestion}",
                    ""
                ])
            
            if len(review.major_issues) > 5:
                md_lines.append(f"*... and {len(review.major_issues) - 5} more major issues*\n")
    else:
        md_lines.extend([
            "## ✅ No Issues Found",
            "",
            "All files meet production quality standards!",
            ""
        ])
    
    # Recommendations
    if review.recommendations:
        md_lines.extend([
            "## Recommendations",
            ""
        ])
        for rec in review.recommendations:
            md_lines.append(f"- {rec}")
        md_lines.append("")
    
    # Files reviewed
    md_lines.extend([
        "## Files Reviewed",
        ""
    ])
    for file_path in review.files_reviewed:
        md_lines.append(f"- `{file_path}`")
    
    return "\n".join(md_lines)


async def review_single_pipeline(pipeline_name: str, reviewer: LLMQualityReviewer) -> bool:
    """Review a single pipeline and return success status."""
    print(f"\n🔍 Reviewing pipeline: {pipeline_name}")
    
    try:
        review = await reviewer.review_pipeline_outputs(pipeline_name)
        
        # Print summary
        status_icon = "✅" if review.production_ready else "❌"
        print(f"{status_icon} {pipeline_name}: {review.overall_score}/100 "
              f"({review.total_issues} issues)")
        
        if review.critical_issues:
            print(f"   🚨 {len(review.critical_issues)} critical issues")
        if review.major_issues:
            print(f"   ⚠️  {len(review.major_issues)} major issues")
        if review.minor_issues:
            print(f"   ℹ️  {len(review.minor_issues)} minor issues")
        
        # Save detailed report
        save_review_report(review)
        
        return review.production_ready
        
    except Exception as e:
        print(f"❌ Failed to review {pipeline_name}: {e}")
        logging.error(f"Pipeline review failed: {pipeline_name} - {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(
        description="LLM-powered quality assessment for pipeline outputs"
    )
    parser.add_argument(
        "pipeline",
        nargs="?",
        help="Pipeline name to review (or --all for all pipelines)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Review all available pipelines"
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Comma-separated list of pipelines to review"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available pipelines and exit"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate reports from existing reviews only"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("quality_reports"),
        help="Output directory for reports (default: quality_reports)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Handle list option
    if args.list:
        pipelines = get_available_pipelines()
        print("Available pipelines:")
        for pipeline in pipelines:
            print(f"  - {pipeline}")
        return
    
    # Handle report-only option
    if args.report_only:
        print("Report-only mode not yet implemented")
        return
    
    # Determine pipelines to review
    if args.all:
        pipelines_to_review = get_available_pipelines()
    elif args.batch:
        pipelines_to_review = [p.strip() for p in args.batch.split(",")]
    elif args.pipeline:
        pipelines_to_review = [args.pipeline]
    else:
        parser.print_help()
        return
    
    if not pipelines_to_review:
        print("No pipelines to review")
        return
    
    print(f"Starting quality review for {len(pipelines_to_review)} pipeline(s)")
    print(f"Reports will be saved to: {args.output}")
    
    # Initialize reviewer
    try:
        credential_manager = create_credential_manager()
        reviewer = LLMQualityReviewer(credential_manager=credential_manager)
    except Exception as e:
        print(f"Failed to initialize quality reviewer: {e}")
        logging.error(f"Reviewer initialization failed: {e}")
        return
    
    # Review pipelines
    start_time = time.time()
    successful_reviews = 0
    production_ready_count = 0
    
    for pipeline in pipelines_to_review:
        try:
            is_production_ready = await review_single_pipeline(pipeline, reviewer)
            successful_reviews += 1
            if is_production_ready:
                production_ready_count += 1
        except Exception as e:
            print(f"❌ Error reviewing {pipeline}: {e}")
            logging.error(f"Pipeline review error: {pipeline} - {e}")
    
    # Final summary
    duration = time.time() - start_time
    print(f"\n📊 Quality Review Summary")
    print(f"   Pipelines reviewed: {successful_reviews}/{len(pipelines_to_review)}")
    print(f"   Production ready: {production_ready_count}/{successful_reviews}")
    print(f"   Total time: {duration:.2f} seconds")
    print(f"   Reports saved to: {args.output}")
    
    if successful_reviews < len(pipelines_to_review):
        print(f"\n⚠️  {len(pipelines_to_review) - successful_reviews} pipelines failed review")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())