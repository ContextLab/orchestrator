#!/usr/bin/env python3
"""
Comprehensive Batch Pipeline Quality Review System

This script provides mass pipeline review capabilities with concurrent processing,
comprehensive reporting, and integration with existing validation tools.

Usage:
    python scripts/quality_review/batch_reviewer.py --all
    python scripts/quality_review/batch_reviewer.py --batch pipeline1,pipeline2,pipeline3
    python scripts/quality_review/batch_reviewer.py --continuous
    python scripts/quality_review/batch_reviewer.py --dashboard
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.core.llm_quality_reviewer import LLMQualityReviewer, LLMQualityError
from orchestrator.core.credential_manager import create_credential_manager
from orchestrator.core.quality_assessment import PipelineQualityReview
from validation.validate_all_pipelines import PipelineValidator


class BatchReviewConfig:
    """Configuration for batch review operations."""
    
    def __init__(
        self,
        max_concurrent_reviews: int = 3,
        timeout_per_pipeline: int = 300,  # 5 minutes per pipeline
        enable_caching: bool = True,
        cache_duration_hours: int = 24,
        output_directory: str = "quality_reports",
        enable_dashboard: bool = True,
        continuous_monitoring: bool = False,
        integration_with_validation: bool = True
    ):
        self.max_concurrent_reviews = max_concurrent_reviews
        self.timeout_per_pipeline = timeout_per_pipeline
        self.enable_caching = enable_caching
        self.cache_duration_hours = cache_duration_hours
        self.output_directory = Path(output_directory)
        self.enable_dashboard = enable_dashboard
        self.continuous_monitoring = continuous_monitoring
        self.integration_with_validation = integration_with_validation


class ReviewCache:
    """Simple file-based cache for review results."""
    
    def __init__(self, cache_dir: Path, duration_hours: int = 24):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.duration_seconds = duration_hours * 3600
    
    def get_cache_path(self, pipeline_name: str) -> Path:
        """Get cache file path for pipeline."""
        return self.cache_dir / f"{pipeline_name}_review_cache.json"
    
    def is_cached(self, pipeline_name: str) -> bool:
        """Check if pipeline review is cached and still valid."""
        cache_path = self.get_cache_path(pipeline_name)
        
        if not cache_path.exists():
            return False
        
        # Check if cache is still valid
        cache_age = time.time() - cache_path.stat().st_mtime
        return cache_age < self.duration_seconds
    
    def get_cached_review(self, pipeline_name: str) -> Optional[Dict[str, Any]]:
        """Get cached review if available and valid."""
        if not self.is_cached(pipeline_name):
            return None
        
        try:
            cache_path = self.get_cache_path(pipeline_name)
            with open(cache_path) as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load cache for {pipeline_name}: {e}")
            return None
    
    def cache_review(self, pipeline_name: str, review_data: Dict[str, Any]):
        """Cache review data for pipeline."""
        try:
            cache_path = self.get_cache_path(pipeline_name)
            with open(cache_path, 'w') as f:
                json.dump(review_data, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to cache review for {pipeline_name}: {e}")


class BatchProgressTracker:
    """Tracks progress of batch review operations."""
    
    def __init__(self, total_pipelines: int):
        self.total_pipelines = total_pipelines
        self.completed_pipelines = 0
        self.failed_pipelines = 0
        self.start_time = time.time()
        self.pipeline_times = {}
    
    def start_pipeline(self, pipeline_name: str):
        """Mark start of pipeline review."""
        self.pipeline_times[pipeline_name] = {"start": time.time()}
    
    def complete_pipeline(self, pipeline_name: str, success: bool = True):
        """Mark completion of pipeline review."""
        if pipeline_name in self.pipeline_times:
            self.pipeline_times[pipeline_name]["end"] = time.time()
            self.pipeline_times[pipeline_name]["duration"] = (
                self.pipeline_times[pipeline_name]["end"] - 
                self.pipeline_times[pipeline_name]["start"]
            )
        
        if success:
            self.completed_pipelines += 1
        else:
            self.failed_pipelines += 1
    
    def get_progress_report(self) -> Dict[str, Any]:
        """Get current progress report."""
        elapsed_time = time.time() - self.start_time
        
        # Calculate average time per pipeline
        completed_times = [
            data["duration"] for data in self.pipeline_times.values() 
            if "duration" in data
        ]
        avg_time = sum(completed_times) / len(completed_times) if completed_times else 0
        
        # Estimate remaining time
        remaining_pipelines = self.total_pipelines - self.completed_pipelines - self.failed_pipelines
        estimated_remaining_time = remaining_pipelines * avg_time
        
        return {
            "total_pipelines": self.total_pipelines,
            "completed": self.completed_pipelines,
            "failed": self.failed_pipelines,
            "remaining": remaining_pipelines,
            "progress_percentage": (
                (self.completed_pipelines + self.failed_pipelines) / self.total_pipelines * 100
            ),
            "elapsed_time_seconds": elapsed_time,
            "average_time_per_pipeline": avg_time,
            "estimated_remaining_time": estimated_remaining_time,
            "pipeline_times": self.pipeline_times
        }
    
    def print_progress(self):
        """Print current progress to console."""
        report = self.get_progress_report()
        
        print(f"\n📊 Batch Review Progress")
        print(f"   Completed: {report['completed']}/{report['total_pipelines']} "
              f"({report['progress_percentage']:.1f}%)")
        print(f"   Failed: {report['failed']}")
        print(f"   Elapsed: {report['elapsed_time_seconds']:.1f}s")
        print(f"   Avg time/pipeline: {report['average_time_per_pipeline']:.1f}s")
        
        if report['estimated_remaining_time'] > 0:
            print(f"   Estimated remaining: {report['estimated_remaining_time']:.1f}s")


class ComprehensiveBatchReviewer:
    """Comprehensive batch processing system for pipeline quality reviews."""
    
    def __init__(self, config: BatchReviewConfig = None):
        self.config = config or BatchReviewConfig()
        self.logger = self._setup_logging()
        
        # Initialize components
        self.credential_manager = create_credential_manager()
        self.reviewer = LLMQualityReviewer(credential_manager=self.credential_manager)
        self.validator = PipelineValidator() if self.config.integration_with_validation else None
        
        # Setup cache
        self.cache = ReviewCache(
            cache_dir=self.config.output_directory / "cache",
            duration_hours=self.config.cache_duration_hours
        ) if self.config.enable_caching else None
        
        # Create output directories
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        (self.config.output_directory / "reports").mkdir(exist_ok=True)
        (self.config.output_directory / "aggregated").mkdir(exist_ok=True)
        
        # Ensure cache directory exists if caching is enabled
        if self.config.enable_caching:
            (self.config.output_directory / "cache").mkdir(exist_ok=True)
        
        # Pipeline discovery
        self.available_pipelines = self._discover_pipelines()
        
        self.logger.info(f"Initialized Comprehensive Batch Reviewer")
        self.logger.info(f"  Available pipelines: {len(self.available_pipelines)}")
        self.logger.info(f"  Max concurrent reviews: {self.config.max_concurrent_reviews}")
        self.logger.info(f"  Cache enabled: {self.config.enable_caching}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging."""
        logger = logging.getLogger("BatchReviewer")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        try:
            log_file = self.config.output_directory / "batch_review.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # If we can't create the log file, continue without file logging
            print(f"Warning: Could not create log file: {e}")
        
        return logger
    
    def _discover_pipelines(self) -> List[str]:
        """Discover available pipelines from outputs directory."""
        outputs_dir = Path("examples/outputs")
        if not outputs_dir.exists():
            self.logger.warning("No examples/outputs directory found")
            return []
        
        pipelines = []
        for item in outputs_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                pipelines.append(item.name)
        
        self.logger.info(f"Discovered {len(pipelines)} pipelines")
        return sorted(pipelines)
    
    async def review_pipeline_with_timeout(
        self,
        pipeline_name: str,
        progress_tracker: BatchProgressTracker
    ) -> Tuple[str, Optional[PipelineQualityReview], Optional[Exception]]:
        """Review single pipeline with timeout and error handling."""
        progress_tracker.start_pipeline(pipeline_name)
        
        try:
            # Check cache first
            if self.cache and self.cache.is_cached(pipeline_name):
                cached_data = self.cache.get_cached_review(pipeline_name)
                if cached_data:
                    self.logger.info(f"Using cached review for {pipeline_name}")
                    progress_tracker.complete_pipeline(pipeline_name, success=True)
                    
                    # Convert cached data back to PipelineQualityReview
                    review = PipelineQualityReview.from_dict(cached_data)
                    return pipeline_name, review, None
            
            # Perform review with timeout
            try:
                review = await asyncio.wait_for(
                    self.reviewer.review_pipeline_outputs(pipeline_name),
                    timeout=self.config.timeout_per_pipeline
                )
                
                # Cache result if caching is enabled
                if self.cache:
                    self.cache.cache_review(pipeline_name, review.to_dict())
                
                progress_tracker.complete_pipeline(pipeline_name, success=True)
                self.logger.info(
                    f"✅ Completed review: {pipeline_name} "
                    f"(Score: {review.overall_score}/100, "
                    f"Issues: {review.total_issues})"
                )
                return pipeline_name, review, None
                
            except asyncio.TimeoutError:
                error = Exception(f"Review timeout after {self.config.timeout_per_pipeline}s")
                progress_tracker.complete_pipeline(pipeline_name, success=False)
                self.logger.error(f"⏰ Timeout reviewing {pipeline_name}")
                return pipeline_name, None, error
        
        except Exception as e:
            progress_tracker.complete_pipeline(pipeline_name, success=False)
            self.logger.error(f"❌ Failed to review {pipeline_name}: {e}")
            return pipeline_name, None, e
    
    async def batch_review_pipelines(
        self,
        pipeline_names: List[str],
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """Review multiple pipelines concurrently."""
        self.logger.info(f"Starting batch review of {len(pipeline_names)} pipelines")
        
        # Initialize progress tracking
        progress_tracker = BatchProgressTracker(len(pipeline_names))
        
        # Progress display task
        progress_task = None
        if show_progress:
            progress_task = asyncio.create_task(
                self._show_progress_periodically(progress_tracker)
            )
        
        try:
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(self.config.max_concurrent_reviews)
            
            async def review_with_semaphore(pipeline_name: str):
                async with semaphore:
                    return await self.review_pipeline_with_timeout(
                        pipeline_name, progress_tracker
                    )
            
            # Execute reviews concurrently
            tasks = [review_with_semaphore(name) for name in pipeline_names]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            successful_reviews = {}
            failed_reviews = {}
            
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Task failed with exception: {result}")
                    continue
                    
                pipeline_name, review, error = result
                if review is not None:
                    successful_reviews[pipeline_name] = review
                else:
                    failed_reviews[pipeline_name] = str(error) if error else "Unknown error"
            
            # Generate comprehensive report
            batch_report = self._generate_batch_report(
                successful_reviews, failed_reviews, progress_tracker
            )
            
            # Save batch report
            self._save_batch_report(batch_report)
            
            self.logger.info(
                f"Batch review completed: {len(successful_reviews)} successful, "
                f"{len(failed_reviews)} failed"
            )
            
            return batch_report
        
        finally:
            # Cancel progress display task
            if progress_task and not progress_task.done():
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass
    
    async def _show_progress_periodically(self, tracker: BatchProgressTracker):
        """Show progress updates periodically."""
        try:
            while tracker.completed_pipelines + tracker.failed_pipelines < tracker.total_pipelines:
                await asyncio.sleep(30)  # Update every 30 seconds
                tracker.print_progress()
        except asyncio.CancelledError:
            pass
    
    def _generate_batch_report(
        self,
        successful_reviews: Dict[str, PipelineQualityReview],
        failed_reviews: Dict[str, str],
        progress_tracker: BatchProgressTracker
    ) -> Dict[str, Any]:
        """Generate comprehensive batch review report."""
        
        # Calculate aggregate statistics
        total_pipelines = len(successful_reviews) + len(failed_reviews)
        
        if successful_reviews:
            scores = [review.overall_score for review in successful_reviews.values()]
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            
            # Count issues by severity
            total_critical = sum(len(review.critical_issues) for review in successful_reviews.values())
            total_major = sum(len(review.major_issues) for review in successful_reviews.values())
            total_minor = sum(len(review.minor_issues) for review in successful_reviews.values())
            
            # Production readiness
            production_ready = sum(1 for review in successful_reviews.values() if review.production_ready)
        else:
            avg_score = min_score = max_score = 0
            total_critical = total_major = total_minor = 0
            production_ready = 0
        
        # Performance metrics
        progress_report = progress_tracker.get_progress_report()
        
        return {
            "batch_review_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_pipelines": total_pipelines,
                "successful_reviews": len(successful_reviews),
                "failed_reviews": len(failed_reviews),
                "success_rate": (len(successful_reviews) / total_pipelines * 100) if total_pipelines > 0 else 0
            },
            "quality_metrics": {
                "average_score": avg_score,
                "minimum_score": min_score,
                "maximum_score": max_score,
                "production_ready_count": production_ready,
                "production_ready_percentage": (production_ready / len(successful_reviews) * 100) if successful_reviews else 0,
                "total_critical_issues": total_critical,
                "total_major_issues": total_major,
                "total_minor_issues": total_minor
            },
            "performance_metrics": {
                "total_duration_seconds": progress_report["elapsed_time_seconds"],
                "average_time_per_pipeline": progress_report["average_time_per_pipeline"],
                "fastest_pipeline": min(
                    progress_report["pipeline_times"].items(),
                    key=lambda x: x[1].get("duration", float('inf'))
                )[0] if progress_report["pipeline_times"] else None,
                "slowest_pipeline": max(
                    progress_report["pipeline_times"].items(),
                    key=lambda x: x[1].get("duration", 0)
                )[0] if progress_report["pipeline_times"] else None
            },
            "detailed_reviews": {
                pipeline_name: review.to_dict()
                for pipeline_name, review in successful_reviews.items()
            },
            "failed_pipelines": failed_reviews,
            "pipeline_rankings": self._rank_pipelines_by_quality(successful_reviews)
        }
    
    def _rank_pipelines_by_quality(
        self,
        reviews: Dict[str, PipelineQualityReview]
    ) -> List[Dict[str, Any]]:
        """Rank pipelines by quality score."""
        rankings = []
        
        for pipeline_name, review in reviews.items():
            rankings.append({
                "pipeline_name": pipeline_name,
                "overall_score": review.overall_score,
                "production_ready": review.production_ready,
                "total_issues": review.total_issues,
                "critical_issues": len(review.critical_issues),
                "major_issues": len(review.major_issues),
                "minor_issues": len(review.minor_issues)
            })
        
        # Sort by score (descending), then by issue count (ascending)
        rankings.sort(key=lambda x: (-x["overall_score"], x["total_issues"]))
        
        return rankings
    
    def _save_batch_report(self, batch_report: Dict[str, Any]):
        """Save batch report to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_path = self.config.output_directory / "aggregated" / f"batch_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(batch_report, f, indent=2)
        
        # Save Markdown summary
        md_path = self.config.output_directory / "aggregated" / f"batch_summary_{timestamp}.md"
        with open(md_path, 'w') as f:
            f.write(self._generate_markdown_summary(batch_report))
        
        # Save latest report (overwrite)
        latest_json = self.config.output_directory / "aggregated" / "latest_batch_report.json"
        latest_md = self.config.output_directory / "aggregated" / "latest_batch_summary.md"
        
        with open(latest_json, 'w') as f:
            json.dump(batch_report, f, indent=2)
        
        with open(latest_md, 'w') as f:
            f.write(self._generate_markdown_summary(batch_report))
        
        self.logger.info(f"Batch report saved:")
        self.logger.info(f"  JSON: {json_path}")
        self.logger.info(f"  Markdown: {md_path}")
    
    def _generate_markdown_summary(self, batch_report: Dict[str, Any]) -> str:
        """Generate markdown summary of batch report."""
        summary = batch_report["batch_review_summary"]
        quality = batch_report["quality_metrics"]
        performance = batch_report["performance_metrics"]
        rankings = batch_report["pipeline_rankings"]
        
        md_lines = [
            f"# Batch Quality Review Report",
            f"",
            f"**Generated:** {summary['timestamp']}",
            f"**Total Pipelines:** {summary['total_pipelines']}",
            f"**Success Rate:** {summary['success_rate']:.1f}%",
            f"",
            f"## Quality Overview",
            f"",
            f"- **Average Score:** {quality['average_score']:.1f}/100",
            f"- **Production Ready:** {quality['production_ready_count']}/{summary['successful_reviews']} ({quality['production_ready_percentage']:.1f}%)",
            f"- **Score Range:** {quality['minimum_score']:.1f} - {quality['maximum_score']:.1f}",
            f"",
            f"### Issues Summary",
            f"- **Critical Issues:** {quality['total_critical_issues']}",
            f"- **Major Issues:** {quality['total_major_issues']}",
            f"- **Minor Issues:** {quality['total_minor_issues']}",
            f"",
            f"## Performance Metrics",
            f"",
            f"- **Total Duration:** {performance['total_duration_seconds']:.1f} seconds",
            f"- **Average Time per Pipeline:** {performance['average_time_per_pipeline']:.1f} seconds",
            f"- **Fastest Pipeline:** {performance.get('fastest_pipeline', 'N/A')}",
            f"- **Slowest Pipeline:** {performance.get('slowest_pipeline', 'N/A')}",
            f"",
            f"## Pipeline Quality Rankings",
            f""
        ]
        
        # Top 10 pipelines by quality
        md_lines.extend([
            "| Rank | Pipeline | Score | Production Ready | Critical | Major | Minor |",
            "|------|----------|-------|------------------|----------|-------|--------|"
        ])
        
        for i, ranking in enumerate(rankings[:10], 1):
            ready_icon = "✅" if ranking["production_ready"] else "❌"
            md_lines.append(
                f"| {i} | {ranking['pipeline_name']} | {ranking['overall_score']:.1f} | "
                f"{ready_icon} | {ranking['critical_issues']} | {ranking['major_issues']} | "
                f"{ranking['minor_issues']} |"
            )
        
        # Failed pipelines
        failed_pipelines = batch_report.get("failed_pipelines", {})
        if failed_pipelines:
            md_lines.extend([
                f"",
                f"## Failed Pipeline Reviews",
                f""
            ])
            
            for pipeline_name, error in failed_pipelines.items():
                md_lines.append(f"- **{pipeline_name}**: {error}")
        
        return "\n".join(md_lines)
    
    async def continuous_monitoring(self, check_interval_minutes: int = 60):
        """Continuous monitoring mode for production automation."""
        self.logger.info(f"Starting continuous monitoring (check every {check_interval_minutes} minutes)")
        
        while True:
            try:
                self.logger.info("Running scheduled batch review...")
                
                # Review all available pipelines
                await self.batch_review_pipelines(
                    self.available_pipelines,
                    show_progress=False
                )
                
                self.logger.info(f"Scheduled review completed. Next check in {check_interval_minutes} minutes.")
                
                # Wait for next check
                await asyncio.sleep(check_interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Error in continuous monitoring: {e}")
                # Wait shorter time before retry on error
                await asyncio.sleep(5 * 60)  # 5 minutes
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for quality dashboard."""
        # Load latest batch report
        latest_report_path = self.config.output_directory / "aggregated" / "latest_batch_report.json"
        
        if not latest_report_path.exists():
            return {"error": "No batch report data available"}
        
        with open(latest_report_path) as f:
            batch_report = json.load(f)
        
        # Extract dashboard-relevant data
        dashboard_data = {
            "last_updated": batch_report["batch_review_summary"]["timestamp"],
            "overall_statistics": {
                "total_pipelines": batch_report["batch_review_summary"]["total_pipelines"],
                "success_rate": batch_report["batch_review_summary"]["success_rate"],
                "average_quality_score": batch_report["quality_metrics"]["average_score"],
                "production_ready_percentage": batch_report["quality_metrics"]["production_ready_percentage"]
            },
            "quality_distribution": self._generate_quality_distribution(batch_report),
            "top_issues": self._extract_top_issues(batch_report),
            "performance_trends": {
                "average_review_time": batch_report["performance_metrics"]["average_time_per_pipeline"],
                "total_duration": batch_report["performance_metrics"]["total_duration_seconds"]
            },
            "pipeline_rankings": batch_report["pipeline_rankings"][:10]  # Top 10
        }
        
        return dashboard_data
    
    def _generate_quality_distribution(self, batch_report: Dict[str, Any]) -> Dict[str, int]:
        """Generate quality score distribution for dashboard."""
        distribution = {
            "excellent_90_plus": 0,
            "good_80_89": 0,
            "fair_70_79": 0,
            "poor_60_69": 0,
            "critical_below_60": 0
        }
        
        for pipeline_data in batch_report["detailed_reviews"].values():
            score = pipeline_data["overall_score"]
            
            if score >= 90:
                distribution["excellent_90_plus"] += 1
            elif score >= 80:
                distribution["good_80_89"] += 1
            elif score >= 70:
                distribution["fair_70_79"] += 1
            elif score >= 60:
                distribution["poor_60_69"] += 1
            else:
                distribution["critical_below_60"] += 1
        
        return distribution
    
    def _extract_top_issues(self, batch_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract most common issues across all pipelines."""
        issue_counts = {}
        
        for pipeline_data in batch_report["detailed_reviews"].values():
            # Count all issues
            all_issues = (
                pipeline_data.get("critical_issues", []) +
                pipeline_data.get("major_issues", []) +
                pipeline_data.get("minor_issues", [])
            )
            
            for issue in all_issues:
                issue_desc = issue.get("description", "Unknown issue")
                if issue_desc in issue_counts:
                    issue_counts[issue_desc] += 1
                else:
                    issue_counts[issue_desc] = 1
        
        # Return top 10 issues
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return [
            {"description": desc, "count": count}
            for desc, count in sorted_issues[:10]
        ]


async def main():
    """Main function for batch review operations."""
    parser = argparse.ArgumentParser(
        description="Comprehensive batch pipeline quality review system"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Review all available pipelines"
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Comma-separated list of specific pipelines to review"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run in continuous monitoring mode"
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Generate dashboard data and exit"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum concurrent reviews (default: 3)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per pipeline in seconds (default: 300)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="quality_reports",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=60,
        help="Check interval for continuous monitoring in minutes (default: 60)"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = BatchReviewConfig(
        max_concurrent_reviews=args.max_concurrent,
        timeout_per_pipeline=args.timeout,
        enable_caching=not args.no_cache,
        output_directory=args.output_dir,
        continuous_monitoring=args.continuous
    )
    
    # Initialize batch reviewer
    reviewer = ComprehensiveBatchReviewer(config)
    
    # Handle different modes
    if args.dashboard:
        dashboard_data = reviewer.generate_dashboard_data()
        print(json.dumps(dashboard_data, indent=2))
        return
    
    if args.continuous:
        await reviewer.continuous_monitoring(args.check_interval)
        return
    
    # Determine pipelines to review
    if args.all:
        pipelines_to_review = reviewer.available_pipelines
    elif args.batch:
        pipelines_to_review = [p.strip() for p in args.batch.split(",")]
    else:
        parser.print_help()
        return
    
    if not pipelines_to_review:
        print("No pipelines specified for review")
        return
    
    print(f"🚀 Starting batch review of {len(pipelines_to_review)} pipelines")
    print(f"📊 Configuration:")
    print(f"   Max concurrent: {config.max_concurrent_reviews}")
    print(f"   Timeout per pipeline: {config.timeout_per_pipeline}s")
    print(f"   Caching enabled: {config.enable_caching}")
    print(f"   Output directory: {config.output_directory}")
    
    # Execute batch review
    start_time = time.time()
    
    try:
        batch_report = await reviewer.batch_review_pipelines(pipelines_to_review)
        
        duration = time.time() - start_time
        
        # Print final summary
        summary = batch_report["batch_review_summary"]
        quality = batch_report["quality_metrics"]
        
        print(f"\n🎉 Batch Review Complete!")
        print(f"   Total time: {duration:.1f} seconds")
        print(f"   Success rate: {summary['success_rate']:.1f}%")
        print(f"   Average quality score: {quality['average_score']:.1f}/100")
        print(f"   Production ready: {quality['production_ready_count']}/{summary['successful_reviews']}")
        
        if batch_report.get("failed_pipelines"):
            print(f"\n⚠️  Failed reviews: {len(batch_report['failed_pipelines'])}")
            for pipeline, error in batch_report["failed_pipelines"].items():
                print(f"   - {pipeline}: {error}")
        
        # Show top performing pipelines
        rankings = batch_report["pipeline_rankings"]
        if rankings:
            print(f"\n🏆 Top 5 Quality Pipelines:")
            for i, ranking in enumerate(rankings[:5], 1):
                ready_icon = "✅" if ranking["production_ready"] else "❌"
                print(f"   {i}. {ranking['pipeline_name']}: {ranking['overall_score']:.1f}/100 {ready_icon}")
        
    except KeyboardInterrupt:
        print("\n🛑 Batch review interrupted by user")
    except Exception as e:
        print(f"\n❌ Batch review failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())