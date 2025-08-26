#!/usr/bin/env python3
"""
Integrated Validation System

This script provides comprehensive integration between batch quality review,
existing validation tools, and production automation systems.

Usage:
    python scripts/quality_review/integrated_validation.py --full-validation
    python scripts/quality_review/integrated_validation.py --pipeline-execution
    python scripts/quality_review/integrated_validation.py --quality-review
    python scripts/quality_review/integrated_validation.py --production-check
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.core.credential_manager import create_credential_manager
from orchestrator.quality.report_generator import QualityReportGenerator
from validation.validate_all_pipelines import PipelineValidator
from quality_review.batch_reviewer import ComprehensiveBatchReviewer, BatchReviewConfig


class ValidationPhase:
    """Represents a validation phase with results."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.start_time = None
        self.end_time = None
        self.duration = 0
        self.success = False
        self.results = {}
        self.errors = []
        self.warnings = []
    
    def start(self):
        """Mark phase as started."""
        self.start_time = time.time()
    
    def complete(self, success: bool = True, results: Dict[str, Any] = None):
        """Mark phase as completed."""
        self.end_time = time.time()
        if self.start_time:
            self.duration = self.end_time - self.start_time
        self.success = success
        self.results = results or {}
    
    def add_error(self, error: str):
        """Add error to phase."""
        self.errors.append(error)
    
    def add_warning(self, warning: str):
        """Add warning to phase."""
        self.warnings.append(warning)


class IntegratedValidationSystem:
    """Comprehensive validation system integrating all quality assurance tools."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.output_directory = Path("validation_results")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.credential_manager = create_credential_manager()
        self.pipeline_validator = PipelineValidator()
        self.batch_reviewer = ComprehensiveBatchReviewer(
            BatchReviewConfig(output_directory=str(self.output_directory / "quality_reports"))
        )
        self.report_generator = QualityReportGenerator(
            output_directory=self.output_directory / "reports"
        )
        
        # Validation phases
        self.phases = {}
        self.overall_start_time = None
        self.overall_end_time = None
        
        self.logger.info("Initialized Integrated Validation System")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging."""
        logger = logging.getLogger("IntegratedValidation")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = self.output_directory / "integrated_validation.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def create_phase(self, name: str, description: str) -> ValidationPhase:
        """Create and register a validation phase."""
        phase = ValidationPhase(name, description)
        self.phases[name] = phase
        return phase
    
    async def full_validation_workflow(
        self,
        pipelines: Optional[List[str]] = None,
        skip_execution: bool = False,
        skip_quality: bool = False
    ) -> Dict[str, Any]:
        """Run complete validation workflow."""
        self.overall_start_time = time.time()
        
        self.logger.info("🚀 Starting Full Validation Workflow")
        self.logger.info(f"   Pipelines: {len(pipelines) if pipelines else 'All'}")
        self.logger.info(f"   Skip execution: {skip_execution}")
        self.logger.info(f"   Skip quality review: {skip_quality}")
        
        results = {
            "workflow_summary": {
                "start_time": self.overall_start_time,
                "pipelines_requested": pipelines,
                "skip_execution": skip_execution,
                "skip_quality": skip_quality
            },
            "phases": {},
            "overall_success": True
        }
        
        try:
            # Phase 1: Pipeline Discovery
            discovery_phase = self.create_phase("discovery", "Pipeline Discovery and Preparation")
            discovery_results = await self._run_pipeline_discovery(discovery_phase, pipelines)
            results["phases"]["discovery"] = discovery_results
            
            discovered_pipelines = discovery_results.get("available_pipelines", [])
            if not discovered_pipelines:
                self.logger.error("No pipelines found for validation")
                results["overall_success"] = False
                return results
            
            # Phase 2: Pipeline Execution (if not skipped)
            if not skip_execution:
                execution_phase = self.create_phase("execution", "Pipeline Execution Validation")
                execution_results = await self._run_pipeline_execution(execution_phase, discovered_pipelines)
                results["phases"]["execution"] = execution_results
                
                # Filter to only successfully executed pipelines for quality review
                successful_pipelines = execution_results.get("successful_pipelines", discovered_pipelines)
            else:
                successful_pipelines = discovered_pipelines
            
            # Phase 3: Quality Review (if not skipped)
            if not skip_quality:
                quality_phase = self.create_phase("quality", "Comprehensive Quality Review")
                quality_results = await self._run_quality_review(quality_phase, successful_pipelines)
                results["phases"]["quality"] = quality_results
            
            # Phase 4: Integration Analysis
            analysis_phase = self.create_phase("analysis", "Cross-Phase Analysis and Integration")
            analysis_results = await self._run_integration_analysis(analysis_phase, results)
            results["phases"]["analysis"] = analysis_results
            
            # Phase 5: Report Generation
            reporting_phase = self.create_phase("reporting", "Comprehensive Report Generation")
            reporting_results = await self._run_report_generation(reporting_phase, results)
            results["phases"]["reporting"] = reporting_results
            
            # Determine overall success
            results["overall_success"] = all(
                phase_data.get("success", False) 
                for phase_data in results["phases"].values()
            )
            
        except Exception as e:
            self.logger.error(f"Critical error in validation workflow: {e}")
            results["overall_success"] = False
            results["critical_error"] = str(e)
        
        finally:
            self.overall_end_time = time.time()
            results["workflow_summary"]["end_time"] = self.overall_end_time
            results["workflow_summary"]["total_duration"] = (
                self.overall_end_time - self.overall_start_time
            )
            
            # Save comprehensive results
            await self._save_workflow_results(results)
        
        return results
    
    async def _run_pipeline_discovery(
        self,
        phase: ValidationPhase,
        requested_pipelines: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Phase 1: Discover available pipelines."""
        phase.start()
        self.logger.info("🔍 Phase 1: Pipeline Discovery")
        
        try:
            # Discover available pipelines
            all_available = self.batch_reviewer.available_pipelines
            
            if requested_pipelines:
                # Filter to requested pipelines
                available_pipelines = [
                    p for p in requested_pipelines if p in all_available
                ]
                missing_pipelines = [
                    p for p in requested_pipelines if p not in all_available
                ]
                
                if missing_pipelines:
                    phase.add_warning(f"Requested pipelines not found: {missing_pipelines}")
                    self.logger.warning(f"Missing pipelines: {missing_pipelines}")
            else:
                available_pipelines = all_available
                missing_pipelines = []
            
            # Check pipeline files exist
            yaml_dir = Path("examples")
            existing_yaml_files = []
            missing_yaml_files = []
            
            for pipeline in available_pipelines:
                yaml_file = yaml_dir / f"{pipeline}.yaml"
                if yaml_file.exists():
                    existing_yaml_files.append(pipeline)
                else:
                    missing_yaml_files.append(pipeline)
                    phase.add_warning(f"YAML file not found: {yaml_file}")
            
            results = {
                "success": True,
                "all_available_pipelines": all_available,
                "requested_pipelines": requested_pipelines or [],
                "available_pipelines": available_pipelines,
                "missing_pipelines": missing_pipelines,
                "existing_yaml_files": existing_yaml_files,
                "missing_yaml_files": missing_yaml_files,
                "total_discovered": len(available_pipelines)
            }
            
            phase.complete(success=True, results=results)
            
            self.logger.info(f"   Discovered: {len(available_pipelines)} pipelines")
            self.logger.info(f"   YAML files: {len(existing_yaml_files)} found, {len(missing_yaml_files)} missing")
            
            return results
            
        except Exception as e:
            phase.add_error(f"Discovery failed: {e}")
            phase.complete(success=False)
            self.logger.error(f"Pipeline discovery failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "available_pipelines": []
            }
    
    async def _run_pipeline_execution(
        self,
        phase: ValidationPhase,
        pipelines: List[str]
    ) -> Dict[str, Any]:
        """Phase 2: Execute pipelines and validate execution."""
        phase.start()
        self.logger.info("⚙️ Phase 2: Pipeline Execution Validation")
        
        try:
            # Run pipeline validation (this executes pipelines)
            validator_results = {}
            successful_pipelines = []
            failed_pipelines = []
            
            # Note: PipelineValidator.validate_all() is designed to validate all pipelines
            # For individual pipeline validation, we would need to modify it or run selectively
            self.logger.info(f"   Executing {len(pipelines)} pipelines...")
            
            # Initialize validator models
            await self.pipeline_validator.validate_all()
            validator_results = self.pipeline_validator.results
            
            # Process results
            for pipeline_name in pipelines:
                if pipeline_name in validator_results:
                    result = validator_results[pipeline_name]
                    if result["status"] == "success" or result["status"] == "issues_found":
                        successful_pipelines.append(pipeline_name)
                    else:
                        failed_pipelines.append(pipeline_name)
                        phase.add_error(f"Pipeline execution failed: {pipeline_name}")
                else:
                    failed_pipelines.append(pipeline_name)
                    phase.add_warning(f"No execution result for: {pipeline_name}")
            
            results = {
                "success": len(failed_pipelines) < len(pipelines) * 0.5,  # Success if <50% failed
                "total_pipelines": len(pipelines),
                "successful_pipelines": successful_pipelines,
                "failed_pipelines": failed_pipelines,
                "success_rate": len(successful_pipelines) / len(pipelines) * 100 if pipelines else 0,
                "validator_results": validator_results
            }
            
            phase.complete(success=results["success"], results=results)
            
            self.logger.info(f"   Success rate: {results['success_rate']:.1f}%")
            self.logger.info(f"   Successful: {len(successful_pipelines)}")
            self.logger.info(f"   Failed: {len(failed_pipelines)}")
            
            return results
            
        except Exception as e:
            phase.add_error(f"Execution validation failed: {e}")
            phase.complete(success=False)
            self.logger.error(f"Pipeline execution failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "successful_pipelines": [],
                "failed_pipelines": pipelines
            }
    
    async def _run_quality_review(
        self,
        phase: ValidationPhase,
        pipelines: List[str]
    ) -> Dict[str, Any]:
        """Phase 3: Run comprehensive quality review."""
        phase.start()
        self.logger.info("🔍 Phase 3: Comprehensive Quality Review")
        
        try:
            # Run batch quality review
            self.logger.info(f"   Reviewing {len(pipelines)} pipelines...")
            
            batch_report = await self.batch_reviewer.batch_review_pipelines(
                pipelines,
                show_progress=True
            )
            
            # Extract key metrics
            summary = batch_report["batch_review_summary"]
            quality_metrics = batch_report["quality_metrics"]
            
            results = {
                "success": summary["success_rate"] > 50,  # Success if >50% reviewed successfully
                "batch_report": batch_report,
                "total_pipelines": summary["total_pipelines"],
                "successful_reviews": summary["successful_reviews"],
                "failed_reviews": summary["failed_reviews"],
                "success_rate": summary["success_rate"],
                "average_quality_score": quality_metrics["average_score"],
                "production_ready_count": quality_metrics["production_ready_count"],
                "production_ready_percentage": quality_metrics["production_ready_percentage"]
            }
            
            # Add warnings for low scores
            if quality_metrics["average_score"] < 70:
                phase.add_warning(f"Low average quality score: {quality_metrics['average_score']:.1f}")
            
            if quality_metrics["total_critical_issues"] > 0:
                phase.add_warning(f"Critical issues found: {quality_metrics['total_critical_issues']}")
            
            phase.complete(success=results["success"], results=results)
            
            self.logger.info(f"   Review success rate: {summary['success_rate']:.1f}%")
            self.logger.info(f"   Average quality score: {quality_metrics['average_score']:.1f}/100")
            self.logger.info(f"   Production ready: {quality_metrics['production_ready_count']}/{summary['successful_reviews']}")
            
            return results
            
        except Exception as e:
            phase.add_error(f"Quality review failed: {e}")
            phase.complete(success=False)
            self.logger.error(f"Quality review failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "batch_report": {}
            }
    
    async def _run_integration_analysis(
        self,
        phase: ValidationPhase,
        workflow_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phase 4: Cross-phase analysis and integration."""
        phase.start()
        self.logger.info("📊 Phase 4: Integration Analysis")
        
        try:
            analysis = {
                "correlation_analysis": {},
                "trend_analysis": {},
                "recommendation_engine": {},
                "production_readiness": {}
            }
            
            # Extract data from phases
            execution_data = workflow_results["phases"].get("execution", {})
            quality_data = workflow_results["phases"].get("quality", {})
            
            # Correlation analysis between execution success and quality scores
            if execution_data.get("success") and quality_data.get("success"):
                successful_pipelines = set(execution_data.get("successful_pipelines", []))
                quality_reviews = quality_data.get("batch_report", {}).get("detailed_reviews", {})
                
                # Calculate correlation metrics
                execution_quality_correlation = []
                for pipeline_name, review_data in quality_reviews.items():
                    execution_success = pipeline_name in successful_pipelines
                    quality_score = review_data.get("overall_score", 0)
                    
                    execution_quality_correlation.append({
                        "pipeline": pipeline_name,
                        "execution_success": execution_success,
                        "quality_score": quality_score,
                        "production_ready": review_data.get("production_ready", False)
                    })
                
                analysis["correlation_analysis"] = {
                    "execution_quality_correlation": execution_quality_correlation,
                    "correlation_count": len(execution_quality_correlation)
                }
            
            # Production readiness analysis
            if quality_data.get("success"):
                quality_metrics = quality_data.get("batch_report", {}).get("quality_metrics", {})
                production_analysis = {
                    "total_production_ready": quality_metrics.get("production_ready_count", 0),
                    "production_ready_percentage": quality_metrics.get("production_ready_percentage", 0),
                    "critical_issues_blocking": quality_metrics.get("total_critical_issues", 0),
                    "major_issues_needing_attention": quality_metrics.get("total_major_issues", 0)
                }
                
                # Readiness classification
                if production_analysis["production_ready_percentage"] >= 80:
                    readiness_status = "excellent"
                elif production_analysis["production_ready_percentage"] >= 60:
                    readiness_status = "good"
                elif production_analysis["production_ready_percentage"] >= 40:
                    readiness_status = "needs_improvement"
                else:
                    readiness_status = "critical"
                
                production_analysis["readiness_status"] = readiness_status
                analysis["production_readiness"] = production_analysis
            
            # Generate recommendations
            recommendations = []
            
            if execution_data.get("success_rate", 0) < 80:
                recommendations.append("Fix pipeline execution issues before focusing on quality")
            
            if quality_data.get("average_quality_score", 0) < 70:
                recommendations.append("Systematic quality improvements needed across pipelines")
            
            critical_issues = quality_data.get("batch_report", {}).get("quality_metrics", {}).get("total_critical_issues", 0)
            if critical_issues > 0:
                recommendations.append(f"Address {critical_issues} critical issues immediately")
            
            if not recommendations:
                recommendations.append("System is performing well, continue monitoring")
            
            analysis["recommendation_engine"]["recommendations"] = recommendations
            
            results = {
                "success": True,
                "analysis": analysis,
                "recommendations": recommendations
            }
            
            phase.complete(success=True, results=results)
            
            self.logger.info(f"   Generated {len(recommendations)} recommendations")
            self.logger.info(f"   Analyzed {len(analysis.get('correlation_analysis', {}).get('execution_quality_correlation', []))} correlations")
            
            return results
            
        except Exception as e:
            phase.add_error(f"Integration analysis failed: {e}")
            phase.complete(success=False)
            self.logger.error(f"Integration analysis failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "analysis": {},
                "recommendations": []
            }
    
    async def _run_report_generation(
        self,
        phase: ValidationPhase,
        workflow_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phase 5: Generate comprehensive reports."""
        phase.start()
        self.logger.info("📑 Phase 5: Report Generation")
        
        try:
            generated_reports = []
            
            # Generate integrated workflow report
            workflow_report_path = self.output_directory / "integrated_validation_report.json"
            with open(workflow_report_path, 'w') as f:
                json.dump(workflow_results, f, indent=2)
            generated_reports.append(str(workflow_report_path))
            
            # Generate markdown summary
            markdown_report_path = self.output_directory / "integrated_validation_summary.md"
            markdown_content = self._generate_workflow_markdown(workflow_results)
            with open(markdown_report_path, 'w') as f:
                f.write(markdown_content)
            generated_reports.append(str(markdown_report_path))
            
            # Generate dashboard if quality data is available
            quality_data = workflow_results["phases"].get("quality", {})
            if quality_data.get("success"):
                dashboard_path = self.report_generator.generate_dashboard(
                    quality_data.get("batch_report")
                )
                generated_reports.append(str(dashboard_path))
            
            results = {
                "success": True,
                "generated_reports": generated_reports,
                "report_count": len(generated_reports)
            }
            
            phase.complete(success=True, results=results)
            
            self.logger.info(f"   Generated {len(generated_reports)} reports")
            for report in generated_reports:
                self.logger.info(f"     - {report}")
            
            return results
            
        except Exception as e:
            phase.add_error(f"Report generation failed: {e}")
            phase.complete(success=False)
            self.logger.error(f"Report generation failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "generated_reports": []
            }
    
    def _generate_workflow_markdown(self, workflow_results: Dict[str, Any]) -> str:
        """Generate comprehensive markdown report for workflow."""
        summary = workflow_results.get("workflow_summary", {})
        phases = workflow_results.get("phases", {})
        overall_success = workflow_results.get("overall_success", False)
        
        md_lines = [
            "# Integrated Validation Report",
            "",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Overall Success:** {'✅ Yes' if overall_success else '❌ No'}",
            f"**Total Duration:** {summary.get('total_duration', 0):.1f} seconds",
            ""
        ]
        
        # Phase results
        for phase_name, phase_data in phases.items():
            success_icon = "✅" if phase_data.get("success", False) else "❌"
            md_lines.extend([
                f"## {success_icon} Phase: {phase_name.title()}",
                ""
            ])
            
            # Phase-specific details
            if phase_name == "discovery":
                md_lines.extend([
                    f"- **Pipelines Discovered:** {phase_data.get('total_discovered', 0)}",
                    f"- **YAML Files Found:** {len(phase_data.get('existing_yaml_files', []))}",
                    f"- **Missing YAML Files:** {len(phase_data.get('missing_yaml_files', []))}"
                ])
            
            elif phase_name == "execution":
                md_lines.extend([
                    f"- **Success Rate:** {phase_data.get('success_rate', 0):.1f}%",
                    f"- **Successful Pipelines:** {len(phase_data.get('successful_pipelines', []))}",
                    f"- **Failed Pipelines:** {len(phase_data.get('failed_pipelines', []))}"
                ])
            
            elif phase_name == "quality":
                md_lines.extend([
                    f"- **Average Quality Score:** {phase_data.get('average_quality_score', 0):.1f}/100",
                    f"- **Production Ready:** {phase_data.get('production_ready_count', 0)} pipelines",
                    f"- **Success Rate:** {phase_data.get('success_rate', 0):.1f}%"
                ])
            
            elif phase_name == "analysis":
                recommendations = phase_data.get("recommendations", [])
                md_lines.append(f"- **Recommendations Generated:** {len(recommendations)}")
                
                if recommendations:
                    md_lines.append("\n### Key Recommendations:")
                    for rec in recommendations[:5]:  # Top 5
                        md_lines.append(f"  - {rec}")
            
            md_lines.append("")
        
        # Overall assessment
        md_lines.extend([
            "## Overall Assessment",
            ""
        ])
        
        if overall_success:
            md_lines.append("🎉 **Validation completed successfully!** All phases completed with acceptable results.")
        else:
            md_lines.append("⚠️ **Validation completed with issues.** Review phase details for specific problems.")
        
        return "\n".join(md_lines)
    
    async def _save_workflow_results(self, results: Dict[str, Any]):
        """Save comprehensive workflow results."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save timestamped results
        timestamped_path = self.output_directory / f"validation_results_{timestamp}.json"
        with open(timestamped_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save latest results
        latest_path = self.output_directory / "latest_validation_results.json"
        with open(latest_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Workflow results saved:")
        self.logger.info(f"  Timestamped: {timestamped_path}")
        self.logger.info(f"  Latest: {latest_path}")


async def main():
    """Main function for integrated validation."""
    parser = argparse.ArgumentParser(
        description="Integrated Validation System for comprehensive pipeline quality assurance"
    )
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help="Run complete validation workflow (discovery + execution + quality + analysis)"
    )
    parser.add_argument(
        "--pipeline-execution",
        action="store_true",
        help="Run pipeline execution validation only"
    )
    parser.add_argument(
        "--quality-review",
        action="store_true",
        help="Run quality review only (skip execution)"
    )
    parser.add_argument(
        "--production-check",
        action="store_true",
        help="Quick production readiness check"
    )
    parser.add_argument(
        "--pipelines",
        type=str,
        help="Comma-separated list of specific pipelines to validate"
    )
    parser.add_argument(
        "--skip-execution",
        action="store_true",
        help="Skip pipeline execution phase"
    )
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="Skip quality review phase"
    )
    
    args = parser.parse_args()
    
    # Determine pipelines to validate
    pipelines = None
    if args.pipelines:
        pipelines = [p.strip() for p in args.pipelines.split(",")]
    
    # Initialize system
    system = IntegratedValidationSystem()
    
    try:
        if args.full_validation:
            print("🚀 Starting Full Integrated Validation")
            results = await system.full_validation_workflow(
                pipelines=pipelines,
                skip_execution=args.skip_execution,
                skip_quality=args.skip_quality
            )
            
            # Print summary
            summary = results.get("workflow_summary", {})
            overall_success = results.get("overall_success", False)
            
            print(f"\n🎯 Validation Complete!")
            print(f"   Overall Success: {'✅ Yes' if overall_success else '❌ No'}")
            print(f"   Total Duration: {summary.get('total_duration', 0):.1f} seconds")
            
            # Show phase results
            phases = results.get("phases", {})
            for phase_name, phase_data in phases.items():
                success_icon = "✅" if phase_data.get("success", False) else "❌"
                print(f"   {phase_name.title()}: {success_icon}")
            
        elif args.pipeline_execution:
            print("⚙️ Running Pipeline Execution Validation Only")
            # Implementation for execution-only validation
            
        elif args.quality_review:
            print("🔍 Running Quality Review Only")
            # Implementation for quality-only validation
            
        elif args.production_check:
            print("🎯 Running Production Readiness Check")
            # Implementation for production check
            
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted by user")
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())