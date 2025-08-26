"""Test result reporting and analysis for pipeline testing."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .pipeline_test_suite import TestResults, PipelineTestResult

logger = logging.getLogger(__name__)


class PipelineTestReporter:
    """
    Comprehensive test result reporting and analysis.
    
    Generates detailed reports in multiple formats:
    - JSON for programmatic access
    - Markdown for human readability
    - Summary statistics for CI/CD integration
    - Trend analysis for performance monitoring
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize test reporter.
        
        Args:
            output_dir: Directory for report outputs
        """
        self.output_dir = output_dir or Path("test_results")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized test reporter, output dir: {self.output_dir}")
    
    def generate_comprehensive_report(self, 
                                    results: TestResults,
                                    test_mode: str = "full",
                                    report_name: Optional[str] = None) -> Dict[str, Path]:
        """
        Generate comprehensive test report in multiple formats.
        
        Args:
            results: Test results to report
            test_mode: Test mode that was run
            report_name: Custom report name (auto-generated if None)
            
        Returns:
            Dict[str, Path]: Mapping of report types to file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = report_name or f"pipeline_test_report_{test_mode}_{timestamp}"
        
        reports = {}
        
        # Generate JSON report
        json_path = self.output_dir / f"{report_name}.json"
        self._generate_json_report(results, json_path, test_mode)
        reports["json"] = json_path
        
        # Generate Markdown report
        md_path = self.output_dir / f"{report_name}.md"
        self._generate_markdown_report(results, md_path, test_mode)
        reports["markdown"] = md_path
        
        # Generate summary for CI/CD
        summary_path = self.output_dir / f"{report_name}_summary.json"
        self._generate_ci_summary(results, summary_path, test_mode)
        reports["summary"] = summary_path
        
        # Generate detailed analysis
        analysis_path = self.output_dir / f"{report_name}_analysis.md"
        self._generate_detailed_analysis(results, analysis_path, test_mode)
        reports["analysis"] = analysis_path
        
        logger.info(f"Generated {len(reports)} reports for {results.total_tests} pipeline tests")
        return reports
    
    def _generate_json_report(self, 
                            results: TestResults, 
                            output_path: Path,
                            test_mode: str):
        """Generate JSON report for programmatic access."""
        
        # Convert results to serializable format
        report_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "test_mode": test_mode,
                "total_tests": results.total_tests,
                "successful_tests": results.successful_tests,
                "failed_tests": results.failed_tests,
                "success_rate": results.success_rate,
                "total_time": results.total_time,
                "total_cost": results.total_cost,
                "average_quality_score": results.average_quality_score
            },
            "summary": {
                "failed_pipelines": results.get_failed_pipelines(),
                "high_quality_pipelines": results.get_high_quality_pipelines(),
                "execution_stats": {
                    "avg_execution_time": self._calculate_avg_execution_time(results),
                    "avg_cost": results.total_cost / results.total_tests if results.total_tests > 0 else 0,
                    "cost_distribution": self._calculate_cost_distribution(results),
                    "time_distribution": self._calculate_time_distribution(results)
                }
            },
            "detailed_results": {}
        }
        
        # Add detailed results for each pipeline
        for pipeline_name, result in results.results.items():
            report_data["detailed_results"][pipeline_name] = {
                "overall_success": result.overall_success,
                "quality_score": result.quality_score,
                "test_duration": result.test_duration,
                "execution": {
                    "success": result.execution.success,
                    "execution_time": result.execution.execution_time,
                    "estimated_cost": result.execution.estimated_cost,
                    "api_calls_count": result.execution.api_calls_count,
                    "tokens_used": result.execution.tokens_used,
                    "error_message": result.execution.error_message
                },
                "templates": {
                    "resolved_correctly": result.templates.resolved_correctly,
                    "issues_count": len(result.templates.issues),
                    "issues": result.templates.issues
                },
                "organization": {
                    "valid": result.organization.valid,
                    "issues_count": len(result.organization.issues),
                    "output_files_count": len(result.organization.output_files_found)
                },
                "performance": {
                    "performance_score": result.performance.performance_score,
                    "regression_detected": result.performance.regression_detected,
                    "metrics": result.performance.metrics
                },
                "warnings": result.warnings
            }
        
        # Write JSON report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Generated JSON report: {output_path}")
    
    def _generate_markdown_report(self, 
                                results: TestResults, 
                                output_path: Path,
                                test_mode: str):
        """Generate human-readable Markdown report."""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        md_content = f"""# Pipeline Test Report - {test_mode.title()} Mode

**Generated:** {timestamp}

## Executive Summary

- **Total Tests:** {results.total_tests}
- **Successful:** {results.successful_tests} ({results.success_rate:.1f}%)
- **Failed:** {results.failed_tests}
- **Total Duration:** {results.total_time:.1f} seconds
- **Total Cost:** ${results.total_cost:.4f}
- **Average Quality Score:** {results.average_quality_score:.1f}/100

## Results Overview

"""
        
        if results.failed_tests > 0:
            md_content += f"### ❌ Failed Pipelines ({results.failed_tests})\n\n"
            failed_pipelines = results.get_failed_pipelines()
            for pipeline_name in failed_pipelines:
                result = results.results[pipeline_name]
                md_content += f"- **{pipeline_name}**\n"
                if result.execution.error_message:
                    md_content += f"  - Error: {result.execution.error_message}\n"
                md_content += f"  - Quality Score: {result.quality_score:.1f}/100\n"
                if result.templates.issues:
                    md_content += f"  - Template Issues: {len(result.templates.issues)}\n"
                if result.organization.issues:
                    md_content += f"  - Organization Issues: {len(result.organization.issues)}\n"
                md_content += "\n"
        
        if results.successful_tests > 0:
            md_content += f"### ✅ Successful Pipelines ({results.successful_tests})\n\n"
            successful_pipelines = [name for name, result in results.results.items() 
                                  if result.overall_success]
            
            # Group by quality score
            high_quality = [name for name in successful_pipelines 
                          if results.results[name].quality_score >= 90]
            good_quality = [name for name in successful_pipelines 
                          if 70 <= results.results[name].quality_score < 90]
            acceptable_quality = [name for name in successful_pipelines 
                                if results.results[name].quality_score < 70]
            
            if high_quality:
                md_content += f"#### High Quality (90-100): {len(high_quality)} pipelines\n"
                for name in high_quality[:10]:  # Show first 10
                    result = results.results[name]
                    md_content += f"- {name} (Score: {result.quality_score:.1f}, Time: {result.execution.execution_time:.1f}s)\n"
                if len(high_quality) > 10:
                    md_content += f"- ...and {len(high_quality) - 10} more\n"
                md_content += "\n"
            
            if good_quality:
                md_content += f"#### Good Quality (70-89): {len(good_quality)} pipelines\n"
                for name in good_quality[:5]:  # Show first 5
                    result = results.results[name]
                    md_content += f"- {name} (Score: {result.quality_score:.1f})\n"
                if len(good_quality) > 5:
                    md_content += f"- ...and {len(good_quality) - 5} more\n"
                md_content += "\n"
            
            if acceptable_quality:
                md_content += f"#### Acceptable Quality (<70): {len(acceptable_quality)} pipelines\n"
                md_content += "These pipelines passed but may need improvement.\n\n"
        
        # Performance Analysis
        md_content += "## Performance Analysis\n\n"
        
        # Execution time analysis
        exec_times = [result.execution.execution_time for result in results.results.values() 
                     if result.execution.success]
        if exec_times:
            avg_time = sum(exec_times) / len(exec_times)
            max_time = max(exec_times)
            min_time = min(exec_times)
            
            md_content += f"### Execution Times\n"
            md_content += f"- Average: {avg_time:.1f}s\n"
            md_content += f"- Range: {min_time:.1f}s - {max_time:.1f}s\n"
            
            # Find slowest pipelines
            slow_pipelines = [(name, result.execution.execution_time) 
                            for name, result in results.results.items()
                            if result.execution.success and result.execution.execution_time > avg_time * 2]
            
            if slow_pipelines:
                slow_pipelines.sort(key=lambda x: x[1], reverse=True)
                md_content += f"\n#### Slowest Pipelines (>{avg_time*2:.1f}s)\n"
                for name, time in slow_pipelines[:5]:
                    md_content += f"- {name}: {time:.1f}s\n"
            
            md_content += "\n"
        
        # Cost analysis
        costs = [result.execution.estimated_cost for result in results.results.values()
                if result.execution.success and result.execution.estimated_cost > 0]
        if costs:
            avg_cost = sum(costs) / len(costs)
            max_cost = max(costs)
            total_cost = sum(costs)
            
            md_content += f"### Cost Analysis\n"
            md_content += f"- Total Cost: ${total_cost:.4f}\n"
            md_content += f"- Average per Pipeline: ${avg_cost:.4f}\n"
            md_content += f"- Most Expensive: ${max_cost:.4f}\n"
            
            # Find most expensive pipelines
            expensive_pipelines = [(name, result.execution.estimated_cost)
                                 for name, result in results.results.items()
                                 if result.execution.success and result.execution.estimated_cost > avg_cost * 2]
            
            if expensive_pipelines:
                expensive_pipelines.sort(key=lambda x: x[1], reverse=True)
                md_content += f"\n#### Most Expensive Pipelines (>${avg_cost*2:.4f}+)\n"
                for name, cost in expensive_pipelines[:5]:
                    md_content += f"- {name}: ${cost:.4f}\n"
            
            md_content += "\n"
        
        # Issues Summary
        template_issues = sum(1 for result in results.results.values() 
                            if not result.templates.resolved_correctly)
        org_issues = sum(1 for result in results.results.values()
                        if not result.organization.valid)
        perf_regressions = sum(1 for result in results.results.values()
                             if result.performance.regression_detected)
        
        if template_issues or org_issues or perf_regressions:
            md_content += "## Issues Summary\n\n"
            if template_issues:
                md_content += f"- **Template Issues:** {template_issues} pipelines\n"
            if org_issues:
                md_content += f"- **Organization Issues:** {org_issues} pipelines\n"
            if perf_regressions:
                md_content += f"- **Performance Regressions:** {perf_regressions} pipelines\n"
            md_content += "\n"
        
        # Recommendations
        md_content += "## Recommendations\n\n"
        
        if results.success_rate < 80:
            md_content += "- **Low Success Rate:** Consider investigating common failure patterns and improving error handling.\n"
        
        if results.average_quality_score < 75:
            md_content += "- **Quality Improvement:** Focus on template resolution and file organization issues.\n"
        
        if results.total_cost > 10.0:
            md_content += "- **Cost Optimization:** Consider using more cost-effective models for testing.\n"
        
        if results.total_time > 1800:  # 30 minutes
            md_content += "- **Performance:** Test suite is taking a long time - consider optimizing pipeline timeouts.\n"
        
        md_content += f"\n---\n*Report generated by Pipeline Testing Infrastructure v1.0*\n"
        
        # Write Markdown report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Generated Markdown report: {output_path}")
    
    def _generate_ci_summary(self, 
                           results: TestResults, 
                           output_path: Path,
                           test_mode: str):
        """Generate CI/CD integration summary."""
        
        summary = {
            "test_mode": test_mode,
            "timestamp": datetime.now().isoformat(),
            "status": "PASS" if results.success_rate >= 80 else "FAIL",
            "total_tests": results.total_tests,
            "passed": results.successful_tests,
            "failed": results.failed_tests,
            "success_rate": results.success_rate,
            "duration_seconds": results.total_time,
            "total_cost": results.total_cost,
            "quality_score": results.average_quality_score,
            "failed_pipelines": results.get_failed_pipelines(),
            "issues": {
                "template_failures": sum(1 for r in results.results.values() 
                                       if not r.templates.resolved_correctly),
                "organization_failures": sum(1 for r in results.results.values()
                                           if not r.organization.valid),
                "performance_regressions": sum(1 for r in results.results.values()
                                             if r.performance.regression_detected)
            },
            "thresholds": {
                "min_success_rate": 80.0,
                "min_quality_score": 70.0,
                "max_total_cost": 50.0,
                "max_duration": 3600  # 1 hour
            },
            "recommendations": self._generate_recommendations(results)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Generated CI summary: {output_path}")
    
    def _generate_detailed_analysis(self, 
                                  results: TestResults, 
                                  output_path: Path,
                                  test_mode: str):
        """Generate detailed analysis report."""
        
        analysis_content = f"""# Detailed Pipeline Analysis - {test_mode.title()} Mode

## Test Suite Metrics

"""
        
        # Calculate detailed metrics
        execution_times = []
        costs = []
        quality_scores = []
        
        for result in results.results.values():
            if result.execution.success:
                execution_times.append(result.execution.execution_time)
                costs.append(result.execution.estimated_cost)
            quality_scores.append(result.quality_score)
        
        if execution_times:
            analysis_content += f"""### Performance Distribution

- **Execution Time Statistics:**
  - Mean: {sum(execution_times) / len(execution_times):.2f}s
  - Median: {sorted(execution_times)[len(execution_times)//2]:.2f}s
  - 95th Percentile: {sorted(execution_times)[int(len(execution_times)*0.95)]:.2f}s
  - Standard Deviation: {self._calculate_std_dev(execution_times):.2f}s

"""
        
        if costs:
            total_cost = sum(costs)
            analysis_content += f"""### Cost Analysis

- **Cost Statistics:**
  - Total: ${total_cost:.4f}
  - Mean per Pipeline: ${sum(costs) / len(costs):.4f}
  - Median: ${sorted(costs)[len(costs)//2]:.4f}
  - 95th Percentile: ${sorted(costs)[int(len(costs)*0.95)]:.4f}

"""
        
        # Quality score distribution
        if quality_scores:
            high_quality_count = sum(1 for score in quality_scores if score >= 90)
            good_quality_count = sum(1 for score in quality_scores if 70 <= score < 90)
            poor_quality_count = sum(1 for score in quality_scores if score < 70)
            
            analysis_content += f"""### Quality Score Distribution

- **High Quality (90-100):** {high_quality_count} ({high_quality_count/len(quality_scores)*100:.1f}%)
- **Good Quality (70-89):** {good_quality_count} ({good_quality_count/len(quality_scores)*100:.1f}%)
- **Poor Quality (<70):** {poor_quality_count} ({poor_quality_count/len(quality_scores)*100:.1f}%)

"""
        
        # Failure analysis
        failed_pipelines = results.get_failed_pipelines()
        if failed_pipelines:
            analysis_content += f"### Failure Analysis\n\n"
            
            # Categorize failures
            execution_failures = []
            template_failures = []
            organization_failures = []
            
            for name in failed_pipelines:
                result = results.results[name]
                if not result.execution.success:
                    execution_failures.append((name, result.execution.error_message))
                if not result.templates.resolved_correctly:
                    template_failures.append((name, result.templates.issues))
                if not result.organization.valid:
                    organization_failures.append((name, result.organization.issues))
            
            if execution_failures:
                analysis_content += f"#### Execution Failures ({len(execution_failures)})\n\n"
                for name, error in execution_failures[:10]:  # Show first 10
                    analysis_content += f"- **{name}:** {error}\n"
                analysis_content += "\n"
            
            if template_failures:
                analysis_content += f"#### Template Resolution Issues ({len(template_failures)})\n\n"
                for name, issues in template_failures[:5]:  # Show first 5
                    analysis_content += f"- **{name}:** {len(issues)} issues\n"
                analysis_content += "\n"
        
        # Write detailed analysis
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(analysis_content)
        
        logger.info(f"Generated detailed analysis: {output_path}")
    
    def _calculate_avg_execution_time(self, results: TestResults) -> float:
        """Calculate average execution time for successful tests."""
        successful_times = [result.execution.execution_time for result in results.results.values()
                          if result.execution.success]
        return sum(successful_times) / len(successful_times) if successful_times else 0.0
    
    def _calculate_cost_distribution(self, results: TestResults) -> Dict[str, int]:
        """Calculate cost distribution buckets."""
        costs = [result.execution.estimated_cost for result in results.results.values()
                if result.execution.success]
        
        distribution = {
            "free": sum(1 for cost in costs if cost == 0),
            "low": sum(1 for cost in costs if 0 < cost <= 0.01),
            "medium": sum(1 for cost in costs if 0.01 < cost <= 0.10),
            "high": sum(1 for cost in costs if cost > 0.10)
        }
        
        return distribution
    
    def _calculate_time_distribution(self, results: TestResults) -> Dict[str, int]:
        """Calculate execution time distribution buckets."""
        times = [result.execution.execution_time for result in results.results.values()
                if result.execution.success]
        
        distribution = {
            "fast": sum(1 for time in times if time <= 30),
            "medium": sum(1 for time in times if 30 < time <= 120),
            "slow": sum(1 for time in times if time > 120)
        }
        
        return distribution
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _generate_recommendations(self, results: TestResults) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []
        
        if results.success_rate < 70:
            recommendations.append("Critical: Low success rate indicates systematic issues")
        elif results.success_rate < 85:
            recommendations.append("Investigate common failure patterns")
        
        if results.average_quality_score < 70:
            recommendations.append("Focus on improving template resolution and file organization")
        
        if results.total_cost > 20.0:
            recommendations.append("Consider optimizing test costs by using smaller models")
        
        if results.total_time > 3600:  # 1 hour
            recommendations.append("Test suite taking too long - optimize timeouts and parallelization")
        
        template_issues = sum(1 for r in results.results.values() 
                            if not r.templates.resolved_correctly)
        if template_issues > results.total_tests * 0.1:
            recommendations.append("High rate of template issues - review template syntax")
        
        return recommendations