"""
CI/CD integration utilities for pipeline testing infrastructure.

This module provides integration capabilities for CI/CD systems including GitHub Actions,
build system integration, status reporting, and artifact generation.
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

logger = logging.getLogger(__name__)


class CISystem(Enum):
    """Supported CI/CD systems."""
    GITHUB_ACTIONS = "github_actions"
    JENKINS = "jenkins"
    GITLAB_CI = "gitlab_ci"
    AZURE_DEVOPS = "azure_devops"
    GENERIC = "generic"


class TestStatus(Enum):
    """Test execution status for CI/CD systems."""
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning" 
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class CITestResult:
    """CI/CD-friendly test result representation."""
    
    pipeline_name: str
    status: TestStatus
    execution_time_seconds: float
    success_rate: float
    quality_score: float
    error_message: Optional[str] = None
    warning_messages: List[str] = None
    artifacts: List[str] = None
    
    def __post_init__(self):
        if self.warning_messages is None:
            self.warning_messages = []
        if self.artifacts is None:
            self.artifacts = []


@dataclass
class CITestSummary:
    """Overall test suite summary for CI/CD systems."""
    
    total_tests: int
    successful_tests: int
    failed_tests: int
    warning_tests: int
    skipped_tests: int
    total_time_seconds: float
    success_rate: float
    average_quality_score: float
    total_cost_dollars: float
    critical_failures: List[str]
    quality_gate_passed: bool
    release_ready: bool
    artifacts: List[str] = None
    
    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []


@dataclass
class CIConfiguration:
    """Configuration for CI/CD integration."""
    
    system: CISystem
    environment: str  # dev, staging, prod
    branch: str
    commit_sha: Optional[str]
    pull_request_number: Optional[int]
    
    # Quality gates
    min_success_rate: float = 0.8
    min_quality_score: float = 85.0
    max_execution_time_minutes: int = 120
    max_cost_dollars: float = 5.0
    
    # Reporting
    generate_artifacts: bool = True
    artifact_retention_days: int = 30
    enable_detailed_reports: bool = True
    enable_performance_reports: bool = True
    
    # Integration settings
    fail_fast: bool = False
    parallel_execution: bool = True
    retry_failed_tests: bool = True
    max_retries: int = 1


class CIIntegrationManager:
    """
    Manages CI/CD system integration for pipeline testing.
    
    Features:
    - Multi-CI system support
    - Quality gate enforcement
    - Status reporting and exit codes
    - Artifact generation and management
    - Performance tracking integration
    - Release validation
    """
    
    def __init__(self, config: CIConfiguration):
        """Initialize CI integration manager."""
        self.config = config
        self.ci_environment = self._detect_ci_environment()
        self.artifacts_dir = Path("ci_artifacts")
        self.artifacts_dir.mkdir(exist_ok=True)
        
    def _detect_ci_environment(self) -> Dict[str, Any]:
        """Detect current CI/CD environment and extract metadata."""
        env_info = {
            "system": self.config.system.value,
            "detected_system": "unknown",
            "branch": self.config.branch,
            "commit": self.config.commit_sha,
            "build_number": None,
            "runner_id": None
        }
        
        # GitHub Actions detection
        if os.getenv("GITHUB_ACTIONS"):
            env_info.update({
                "detected_system": "github_actions",
                "branch": os.getenv("GITHUB_REF_NAME", self.config.branch),
                "commit": os.getenv("GITHUB_SHA", self.config.commit_sha),
                "build_number": os.getenv("GITHUB_RUN_NUMBER"),
                "runner_id": os.getenv("RUNNER_NAME"),
                "workflow": os.getenv("GITHUB_WORKFLOW"),
                "action": os.getenv("GITHUB_ACTION"),
                "repository": os.getenv("GITHUB_REPOSITORY")
            })
            
        # Jenkins detection
        elif os.getenv("JENKINS_URL"):
            env_info.update({
                "detected_system": "jenkins",
                "build_number": os.getenv("BUILD_NUMBER"),
                "job_name": os.getenv("JOB_NAME"),
                "build_url": os.getenv("BUILD_URL"),
                "workspace": os.getenv("WORKSPACE")
            })
            
        # GitLab CI detection
        elif os.getenv("GITLAB_CI"):
            env_info.update({
                "detected_system": "gitlab_ci",
                "branch": os.getenv("CI_COMMIT_REF_NAME", self.config.branch),
                "commit": os.getenv("CI_COMMIT_SHA", self.config.commit_sha),
                "build_number": os.getenv("CI_PIPELINE_ID"),
                "job_name": os.getenv("CI_JOB_NAME"),
                "runner_id": os.getenv("CI_RUNNER_ID")
            })
            
        # Azure DevOps detection
        elif os.getenv("AZURE_HTTP_USER_AGENT"):
            env_info.update({
                "detected_system": "azure_devops", 
                "build_number": os.getenv("BUILD_BUILDNUMBER"),
                "branch": os.getenv("BUILD_SOURCEBRANCHNAME", self.config.branch),
                "commit": os.getenv("BUILD_SOURCEVERSION", self.config.commit_sha)
            })
            
        return env_info
    
    def convert_test_results_to_ci_format(self, test_results) -> Tuple[List[CITestResult], CITestSummary]:
        """
        Convert pipeline test results to CI-friendly format.
        
        Args:
            test_results: TestResults object from pipeline test suite
            
        Returns:
            Tuple of (individual results, overall summary)
        """
        ci_results = []
        critical_failures = []
        
        for pipeline_name, result in test_results.results.items():
            # Determine status
            if result.execution.success:
                if result.quality_score < self.config.min_quality_score:
                    status = TestStatus.WARNING
                else:
                    status = TestStatus.SUCCESS
            else:
                status = TestStatus.FAILURE
                if result.execution.is_timeout:
                    critical_failures.append(f"{pipeline_name}: Timeout")
                elif result.execution.error_message:
                    critical_failures.append(f"{pipeline_name}: {result.execution.error_message}")
                else:
                    critical_failures.append(f"{pipeline_name}: Unknown error")
            
            # Collect warnings
            warnings = []
            if result.templates and not result.templates.resolved_correctly:
                warnings.extend([f"Template issue: {issue}" for issue in result.templates.issues])
            if result.organization and not result.organization.valid:
                warnings.extend([f"Organization issue: {issue}" for issue in result.organization.issues])
            if hasattr(result, 'quality') and result.quality and result.quality.recommendations:
                warnings.extend(result.quality.recommendations[:3])  # Limit warnings
            
            # Collect artifacts
            artifacts = []
            if result.execution.output_files:
                artifacts.extend(result.execution.output_files)
            
            ci_result = CITestResult(
                pipeline_name=pipeline_name,
                status=status,
                execution_time_seconds=result.execution.execution_time or 0.0,
                success_rate=1.0 if result.execution.success else 0.0,
                quality_score=result.quality_score,
                error_message=result.execution.error_message,
                warning_messages=warnings,
                artifacts=artifacts
            )
            ci_results.append(ci_result)
        
        # Quality gate evaluation
        success_rate = test_results.success_rate / 100.0  # Convert percentage
        quality_gate_passed = (
            success_rate >= self.config.min_success_rate and
            test_results.average_quality_score >= self.config.min_quality_score and
            test_results.total_time <= (self.config.max_execution_time_minutes * 60) and
            test_results.total_cost <= self.config.max_cost_dollars
        )
        
        # Release readiness (stricter criteria)
        release_ready = (
            quality_gate_passed and
            success_rate >= 0.95 and  # Higher success rate for releases
            len(critical_failures) == 0 and
            test_results.average_quality_score >= 90.0  # Higher quality for releases
        )
        
        summary = CITestSummary(
            total_tests=test_results.total_tests,
            successful_tests=test_results.successful_tests,
            failed_tests=test_results.failed_tests,
            warning_tests=len([r for r in ci_results if r.status == TestStatus.WARNING]),
            skipped_tests=0,  # Not currently supported
            total_time_seconds=test_results.total_time,
            success_rate=success_rate,
            average_quality_score=test_results.average_quality_score,
            total_cost_dollars=test_results.total_cost,
            critical_failures=critical_failures,
            quality_gate_passed=quality_gate_passed,
            release_ready=release_ready
        )
        
        return ci_results, summary
    
    def generate_ci_artifacts(
        self, 
        ci_results: List[CITestResult], 
        summary: CITestSummary,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> List[Path]:
        """Generate CI/CD artifacts including reports and data files."""
        artifacts = []
        timestamp = int(time.time()) if 'time' in sys.modules else 0
        
        # 1. Generate JSON summary for programmatic consumption
        summary_data = {
            "summary": asdict(summary),
            "results": [asdict(result) for result in ci_results],
            "environment": self.ci_environment,
            "configuration": asdict(self.config),
            "timestamp": timestamp
        }
        
        if additional_data:
            summary_data.update(additional_data)
        
        json_file = self.artifacts_dir / f"test_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        artifacts.append(json_file)
        
        # 2. Generate JUnit XML for CI system integration
        junit_file = self.artifacts_dir / f"junit_results_{timestamp}.xml"
        self._generate_junit_xml(ci_results, summary, junit_file)
        artifacts.append(junit_file)
        
        # 3. Generate human-readable summary
        summary_file = self.artifacts_dir / f"test_summary_{timestamp}.md"
        self._generate_markdown_summary(ci_results, summary, summary_file)
        artifacts.append(summary_file)
        
        # 4. Generate CI-specific status files
        if self.config.system == CISystem.GITHUB_ACTIONS:
            self._generate_github_actions_outputs(summary)
        
        logger.info(f"Generated {len(artifacts)} CI artifacts in {self.artifacts_dir}")
        return artifacts
    
    def _generate_junit_xml(self, results: List[CITestResult], summary: CITestSummary, output_file: Path):
        """Generate JUnit XML format for CI system integration."""
        from xml.etree.ElementTree import Element, SubElement, tostring
        import xml.dom.minidom
        
        testsuites = Element("testsuites")
        testsuites.set("name", "Pipeline Tests")
        testsuites.set("tests", str(summary.total_tests))
        testsuites.set("failures", str(summary.failed_tests))
        testsuites.set("errors", "0")
        testsuites.set("time", str(summary.total_time_seconds))
        
        testsuite = SubElement(testsuites, "testsuite")
        testsuite.set("name", "Pipeline Execution Tests")
        testsuite.set("tests", str(summary.total_tests))
        testsuite.set("failures", str(summary.failed_tests))
        testsuite.set("time", str(summary.total_time_seconds))
        
        for result in results:
            testcase = SubElement(testsuite, "testcase")
            testcase.set("classname", "orchestrator.pipeline_tests")
            testcase.set("name", result.pipeline_name)
            testcase.set("time", str(result.execution_time_seconds))
            
            if result.status == TestStatus.FAILURE:
                failure = SubElement(testcase, "failure")
                failure.set("message", result.error_message or "Pipeline execution failed")
                failure.text = result.error_message or "Pipeline execution failed"
                
            elif result.status == TestStatus.WARNING:
                # JUnit doesn't have warnings, so add as system-out
                system_out = SubElement(testcase, "system-out")
                system_out.text = "\n".join(result.warning_messages)
        
        # Format XML
        rough_string = tostring(testsuites, 'unicode')
        reparsed = xml.dom.minidom.parseString(rough_string)
        pretty = reparsed.toprettyxml(indent="  ")
        
        with open(output_file, 'w') as f:
            f.write(pretty)
    
    def _generate_markdown_summary(self, results: List[CITestResult], summary: CITestSummary, output_file: Path):
        """Generate human-readable markdown summary."""
        content = [
            "# Pipeline Test Results Summary",
            f"",
            f"**Environment:** {self.config.environment}",
            f"**Branch:** {self.ci_environment.get('branch', 'unknown')}",
            f"**Commit:** {self.ci_environment.get('commit', 'unknown')[:8]}",
            f"**Timestamp:** {summary.artifacts[0] if summary.artifacts else 'N/A'}",
            f"",
            "## Overall Results",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Tests | {summary.total_tests} |",
            f"| Success Rate | {summary.success_rate:.1%} |",
            f"| Average Quality | {summary.average_quality_score:.1f}/100 |",
            f"| Total Time | {summary.total_time_seconds/60:.1f} minutes |",
            f"| Total Cost | ${summary.total_cost_dollars:.4f} |",
            f"| Quality Gate | {'✅ PASSED' if summary.quality_gate_passed else '❌ FAILED'} |",
            f"| Release Ready | {'✅ YES' if summary.release_ready else '❌ NO'} |",
            f""
        ]
        
        # Status breakdown
        if summary.failed_tests > 0 or summary.warning_tests > 0:
            content.extend([
                "## Test Status Breakdown",
                "",
                f"- ✅ Successful: {summary.successful_tests}",
                f"- ❌ Failed: {summary.failed_tests}",
                f"- ⚠️ Warnings: {summary.warning_tests}",
                ""
            ])
        
        # Critical failures
        if summary.critical_failures:
            content.extend([
                "## Critical Failures",
                ""
            ])
            for failure in summary.critical_failures:
                content.append(f"- ❌ {failure}")
            content.append("")
        
        # Individual results
        content.extend([
            "## Individual Pipeline Results",
            "",
            "| Pipeline | Status | Time | Quality | Issues |",
            "|----------|--------|------|---------|--------|"
        ])
        
        for result in sorted(results, key=lambda x: x.status.value):
            status_emoji = {
                TestStatus.SUCCESS: "✅",
                TestStatus.WARNING: "⚠️", 
                TestStatus.FAILURE: "❌",
                TestStatus.ERROR: "💥",
                TestStatus.SKIPPED: "⏭️"
            }.get(result.status, "❓")
            
            issues = result.error_message or ("; ".join(result.warning_messages[:2]) if result.warning_messages else "None")
            if len(issues) > 50:
                issues = issues[:47] + "..."
                
            content.append(
                f"| {result.pipeline_name} | {status_emoji} {result.status.value} | "
                f"{result.execution_time_seconds:.1f}s | {result.quality_score:.1f} | {issues} |"
            )
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(content))
    
    def _generate_github_actions_outputs(self, summary: CITestSummary):
        """Generate GitHub Actions specific outputs."""
        if not os.getenv("GITHUB_OUTPUT"):
            return
            
        # Set GitHub Actions outputs
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"success_rate={summary.success_rate:.3f}\n")
            f.write(f"quality_score={summary.average_quality_score:.1f}\n")
            f.write(f"quality_gate_passed={str(summary.quality_gate_passed).lower()}\n")
            f.write(f"release_ready={str(summary.release_ready).lower()}\n")
            f.write(f"total_tests={summary.total_tests}\n")
            f.write(f"failed_tests={summary.failed_tests}\n")
            f.write(f"execution_time={summary.total_time_seconds:.1f}\n")
    
    def determine_exit_code(self, summary: CITestSummary) -> int:
        """
        Determine appropriate exit code for CI/CD system.
        
        Exit codes:
        - 0: Success (all tests passed, quality gates met)
        - 1: Test failures (some pipelines failed)
        - 2: Quality gate failures (tests passed but quality/performance issues)
        - 3: Configuration/setup errors
        - 4: Critical system errors
        """
        if summary.failed_tests > 0:
            logger.error(f"Pipeline tests failed: {summary.failed_tests}/{summary.total_tests}")
            return 1
            
        if not summary.quality_gate_passed:
            logger.warning("Quality gates not met")
            if summary.success_rate < 0.5:
                return 1  # Too many failures
            else:
                return 2  # Quality/performance issues
        
        if summary.warning_tests > summary.total_tests * 0.3:  # > 30% warnings
            logger.warning(f"High warning rate: {summary.warning_tests}/{summary.total_tests}")
            return 2
            
        logger.info("All pipeline tests passed successfully")
        return 0
    
    def create_ci_status_check(
        self, 
        summary: CITestSummary, 
        context: str = "pipeline-tests"
    ) -> Dict[str, Any]:
        """Create CI status check information."""
        if summary.quality_gate_passed:
            state = "success"
            description = f"✅ {summary.successful_tests}/{summary.total_tests} tests passed"
        elif summary.success_rate >= 0.8:
            state = "success" 
            description = f"⚠️ {summary.successful_tests}/{summary.total_tests} tests passed (warnings)"
        else:
            state = "failure"
            description = f"❌ {summary.failed_tests}/{summary.total_tests} tests failed"
        
        return {
            "context": context,
            "state": state,
            "description": description,
            "target_url": None,  # Could link to detailed report
            "details": {
                "success_rate": summary.success_rate,
                "quality_score": summary.average_quality_score,
                "execution_time": summary.total_time_seconds,
                "quality_gate_passed": summary.quality_gate_passed,
                "release_ready": summary.release_ready
            }
        }


def create_ci_config_from_environment() -> CIConfiguration:
    """Create CI configuration by detecting current environment."""
    # Detect CI system
    if os.getenv("GITHUB_ACTIONS"):
        system = CISystem.GITHUB_ACTIONS
        branch = os.getenv("GITHUB_REF_NAME", "main")
        commit = os.getenv("GITHUB_SHA")
        pr_number = None
        if os.getenv("GITHUB_EVENT_NAME") == "pull_request":
            event_path = os.getenv("GITHUB_EVENT_PATH")
            if event_path and os.path.exists(event_path):
                try:
                    with open(event_path) as f:
                        event_data = json.load(f)
                        pr_number = event_data.get("number")
                except Exception:
                    pass
    elif os.getenv("JENKINS_URL"):
        system = CISystem.JENKINS
        branch = os.getenv("GIT_BRANCH", "main")
        commit = os.getenv("GIT_COMMIT")
        pr_number = None
    elif os.getenv("GITLAB_CI"):
        system = CISystem.GITLAB_CI
        branch = os.getenv("CI_COMMIT_REF_NAME", "main")
        commit = os.getenv("CI_COMMIT_SHA")
        pr_number = os.getenv("CI_MERGE_REQUEST_IID")
    else:
        system = CISystem.GENERIC
        branch = os.getenv("CI_BRANCH", "main")
        commit = os.getenv("CI_COMMIT")
        pr_number = None
    
    # Determine environment
    if branch in ["main", "master"]:
        environment = "prod"
    elif branch.startswith("release/") or branch.startswith("hotfix/"):
        environment = "staging"
    else:
        environment = "dev"
    
    return CIConfiguration(
        system=system,
        environment=environment,
        branch=branch,
        commit_sha=commit,
        pull_request_number=int(pr_number) if pr_number and pr_number.isdigit() else None
    )


if __name__ == "__main__":
    # Example usage
    import time
    
    print("CI/CD Integration Manager - Example Usage")
    print("=" * 50)
    
    # Create example configuration
    config = CIConfiguration(
        system=CISystem.GITHUB_ACTIONS,
        environment="dev",
        branch="feature/pipeline-tests",
        commit_sha="abc123def456"
    )
    
    manager = CIIntegrationManager(config)
    print(f"Detected CI environment: {manager.ci_environment['detected_system']}")
    print(f"Quality gate thresholds: success_rate >= {config.min_success_rate:.0%}, "
          f"quality >= {config.min_quality_score}")