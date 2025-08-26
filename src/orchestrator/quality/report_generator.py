"""
Comprehensive Quality Report Generation System

This module provides structured report generation capabilities for pipeline quality assessments,
including JSON, Markdown, HTML, and CSV formats with dashboard integration.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.quality_assessment import PipelineQualityReview, QualityIssue, IssueSeverity, IssueCategory


class ReportTemplate:
    """Templates for different report formats."""
    
    @staticmethod
    def get_html_template() -> str:
        """Get HTML template for quality reports."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Quality Report - {title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f7fa;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .header {{
            border-bottom: 3px solid #007acc;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .score-circle {{
            display: inline-block;
            width: 80px;
            height: 80px;
            border-radius: 50%;
            text-align: center;
            line-height: 80px;
            font-size: 24px;
            font-weight: bold;
            color: white;
            margin-right: 20px;
            vertical-align: middle;
        }}
        .score-excellent {{ background-color: #28a745; }}
        .score-good {{ background-color: #ffc107; color: #333; }}
        .score-fair {{ background-color: #fd7e14; }}
        .score-poor {{ background-color: #dc3545; }}
        .score-critical {{ background-color: #6f42c1; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #007acc;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #007acc;
        }}
        .metric-label {{
            color: #6c757d;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .issues-section {{
            margin-top: 40px;
        }}
        .issue-card {{
            background: #fff;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            margin-bottom: 15px;
            overflow: hidden;
        }}
        .issue-header {{
            padding: 15px;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .issue-critical {{ background-color: #f8d7da; color: #721c24; }}
        .issue-major {{ background-color: #fff3cd; color: #856404; }}
        .issue-minor {{ background-color: #d1ecf1; color: #0c5460; }}
        .issue-body {{
            padding: 15px;
            border-top: 1px solid #dee2e6;
            background-color: #fafafa;
        }}
        .files-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .files-table th, .files-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        .files-table th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .recommendations {{
            background: #e7f3ff;
            border: 1px solid #b8daff;
            border-radius: 8px;
            padding: 20px;
            margin-top: 30px;
        }}
        .recommendations h3 {{
            color: #004085;
            margin-top: 0;
        }}
        .status-badge {{
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .status-ready {{
            background-color: #d4edda;
            color: #155724;
        }}
        .status-not-ready {{
            background-color: #f8d7da;
            color: #721c24;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Pipeline Quality Report</h1>
            <h2>{pipeline_name}</h2>
            <p><strong>Generated:</strong> {timestamp}<br>
               <strong>Review Duration:</strong> {duration:.2f} seconds<br>
               <strong>Model Used:</strong> {model}</p>
        </div>
        
        <div style="margin-bottom: 30px;">
            <div class="score-circle score-{score_class}">{score}</div>
            <div style="display: inline-block; vertical-align: middle;">
                <h3 style="margin: 0;">Overall Quality Score</h3>
                <span class="status-badge status-{ready_class}">{ready_text}</span>
                <p style="margin: 5px 0 0 0; color: #6c757d;">{score_description}</p>
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{files_count}</div>
                <div class="metric-label">Files Reviewed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{total_issues}</div>
                <div class="metric-label">Total Issues</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{critical_count}</div>
                <div class="metric-label">Critical Issues</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{major_count}</div>
                <div class="metric-label">Major Issues</div>
            </div>
        </div>
        
        {issues_html}
        
        {recommendations_html}
        
        <div style="margin-top: 40px;">
            <h3>Files Reviewed</h3>
            <table class="files-table">
                <thead>
                    <tr>
                        <th>File Path</th>
                        <th>Type</th>
                    </tr>
                </thead>
                <tbody>
                    {files_html}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
        """
    
    @staticmethod
    def get_dashboard_template() -> str:
        """Get HTML template for quality dashboard."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Quality Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            background-color: #f5f7fa;
            color: #333;
        }}
        .dashboard-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .dashboard-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .chart-container {{
            position: relative;
            height: 300px;
            margin-top: 20px;
        }}
        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }}
        .pipeline-list {{
            max-height: 400px;
            overflow-y: auto;
        }}
        .pipeline-item {{
            padding: 10px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .pipeline-score {{
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 4px;
            color: white;
        }}
        .score-excellent {{ background-color: #28a745; }}
        .score-good {{ background-color: #ffc107; color: #333; }}
        .score-fair {{ background-color: #fd7e14; }}
        .score-poor {{ background-color: #dc3545; }}
        .last-updated {{
            text-align: center;
            color: #6c757d;
            margin-top: 20px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>🎯 Pipeline Quality Dashboard</h1>
        <p>Comprehensive overview of all pipeline quality metrics</p>
    </div>
    
    <div class="dashboard-grid">
        <div class="dashboard-card">
            <h3>📊 Quality Overview</h3>
            <div style="text-align: center;">
                <div class="stat-number">{average_score:.1f}</div>
                <p>Average Quality Score</p>
            </div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 20px;">
                <div style="text-align: center;">
                    <div style="font-size: 1.5em; font-weight: bold; color: #28a745;">{success_rate:.1f}%</div>
                    <div style="font-size: 0.9em; color: #6c757d;">Success Rate</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.5em; font-weight: bold; color: #007acc;">{production_ready:.1f}%</div>
                    <div style="font-size: 0.9em; color: #6c757d;">Production Ready</div>
                </div>
            </div>
        </div>
        
        <div class="dashboard-card">
            <h3>📈 Quality Distribution</h3>
            <div class="chart-container">
                <canvas id="qualityChart"></canvas>
            </div>
        </div>
        
        <div class="dashboard-card">
            <h3>⚠️ Issues Overview</h3>
            <div class="chart-container">
                <canvas id="issuesChart"></canvas>
            </div>
        </div>
        
        <div class="dashboard-card">
            <h3>🏆 Top Performing Pipelines</h3>
            <div class="pipeline-list">
                {top_pipelines_html}
            </div>
        </div>
        
        <div class="dashboard-card">
            <h3>🚨 Attention Required</h3>
            <div class="pipeline-list">
                {attention_pipelines_html}
            </div>
        </div>
        
        <div class="dashboard-card">
            <h3>⏱️ Performance Metrics</h3>
            <div style="text-align: center;">
                <div class="stat-number">{avg_review_time:.1f}s</div>
                <p>Average Review Time</p>
            </div>
            <hr>
            <p><strong>Total Pipelines:</strong> {total_pipelines}</p>
            <p><strong>Last Review Duration:</strong> {total_duration:.1f}s</p>
        </div>
    </div>
    
    <div class="last-updated">
        Last updated: {last_updated}
    </div>
    
    <script>
        // Quality Distribution Chart
        const qualityCtx = document.getElementById('qualityChart').getContext('2d');
        new Chart(qualityCtx, {{
            type: 'doughnut',
            data: {{
                labels: ['Excellent (90+)', 'Good (80-89)', 'Fair (70-79)', 'Poor (60-69)', 'Critical (<60)'],
                datasets: [{{
                    data: [{excellent}, {good}, {fair}, {poor}, {critical}],
                    backgroundColor: ['#28a745', '#ffc107', '#fd7e14', '#dc3545', '#6f42c1']
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'bottom'
                    }}
                }}
            }}
        }});
        
        // Issues Chart
        const issuesCtx = document.getElementById('issuesChart').getContext('2d');
        new Chart(issuesCtx, {{
            type: 'bar',
            data: {{
                labels: ['Critical', 'Major', 'Minor'],
                datasets: [{{
                    data: [{critical_issues}, {major_issues}, {minor_issues}],
                    backgroundColor: ['#dc3545', '#ffc107', '#17a2b8']
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
        """


class QualityReportGenerator:
    """Comprehensive quality report generation system."""
    
    def __init__(self, output_directory: Optional[Path] = None):
        self.output_directory = output_directory or Path("quality_reports")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_directory / "individual").mkdir(exist_ok=True)
        (self.output_directory / "aggregated").mkdir(exist_ok=True)
        (self.output_directory / "dashboard").mkdir(exist_ok=True)
        (self.output_directory / "exports").mkdir(exist_ok=True)
    
    def generate_individual_report(
        self,
        review: PipelineQualityReview,
        formats: List[str] = None
    ) -> Dict[str, Path]:
        """Generate individual pipeline report in multiple formats."""
        if formats is None:
            formats = ["json", "markdown", "html"]
        
        report_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON Report
        if "json" in formats:
            json_path = (
                self.output_directory / "individual" / 
                f"{review.pipeline_name}_report_{timestamp}.json"
            )
            with open(json_path, 'w') as f:
                json.dump(review.to_dict(), f, indent=2)
            report_files["json"] = json_path
        
        # Markdown Report
        if "markdown" in formats:
            md_path = (
                self.output_directory / "individual" / 
                f"{review.pipeline_name}_report_{timestamp}.md"
            )
            with open(md_path, 'w') as f:
                f.write(self._generate_markdown_report(review))
            report_files["markdown"] = md_path
        
        # HTML Report
        if "html" in formats:
            html_path = (
                self.output_directory / "individual" / 
                f"{review.pipeline_name}_report_{timestamp}.html"
            )
            with open(html_path, 'w') as f:
                f.write(self._generate_html_report(review))
            report_files["html"] = html_path
        
        return report_files
    
    def generate_batch_report(
        self,
        reviews: Dict[str, PipelineQualityReview],
        failed_reviews: Dict[str, str] = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Path]:
        """Generate comprehensive batch report."""
        failed_reviews = failed_reviews or {}
        metadata = metadata or {}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_data = self._compile_batch_data(reviews, failed_reviews, metadata)
        
        report_files = {}
        
        # JSON Report
        json_path = self.output_directory / "aggregated" / f"batch_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(batch_data, f, indent=2)
        report_files["json"] = json_path
        
        # Markdown Summary
        md_path = self.output_directory / "aggregated" / f"batch_summary_{timestamp}.md"
        with open(md_path, 'w') as f:
            f.write(self._generate_batch_markdown(batch_data))
        report_files["markdown"] = md_path
        
        # CSV Export
        csv_path = self.output_directory / "exports" / f"quality_data_{timestamp}.csv"
        self._export_to_csv(reviews, csv_path)
        report_files["csv"] = csv_path
        
        # Update latest files
        latest_json = self.output_directory / "aggregated" / "latest_batch_report.json"
        latest_md = self.output_directory / "aggregated" / "latest_batch_summary.md"
        
        with open(latest_json, 'w') as f:
            json.dump(batch_data, f, indent=2)
        
        with open(latest_md, 'w') as f:
            f.write(self._generate_batch_markdown(batch_data))
        
        return report_files
    
    def generate_dashboard(self, batch_data: Dict[str, Any] = None) -> Path:
        """Generate interactive HTML dashboard."""
        if batch_data is None:
            # Load latest batch report
            latest_report = self.output_directory / "aggregated" / "latest_batch_report.json"
            if latest_report.exists():
                with open(latest_report) as f:
                    batch_data = json.load(f)
            else:
                batch_data = {"error": "No batch data available"}
        
        if "error" in batch_data:
            return self._generate_empty_dashboard()
        
        dashboard_path = self.output_directory / "dashboard" / "quality_dashboard.html"
        
        with open(dashboard_path, 'w') as f:
            f.write(self._generate_dashboard_html(batch_data))
        
        return dashboard_path
    
    def _generate_markdown_report(self, review: PipelineQualityReview) -> str:
        """Generate markdown report for individual pipeline."""
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
    
    def _generate_html_report(self, review: PipelineQualityReview) -> str:
        """Generate HTML report for individual pipeline."""
        # Score classification
        if review.overall_score >= 90:
            score_class = "excellent"
            score_description = "Excellent - Production ready, no issues"
        elif review.overall_score >= 80:
            score_class = "good"
            score_description = "Good - Minor issues, acceptable for showcase"
        elif review.overall_score >= 70:
            score_class = "fair"
            score_description = "Fair - Some issues, needs improvement"
        elif review.overall_score >= 60:
            score_class = "poor"
            score_description = "Poor - Major issues, significant work needed"
        else:
            score_class = "critical"
            score_description = "Critical - Not suitable for production"
        
        # Ready status
        ready_class = "ready" if review.production_ready else "not-ready"
        ready_text = "Production Ready" if review.production_ready else "Not Production Ready"
        
        # Issues HTML
        issues_html = ""
        if review.total_issues > 0:
            issues_html = '<div class="issues-section"><h3>Issues Found</h3>'
            
            # Critical issues
            if review.critical_issues:
                issues_html += '<h4>🚨 Critical Issues</h4>'
                for issue in review.critical_issues:
                    issues_html += f'''
                    <div class="issue-card">
                        <div class="issue-header issue-critical">
                            🚨 Critical Issue in {os.path.basename(issue.file_path)}
                        </div>
                        <div class="issue-body">
                            <p><strong>Issue:</strong> {issue.description}</p>
                            <p><strong>Suggestion:</strong> {issue.suggestion}</p>
                        </div>
                    </div>
                    '''
            
            # Major issues
            if review.major_issues:
                issues_html += '<h4>⚠️ Major Issues</h4>'
                for issue in review.major_issues[:5]:  # Limit display
                    issues_html += f'''
                    <div class="issue-card">
                        <div class="issue-header issue-major">
                            ⚠️ Major Issue in {os.path.basename(issue.file_path)}
                        </div>
                        <div class="issue-body">
                            <p><strong>Issue:</strong> {issue.description}</p>
                            <p><strong>Suggestion:</strong> {issue.suggestion}</p>
                        </div>
                    </div>
                    '''
                    
                if len(review.major_issues) > 5:
                    issues_html += f'<p><em>... and {len(review.major_issues) - 5} more major issues</em></p>'
            
            issues_html += '</div>'
        else:
            issues_html = '''
            <div class="issues-section">
                <h3>✅ No Issues Found</h3>
                <p>All files meet production quality standards!</p>
            </div>
            '''
        
        # Recommendations HTML
        recommendations_html = ""
        if review.recommendations:
            recommendations_html = '''
            <div class="recommendations">
                <h3>📋 Recommendations</h3>
                <ul>
            '''
            for rec in review.recommendations:
                recommendations_html += f"<li>{rec}</li>"
            recommendations_html += "</ul></div>"
        
        # Files HTML
        files_html = ""
        for file_path in review.files_reviewed:
            file_type = "Image" if any(file_path.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif']) else "Text"
            files_html += f'''
            <tr>
                <td>{file_path}</td>
                <td>{file_type}</td>
            </tr>
            '''
        
        # Fill template
        template = ReportTemplate.get_html_template()
        return template.format(
            title=review.pipeline_name,
            pipeline_name=review.pipeline_name,
            timestamp=review.reviewed_at,
            duration=review.review_duration_seconds,
            model=review.reviewer_model,
            score=review.overall_score,
            score_class=score_class,
            score_description=score_description,
            ready_class=ready_class,
            ready_text=ready_text,
            files_count=len(review.files_reviewed),
            total_issues=review.total_issues,
            critical_count=len(review.critical_issues),
            major_count=len(review.major_issues),
            issues_html=issues_html,
            recommendations_html=recommendations_html,
            files_html=files_html
        )
    
    def _compile_batch_data(
        self,
        reviews: Dict[str, PipelineQualityReview],
        failed_reviews: Dict[str, str],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile comprehensive batch data."""
        total_pipelines = len(reviews) + len(failed_reviews)
        
        if reviews:
            scores = [review.overall_score for review in reviews.values()]
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            
            # Count issues by severity
            total_critical = sum(len(review.critical_issues) for review in reviews.values())
            total_major = sum(len(review.major_issues) for review in reviews.values())
            total_minor = sum(len(review.minor_issues) for review in reviews.values())
            
            # Production readiness
            production_ready = sum(1 for review in reviews.values() if review.production_ready)
        else:
            avg_score = min_score = max_score = 0
            total_critical = total_major = total_minor = 0
            production_ready = 0
        
        return {
            "batch_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_pipelines": total_pipelines,
                "successful_reviews": len(reviews),
                "failed_reviews": len(failed_reviews),
                "success_rate": (len(reviews) / total_pipelines * 100) if total_pipelines > 0 else 0
            },
            "quality_metrics": {
                "average_score": avg_score,
                "minimum_score": min_score,
                "maximum_score": max_score,
                "production_ready_count": production_ready,
                "production_ready_percentage": (production_ready / len(reviews) * 100) if reviews else 0,
                "total_critical_issues": total_critical,
                "total_major_issues": total_major,
                "total_minor_issues": total_minor
            },
            "detailed_reviews": {
                name: review.to_dict() for name, review in reviews.items()
            },
            "failed_pipelines": failed_reviews,
            "metadata": metadata,
            "pipeline_rankings": self._rank_pipelines(reviews)
        }
    
    def _rank_pipelines(self, reviews: Dict[str, PipelineQualityReview]) -> List[Dict[str, Any]]:
        """Rank pipelines by quality score."""
        rankings = []
        
        for name, review in reviews.items():
            rankings.append({
                "pipeline_name": name,
                "overall_score": review.overall_score,
                "production_ready": review.production_ready,
                "total_issues": review.total_issues,
                "critical_issues": len(review.critical_issues),
                "major_issues": len(review.major_issues),
                "minor_issues": len(review.minor_issues)
            })
        
        rankings.sort(key=lambda x: (-x["overall_score"], x["total_issues"]))
        return rankings
    
    def _generate_batch_markdown(self, batch_data: Dict[str, Any]) -> str:
        """Generate markdown summary for batch report."""
        summary = batch_data["batch_summary"]
        quality = batch_data["quality_metrics"]
        rankings = batch_data["pipeline_rankings"]
        
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
            f"## Pipeline Quality Rankings",
            f"",
            "| Rank | Pipeline | Score | Production Ready | Critical | Major | Minor |",
            "|------|----------|-------|------------------|----------|-------|--------|"
        ]
        
        for i, ranking in enumerate(rankings[:20], 1):  # Top 20
            ready_icon = "✅" if ranking["production_ready"] else "❌"
            md_lines.append(
                f"| {i} | {ranking['pipeline_name']} | {ranking['overall_score']:.1f} | "
                f"{ready_icon} | {ranking['critical_issues']} | {ranking['major_issues']} | "
                f"{ranking['minor_issues']} |"
            )
        
        # Failed pipelines
        failed_pipelines = batch_data.get("failed_pipelines", {})
        if failed_pipelines:
            md_lines.extend([
                f"",
                f"## Failed Pipeline Reviews",
                f""
            ])
            
            for pipeline_name, error in failed_pipelines.items():
                md_lines.append(f"- **{pipeline_name}**: {error}")
        
        return "\n".join(md_lines)
    
    def _generate_dashboard_html(self, batch_data: Dict[str, Any]) -> str:
        """Generate interactive dashboard HTML."""
        summary = batch_data["batch_summary"]
        quality = batch_data["quality_metrics"] 
        rankings = batch_data["pipeline_rankings"]
        
        # Quality distribution
        quality_dist = {"excellent": 0, "good": 0, "fair": 0, "poor": 0, "critical": 0}
        
        for pipeline_data in batch_data["detailed_reviews"].values():
            score = pipeline_data["overall_score"]
            if score >= 90:
                quality_dist["excellent"] += 1
            elif score >= 80:
                quality_dist["good"] += 1
            elif score >= 70:
                quality_dist["fair"] += 1
            elif score >= 60:
                quality_dist["poor"] += 1
            else:
                quality_dist["critical"] += 1
        
        # Top pipelines HTML
        top_pipelines_html = ""
        for i, ranking in enumerate(rankings[:5], 1):
            score = ranking["overall_score"]
            score_class = "excellent" if score >= 90 else "good" if score >= 80 else "fair" if score >= 70 else "poor"
            
            top_pipelines_html += f'''
            <div class="pipeline-item">
                <div>
                    <strong>{i}. {ranking["pipeline_name"]}</strong><br>
                    <small>{ranking["total_issues"]} issues</small>
                </div>
                <span class="pipeline-score score-{score_class}">{score:.0f}</span>
            </div>
            '''
        
        # Attention required HTML
        attention_pipelines = [r for r in rankings if r["overall_score"] < 80 or r["critical_issues"] > 0]
        attention_html = ""
        for ranking in attention_pipelines[:5]:
            score = ranking["overall_score"]
            score_class = "poor" if score < 60 else "fair"
            
            attention_html += f'''
            <div class="pipeline-item">
                <div>
                    <strong>{ranking["pipeline_name"]}</strong><br>
                    <small>{ranking["critical_issues"]} critical, {ranking["major_issues"]} major</small>
                </div>
                <span class="pipeline-score score-{score_class}">{score:.0f}</span>
            </div>
            '''
        
        template = ReportTemplate.get_dashboard_template()
        return template.format(
            average_score=quality["average_score"],
            success_rate=summary["success_rate"],
            production_ready=quality["production_ready_percentage"],
            total_pipelines=summary["total_pipelines"],
            avg_review_time=batch_data.get("metadata", {}).get("average_time_per_pipeline", 0),
            total_duration=batch_data.get("metadata", {}).get("total_duration_seconds", 0),
            last_updated=summary["timestamp"],
            excellent=quality_dist["excellent"],
            good=quality_dist["good"],
            fair=quality_dist["fair"],
            poor=quality_dist["poor"],
            critical=quality_dist["critical"],
            critical_issues=quality["total_critical_issues"],
            major_issues=quality["total_major_issues"],
            minor_issues=quality["total_minor_issues"],
            top_pipelines_html=top_pipelines_html,
            attention_pipelines_html=attention_html
        )
    
    def _export_to_csv(self, reviews: Dict[str, PipelineQualityReview], csv_path: Path):
        """Export review data to CSV format."""
        import csv
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'pipeline_name', 'overall_score', 'production_ready',
                'files_reviewed', 'critical_issues', 'major_issues', 'minor_issues',
                'review_duration_seconds', 'reviewer_model', 'reviewed_at'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for name, review in reviews.items():
                writer.writerow({
                    'pipeline_name': name,
                    'overall_score': review.overall_score,
                    'production_ready': review.production_ready,
                    'files_reviewed': len(review.files_reviewed),
                    'critical_issues': len(review.critical_issues),
                    'major_issues': len(review.major_issues),
                    'minor_issues': len(review.minor_issues),
                    'review_duration_seconds': review.review_duration_seconds,
                    'reviewer_model': review.reviewer_model,
                    'reviewed_at': review.reviewed_at
                })
    
    def _generate_empty_dashboard(self) -> Path:
        """Generate empty dashboard when no data is available."""
        dashboard_path = self.output_directory / "dashboard" / "quality_dashboard.html"
        
        empty_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Pipeline Quality Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
        .empty-state { color: #6c757d; }
    </style>
</head>
<body>
    <div class="empty-state">
        <h1>🎯 Pipeline Quality Dashboard</h1>
        <p>No quality review data available yet.</p>
        <p>Run a batch review to generate dashboard content.</p>
    </div>
</body>
</html>
        """
        
        with open(dashboard_path, 'w') as f:
            f.write(empty_html)
        
        return dashboard_path