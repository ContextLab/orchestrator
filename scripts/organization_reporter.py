#!/usr/bin/env python3
"""
Organization Reporter for Issue #255 Stream C.

Automated reporting system for repository organization compliance:
- Real-time organization health dashboard
- Trend analysis and compliance tracking
- Integration with monitoring and validation systems
- Automated compliance reports for stakeholders
- Historical data analysis and insights

Building on monitoring and validation infrastructure.
"""

import json
import logging
import sqlite3
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import StringIO

# Import our existing infrastructure
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from repository_organization_monitor import RepositoryOrganizationMonitor, OrganizationViolation
    from organization_validator import OrganizationValidator, ValidationReport
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class HealthMetric:
    """Health metric data point."""
    timestamp: datetime
    health_score: float
    violations_count: int
    violations_by_severity: Dict[str, int]
    validation_status: str
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class TrendAnalysis:
    """Trend analysis results."""
    metric_name: str
    time_period: str
    trend_direction: str  # 'improving', 'declining', 'stable'
    trend_strength: float  # 0-1, how strong the trend is
    current_value: float
    previous_value: float
    change_percentage: float
    insights: List[str]


class OrganizationReporter:
    """Comprehensive reporting system for repository organization."""
    
    def __init__(self, root_path: str = ".", config_file: Optional[str] = None):
        self.root_path = Path(root_path).resolve()
        self.config = self._load_config(config_file)
        
        # Initialize components
        self.monitor = RepositoryOrganizationMonitor(str(self.root_path))
        self.validator = OrganizationValidator(str(self.root_path))
        
        # Setup reporting infrastructure
        self.reports_dir = self.root_path / "temp" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.database_path = self.reports_dir / "organization_metrics.db"
        self._setup_database()
        
        # Metrics history
        self.metrics_history = deque(maxlen=1000)
        
        logger.info(f"Organization Reporter initialized for: {self.root_path}")

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load reporting configuration."""
        default_config = {
            "reporting": {
                "health_score_threshold": 80.0,
                "trend_analysis_days": 30,
                "generate_charts": True,
                "include_historical_data": True,
                "max_history_points": 100
            },
            "report_formats": {
                "dashboard": True,
                "detailed_json": True,
                "summary_markdown": True,
                "charts": True,
                "stakeholder_summary": True
            },
            "alert_thresholds": {
                "health_score_critical": 50.0,
                "health_score_warning": 70.0,
                "violations_critical": 20,
                "violations_warning": 10
            },
            "stakeholder_config": {
                "include_technical_details": False,
                "focus_on_trends": True,
                "executive_summary": True
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file) as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Could not load config file {config_file}: {e}")
        
        return default_config

    def _setup_database(self):
        """Setup SQLite database for metrics storage."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Create metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS health_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                health_score REAL,
                violations_count INTEGER,
                critical_violations INTEGER,
                error_violations INTEGER,
                warning_violations INTEGER,
                info_violations INTEGER,
                validation_status TEXT,
                details TEXT
            )
        ''')
        
        # Create trends table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trend_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                metric_name TEXT,
                trend_direction TEXT,
                trend_strength REAL,
                current_value REAL,
                change_percentage REAL,
                analysis_period_days INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()

    def collect_current_metrics(self) -> HealthMetric:
        """Collect current organization health metrics."""
        logger.info("Collecting current health metrics...")
        
        try:
            # Get monitoring data
            health_report = self.monitor.get_current_health_report()
            
            # Get validation status
            validation_report = self.validator.validate('ci_validation')
            
            # Create metric
            metric = HealthMetric(
                timestamp=datetime.now(),
                health_score=health_report['health_score'],
                violations_count=health_report['stats']['violations_detected'],
                violations_by_severity=health_report['violation_summary']['by_severity'],
                validation_status=validation_report.overall_status,
                details={
                    'monitoring_stats': health_report['stats'],
                    'validation_summary': validation_report.summary,
                    'recent_violations': health_report['recent_violations'][:5]  # Last 5
                }
            )
            
            # Store in memory
            self.metrics_history.append(metric)
            
            # Store in database
            self._store_metric_to_database(metric)
            
            logger.info(f"Metrics collected: Health Score {metric.health_score:.1f}, Violations {metric.violations_count}")
            return metric
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            # Return default metric
            return HealthMetric(
                timestamp=datetime.now(),
                health_score=0.0,
                violations_count=0,
                violations_by_severity={},
                validation_status="ERROR"
            )

    def _store_metric_to_database(self, metric: HealthMetric):
        """Store metric in database for historical analysis."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO health_metrics 
                (timestamp, health_score, violations_count, critical_violations, error_violations, 
                 warning_violations, info_violations, validation_status, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metric.timestamp,
                metric.health_score,
                metric.violations_count,
                metric.violations_by_severity.get('critical', 0),
                metric.violations_by_severity.get('error', 0),
                metric.violations_by_severity.get('warning', 0),
                metric.violations_by_severity.get('info', 0),
                metric.validation_status,
                json.dumps(metric.details, default=str)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store metric to database: {e}")

    def load_historical_metrics(self, days: int = 30) -> List[HealthMetric]:
        """Load historical metrics from database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            cursor.execute('''
                SELECT timestamp, health_score, violations_count, critical_violations, 
                       error_violations, warning_violations, info_violations, validation_status, details
                FROM health_metrics 
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
            ''', (cutoff_date,))
            
            metrics = []
            for row in cursor.fetchall():
                try:
                    details = json.loads(row[8]) if row[8] else {}
                except:
                    details = {}
                
                violations_by_severity = {
                    'critical': row[3],
                    'error': row[4],
                    'warning': row[5],
                    'info': row[6]
                }
                
                metrics.append(HealthMetric(
                    timestamp=datetime.fromisoformat(row[0]),
                    health_score=row[1],
                    violations_count=row[2],
                    violations_by_severity=violations_by_severity,
                    validation_status=row[7],
                    details=details
                ))
            
            conn.close()
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to load historical metrics: {e}")
            return []

    def analyze_trends(self, days: int = 30) -> List[TrendAnalysis]:
        """Analyze trends in organization health metrics."""
        logger.info(f"Analyzing trends over {days} days...")
        
        historical_metrics = self.load_historical_metrics(days)
        
        if len(historical_metrics) < 2:
            return []
        
        trends = []
        
        # Analyze health score trend
        health_scores = [m.health_score for m in historical_metrics]
        if health_scores:
            trend = self._calculate_trend(health_scores, "health_score")
            trends.append(TrendAnalysis(
                metric_name="Health Score",
                time_period=f"{days} days",
                trend_direction=trend['direction'],
                trend_strength=trend['strength'],
                current_value=health_scores[-1],
                previous_value=health_scores[0] if health_scores else 0,
                change_percentage=trend['change_percentage'],
                insights=self._generate_health_score_insights(health_scores, trend)
            ))
        
        # Analyze violations trend
        violations_counts = [m.violations_count for m in historical_metrics]
        if violations_counts:
            trend = self._calculate_trend(violations_counts, "violations", inverse=True)  # Lower is better
            trends.append(TrendAnalysis(
                metric_name="Total Violations",
                time_period=f"{days} days",
                trend_direction=trend['direction'],
                trend_strength=trend['strength'],
                current_value=violations_counts[-1],
                previous_value=violations_counts[0] if violations_counts else 0,
                change_percentage=trend['change_percentage'],
                insights=self._generate_violations_insights(violations_counts, trend)
            ))
        
        # Analyze critical violations trend
        critical_counts = [m.violations_by_severity.get('critical', 0) for m in historical_metrics]
        if critical_counts:
            trend = self._calculate_trend(critical_counts, "critical_violations", inverse=True)
            trends.append(TrendAnalysis(
                metric_name="Critical Violations",
                time_period=f"{days} days",
                trend_direction=trend['direction'],
                trend_strength=trend['strength'],
                current_value=critical_counts[-1],
                previous_value=critical_counts[0] if critical_counts else 0,
                change_percentage=trend['change_percentage'],
                insights=self._generate_critical_violations_insights(critical_counts, trend)
            ))
        
        logger.info(f"Generated {len(trends)} trend analyses")
        return trends

    def _calculate_trend(self, values: List[float], metric_name: str, inverse: bool = False) -> Dict[str, Any]:
        """Calculate trend direction and strength for a series of values."""
        if len(values) < 2:
            return {'direction': 'stable', 'strength': 0.0, 'change_percentage': 0.0}
        
        # Simple linear trend calculation
        n = len(values)
        x = list(range(n))
        y = values
        
        # Calculate linear regression slope
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        slope_numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        slope_denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if slope_denominator == 0:
            slope = 0
        else:
            slope = slope_numerator / slope_denominator
        
        # Determine trend direction
        if inverse:  # For metrics where lower is better
            if slope < -0.1:
                direction = 'improving'
            elif slope > 0.1:
                direction = 'declining'
            else:
                direction = 'stable'
        else:  # For metrics where higher is better
            if slope > 0.1:
                direction = 'improving'
            elif slope < -0.1:
                direction = 'declining'
            else:
                direction = 'stable'
        
        # Calculate trend strength (0-1)
        strength = min(abs(slope) / max(abs(y_mean), 1), 1.0)
        
        # Calculate change percentage
        if values[0] != 0:
            change_percentage = ((values[-1] - values[0]) / values[0]) * 100
        else:
            change_percentage = 0.0
        
        return {
            'direction': direction,
            'strength': strength,
            'change_percentage': change_percentage,
            'slope': slope
        }

    def _generate_health_score_insights(self, scores: List[float], trend: Dict[str, Any]) -> List[str]:
        """Generate insights for health score trends."""
        insights = []
        current_score = scores[-1]
        
        if trend['direction'] == 'improving':
            insights.append(f"Repository health is improving with a {abs(trend['change_percentage']):.1f}% increase")
        elif trend['direction'] == 'declining':
            insights.append(f"Repository health is declining with a {abs(trend['change_percentage']):.1f}% decrease")
        else:
            insights.append("Repository health score remains stable")
        
        if current_score >= 90:
            insights.append("Excellent organization health - repository is well maintained")
        elif current_score >= 80:
            insights.append("Good organization health - minor issues detected")
        elif current_score >= 70:
            insights.append("Moderate organization health - some cleanup needed")
        else:
            insights.append("Poor organization health - significant cleanup required")
        
        return insights

    def _generate_violations_insights(self, counts: List[int], trend: Dict[str, Any]) -> List[str]:
        """Generate insights for violations trends."""
        insights = []
        current_count = counts[-1]
        
        if trend['direction'] == 'improving':
            insights.append(f"Violations are decreasing - {abs(trend['change_percentage']):.1f}% reduction")
        elif trend['direction'] == 'declining':
            insights.append(f"Violations are increasing - {abs(trend['change_percentage']):.1f}% growth")
        else:
            insights.append("Violation count remains stable")
        
        if current_count == 0:
            insights.append("No violations detected - repository is perfectly organized")
        elif current_count <= 5:
            insights.append("Very few violations - excellent organization maintenance")
        elif current_count <= 20:
            insights.append("Some violations present - regular cleanup recommended")
        else:
            insights.append("Many violations detected - cleanup action needed")
        
        return insights

    def _generate_critical_violations_insights(self, counts: List[int], trend: Dict[str, Any]) -> List[str]:
        """Generate insights for critical violations trends."""
        insights = []
        current_count = counts[-1]
        
        if current_count == 0:
            insights.append("No critical violations - excellent security and organization")
        else:
            insights.append(f"Critical violations present - immediate attention required")
            
        if trend['direction'] == 'improving':
            insights.append("Critical violations are being addressed effectively")
        elif trend['direction'] == 'declining':
            insights.append("Critical violations are increasing - urgent action needed")
        
        return insights

    def generate_dashboard_report(self) -> str:
        """Generate real-time organization health dashboard."""
        logger.info("Generating organization health dashboard...")
        
        current_metric = self.collect_current_metrics()
        trends = self.analyze_trends(30)
        
        dashboard_lines = [
            "="*100,
            "REPOSITORY ORGANIZATION HEALTH DASHBOARD",
            "="*100,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Repository: {self.root_path}",
            "",
            "CURRENT HEALTH STATUS",
            "-"*50,
            f"🏥 Health Score: {current_metric.health_score:.1f}/100",
            f"📊 Validation Status: {current_metric.validation_status}",
            f"⚠️  Total Violations: {current_metric.violations_count}",
            "",
            "VIOLATION BREAKDOWN",
            "-"*50,
        ]
        
        # Add violation breakdown
        severity_icons = {'critical': '🔴', 'error': '🟠', 'warning': '🟡', 'info': '🔵'}
        for severity in ['critical', 'error', 'warning', 'info']:
            count = current_metric.violations_by_severity.get(severity, 0)
            if count > 0:
                icon = severity_icons[severity]
                dashboard_lines.append(f"{icon} {severity.title()}: {count}")
        
        if not any(current_metric.violations_by_severity.values()):
            dashboard_lines.append("✅ No violations detected")
        
        # Add trend analysis
        if trends:
            dashboard_lines.extend([
                "",
                "TREND ANALYSIS (30 DAYS)",
                "-"*50,
            ])
            
            for trend in trends:
                trend_icon = {
                    'improving': '📈',
                    'declining': '📉', 
                    'stable': '➡️'
                }[trend.trend_direction]
                
                dashboard_lines.append(f"{trend_icon} {trend.metric_name}: {trend.trend_direction.upper()}")
                dashboard_lines.append(f"   Current: {trend.current_value:.1f}, Change: {trend.change_percentage:+.1f}%")
                
                # Add top insight
                if trend.insights:
                    dashboard_lines.append(f"   💡 {trend.insights[0]}")
                dashboard_lines.append("")
        
        # Add alerts if needed
        alerts = self._generate_alerts(current_metric)
        if alerts:
            dashboard_lines.extend([
                "ALERTS & RECOMMENDATIONS",
                "-"*50,
            ])
            for alert in alerts:
                dashboard_lines.append(f"⚠️  {alert}")
            dashboard_lines.append("")
        
        # Add footer
        dashboard_lines.extend([
            "MONITORING STATUS",
            "-"*50,
            f"📡 Monitor: {'Active' if self.monitor.monitoring_active else 'Inactive'}",
            f"🔍 Last Scan: {current_metric.timestamp.strftime('%H:%M:%S')}",
            f"📈 Trend Strength: {max([t.trend_strength for t in trends], default=0):.2f}",
            "",
            "="*100
        ])
        
        return "\n".join(dashboard_lines)

    def _generate_alerts(self, metric: HealthMetric) -> List[str]:
        """Generate alerts based on current metrics."""
        alerts = []
        
        # Health score alerts
        if metric.health_score < self.config['alert_thresholds']['health_score_critical']:
            alerts.append(f"CRITICAL: Health score {metric.health_score:.1f} is below critical threshold")
        elif metric.health_score < self.config['alert_thresholds']['health_score_warning']:
            alerts.append(f"WARNING: Health score {metric.health_score:.1f} is below warning threshold")
        
        # Violation alerts
        if metric.violations_count > self.config['alert_thresholds']['violations_critical']:
            alerts.append(f"CRITICAL: {metric.violations_count} violations exceed critical threshold")
        elif metric.violations_count > self.config['alert_thresholds']['violations_warning']:
            alerts.append(f"WARNING: {metric.violations_count} violations exceed warning threshold")
        
        # Critical violation alerts
        critical_violations = metric.violations_by_severity.get('critical', 0)
        if critical_violations > 0:
            alerts.append(f"URGENT: {critical_violations} critical violations require immediate attention")
        
        return alerts

    def generate_detailed_json_report(self) -> Dict[str, Any]:
        """Generate detailed JSON report for programmatic access."""
        logger.info("Generating detailed JSON report...")
        
        current_metric = self.collect_current_metrics()
        trends = self.analyze_trends(30)
        historical_metrics = self.load_historical_metrics(30)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'repository_path': str(self.root_path),
            'current_metrics': asdict(current_metric),
            'trend_analysis': [asdict(trend) for trend in trends],
            'historical_metrics': [asdict(metric) for metric in historical_metrics[-20:]],  # Last 20
            'alerts': self._generate_alerts(current_metric),
            'recommendations': self._generate_recommendations(current_metric, trends),
            'metadata': {
                'report_version': '1.0',
                'generator': 'OrganizationReporter',
                'config': self.config
            }
        }
        
        return report

    def _generate_recommendations(self, metric: HealthMetric, trends: List[TrendAnalysis]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Health score recommendations
        if metric.health_score < 80:
            recommendations.append("Run repository scanner to identify organization issues")
            recommendations.append("Execute automated cleanup tools to improve health score")
        
        # Violation-based recommendations
        if metric.violations_count > 10:
            recommendations.append("Enable automated monitoring for real-time violation detection")
            recommendations.append("Implement pre-commit hooks to prevent future violations")
        
        # Trend-based recommendations
        for trend in trends:
            if trend.trend_direction == 'declining' and trend.metric_name == 'Health Score':
                recommendations.append("Health score is declining - review recent changes and address violations")
            elif trend.trend_direction == 'declining' and 'Violations' in trend.metric_name:
                recommendations.append("Violations are increasing - schedule regular cleanup maintenance")
        
        # Critical violations
        critical_count = metric.violations_by_severity.get('critical', 0)
        if critical_count > 0:
            recommendations.append("Address critical violations immediately to prevent system issues")
        
        return recommendations

    def generate_stakeholder_summary(self) -> str:
        """Generate executive summary for stakeholders."""
        logger.info("Generating stakeholder summary...")
        
        current_metric = self.collect_current_metrics()
        trends = self.analyze_trends(30)
        
        # Calculate summary metrics
        health_status = "Excellent" if current_metric.health_score >= 90 else \
                       "Good" if current_metric.health_score >= 80 else \
                       "Fair" if current_metric.health_score >= 70 else "Poor"
        
        trend_summary = "Improving" if any(t.trend_direction == 'improving' for t in trends) else \
                       "Declining" if any(t.trend_direction == 'declining' for t in trends) else "Stable"
        
        summary_lines = [
            "# Repository Organization Executive Summary",
            f"**Report Date:** {datetime.now().strftime('%Y-%m-%d')}",
            f"**Repository:** {self.root_path.name}",
            "",
            "## Key Metrics",
            f"- **Overall Health:** {health_status} ({current_metric.health_score:.1f}/100)",
            f"- **Organization Status:** {current_metric.validation_status}",
            f"- **Issue Count:** {current_metric.violations_count} violations detected",
            f"- **Trend Direction:** {trend_summary} over past 30 days",
            "",
            "## Summary",
        ]
        
        # Add contextual summary
        if current_metric.health_score >= 80:
            summary_lines.append("✅ The repository maintains good organization standards with minimal issues.")
        else:
            summary_lines.append("⚠️ The repository requires attention to improve organization standards.")
        
        if current_metric.violations_count == 0:
            summary_lines.append("✅ No organization violations currently detected.")
        elif current_metric.violations_count <= 10:
            summary_lines.append("ℹ️ Minor organization issues detected that can be easily addressed.")
        else:
            summary_lines.append("⚠️ Multiple organization violations require cleanup action.")
        
        # Add trend insights
        summary_lines.extend([
            "",
            "## Recent Trends"
        ])
        
        for trend in trends[:3]:  # Top 3 trends
            if trend.insights:
                summary_lines.append(f"- **{trend.metric_name}:** {trend.insights[0]}")
        
        # Add recommendations
        recommendations = self._generate_recommendations(current_metric, trends)
        if recommendations:
            summary_lines.extend([
                "",
                "## Recommended Actions"
            ])
            for i, rec in enumerate(recommendations[:3], 1):  # Top 3 recommendations
                summary_lines.append(f"{i}. {rec}")
        
        return "\n".join(summary_lines)

    def save_all_reports(self) -> Dict[str, Path]:
        """Generate and save all report formats."""
        logger.info("Generating and saving all reports...")
        
        saved_reports = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Dashboard report
            if self.config['report_formats']['dashboard']:
                dashboard = self.generate_dashboard_report()
                dashboard_path = self.reports_dir / f"dashboard_{timestamp}.txt"
                with open(dashboard_path, 'w') as f:
                    f.write(dashboard)
                saved_reports['dashboard'] = dashboard_path
            
            # Detailed JSON report
            if self.config['report_formats']['detailed_json']:
                json_report = self.generate_detailed_json_report()
                json_path = self.reports_dir / f"detailed_report_{timestamp}.json"
                with open(json_path, 'w') as f:
                    json.dump(json_report, f, indent=2, default=str)
                saved_reports['json'] = json_path
            
            # Stakeholder summary
            if self.config['report_formats']['stakeholder_summary']:
                stakeholder_summary = self.generate_stakeholder_summary()
                summary_path = self.reports_dir / f"stakeholder_summary_{timestamp}.md"
                with open(summary_path, 'w') as f:
                    f.write(stakeholder_summary)
                saved_reports['stakeholder'] = summary_path
            
            # Generate charts if enabled and matplotlib is available
            if self.config['report_formats']['charts']:
                try:
                    chart_path = self._generate_health_trends_chart(timestamp)
                    if chart_path:
                        saved_reports['chart'] = chart_path
                except Exception as e:
                    logger.warning(f"Could not generate charts: {e}")
            
            logger.info(f"Generated {len(saved_reports)} reports")
            return saved_reports
            
        except Exception as e:
            logger.error(f"Failed to save reports: {e}")
            return {}

    def _generate_health_trends_chart(self, timestamp: str) -> Optional[Path]:
        """Generate health trends chart."""
        try:
            historical_metrics = self.load_historical_metrics(30)
            if len(historical_metrics) < 2:
                return None
            
            # Extract data
            dates = [m.timestamp for m in historical_metrics]
            health_scores = [m.health_score for m in historical_metrics]
            violation_counts = [m.violations_count for m in historical_metrics]
            
            # Create chart
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            fig.suptitle('Repository Organization Health Trends', fontsize=16)
            
            # Health score plot
            ax1.plot(dates, health_scores, 'b-', linewidth=2, label='Health Score')
            ax1.set_ylabel('Health Score')
            ax1.set_ylim(0, 100)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Violations plot
            ax2.plot(dates, violation_counts, 'r-', linewidth=2, label='Violations')
            ax2.set_ylabel('Violation Count')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Format dates
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            chart_path = self.reports_dir / f"health_trends_{timestamp}.png"
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Health trends chart saved: {chart_path}")
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to generate chart: {e}")
            return None


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Repository Organization Reporter")
    parser.add_argument("--root", default=".", help="Repository root path")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--dashboard", action='store_true', help="Show dashboard")
    parser.add_argument("--generate-all", action='store_true', help="Generate all reports")
    parser.add_argument("--trends", type=int, default=30, help="Analyze trends for N days")
    parser.add_argument("--stakeholder", action='store_true', help="Generate stakeholder summary")
    parser.add_argument("--json-report", action='store_true', help="Generate JSON report")
    parser.add_argument("--collect-metrics", action='store_true', help="Collect current metrics")
    parser.add_argument("--verbose", action='store_true', help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    reporter = OrganizationReporter(args.root, args.config)
    
    if args.dashboard:
        print(reporter.generate_dashboard_report())
        
    elif args.generate_all:
        saved_reports = reporter.save_all_reports()
        print("📊 All reports generated:")
        for report_type, path in saved_reports.items():
            print(f"  {report_type}: {path}")
            
    elif args.trends:
        trends = reporter.analyze_trends(args.trends)
        print(f"📈 Trend Analysis ({args.trends} days):")
        for trend in trends:
            print(f"\n{trend.metric_name}:")
            print(f"  Direction: {trend.trend_direction}")
            print(f"  Change: {trend.change_percentage:+.1f}%")
            for insight in trend.insights[:2]:
                print(f"  💡 {insight}")
                
    elif args.stakeholder:
        print(reporter.generate_stakeholder_summary())
        
    elif args.json_report:
        import json
        report = reporter.generate_detailed_json_report()
        print(json.dumps(report, indent=2, default=str))
        
    elif args.collect_metrics:
        metric = reporter.collect_current_metrics()
        print(f"✅ Metrics collected:")
        print(f"  Health Score: {metric.health_score:.1f}")
        print(f"  Violations: {metric.violations_count}")
        print(f"  Status: {metric.validation_status}")
        
    else:
        print("Use --dashboard, --generate-all, --trends, --stakeholder, --json-report, or --collect-metrics")


if __name__ == "__main__":
    main()