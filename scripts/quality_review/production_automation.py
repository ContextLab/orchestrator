#!/usr/bin/env python3
"""
Production Automation System for Pipeline Quality Monitoring

This script provides production-ready automation capabilities including:
- Continuous quality monitoring
- Automated alerting and notifications
- Performance tracking and optimization
- Integration with CI/CD systems

Usage:
    python scripts/quality_review/production_automation.py --daemon
    python scripts/quality_review/production_automation.py --check-now
    python scripts/quality_review/production_automation.py --health-check
    python scripts/quality_review/production_automation.py --performance-report
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import smtplib
import sys
import time
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator.core.credential_manager import create_credential_manager
from quality_review.batch_reviewer import ComprehensiveBatchReviewer, BatchReviewConfig
from quality_review.integrated_validation import IntegratedValidationSystem


class AlertLevel:
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


class ProductionAlert:
    """Represents a production alert."""
    
    def __init__(
        self,
        level: str,
        title: str,
        message: str,
        data: Dict[str, Any] = None,
        timestamp: Optional[datetime] = None
    ):
        self.level = level
        self.title = title
        self.message = message
        self.data = data or {}
        self.timestamp = timestamp or datetime.now()
        self.id = f"{int(self.timestamp.timestamp())}_{hash(title) % 10000}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "level": self.level,
            "title": self.title,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }


class PerformanceTracker:
    """Tracks performance metrics over time."""
    
    def __init__(self, data_file: Path):
        self.data_file = data_file
        self.metrics = self._load_historical_data()
    
    def _load_historical_data(self) -> List[Dict[str, Any]]:
        """Load historical performance data."""
        if self.data_file.exists():
            try:
                with open(self.data_file) as f:
                    return json.load(f)
            except Exception:
                return []
        return []
    
    def add_metrics(self, metrics: Dict[str, Any]):
        """Add new performance metrics."""
        metrics["timestamp"] = datetime.now().isoformat()
        self.metrics.append(metrics)
        
        # Keep only last 100 entries
        self.metrics = self.metrics[-100:]
        
        # Save to file
        with open(self.data_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_trend_analysis(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """Analyze trends for a specific metric."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = [
            m for m in self.metrics
            if datetime.fromisoformat(m["timestamp"]) > cutoff_time
            and metric_name in m
        ]
        
        if len(recent_metrics) < 2:
            return {"trend": "insufficient_data", "values": []}
        
        values = [m[metric_name] for m in recent_metrics]
        
        # Simple trend calculation
        if len(values) >= 2:
            recent_avg = sum(values[-3:]) / min(3, len(values))
            older_avg = sum(values[:-3]) / max(1, len(values) - 3) if len(values) > 3 else recent_avg
            
            if recent_avg > older_avg * 1.1:
                trend = "improving"
            elif recent_avg < older_avg * 0.9:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "values": values,
            "current_value": values[-1] if values else 0,
            "average_value": sum(values) / len(values) if values else 0,
            "min_value": min(values) if values else 0,
            "max_value": max(values) if values else 0
        }


class NotificationManager:
    """Manages notifications and alerts."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("NotificationManager")
        
        # Email configuration
        self.email_enabled = self.config.get("email", {}).get("enabled", False)
        self.smtp_server = self.config.get("email", {}).get("smtp_server", "smtp.gmail.com")
        self.smtp_port = self.config.get("email", {}).get("smtp_port", 587)
        self.email_user = self.config.get("email", {}).get("username", "")
        self.email_password = self.config.get("email", {}).get("password", "")
        self.recipients = self.config.get("email", {}).get("recipients", [])
        
        # Webhook configuration
        self.webhook_enabled = self.config.get("webhook", {}).get("enabled", False)
        self.webhook_url = self.config.get("webhook", {}).get("url", "")
    
    async def send_alert(self, alert: ProductionAlert):
        """Send alert through configured channels."""
        self.logger.info(f"Sending {alert.level} alert: {alert.title}")
        
        try:
            # Send email notification
            if self.email_enabled and self.recipients:
                await self._send_email_alert(alert)
            
            # Send webhook notification
            if self.webhook_enabled and self.webhook_url:
                await self._send_webhook_alert(alert)
            
            # Log alert
            self._log_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
    
    async def _send_email_alert(self, alert: ProductionAlert):
        """Send email alert."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = ", ".join(self.recipients)
            msg['Subject'] = f"[{alert.level.upper()}] Pipeline Quality Alert: {alert.title}"
            
            body = f"""
Pipeline Quality Monitoring Alert

Level: {alert.level.upper()}
Title: {alert.title}
Time: {alert.timestamp}

Message:
{alert.message}

Additional Data:
{json.dumps(alert.data, indent=2)}

---
Automated Pipeline Quality Monitoring System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email alert sent to {len(self.recipients)} recipients")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    async def _send_webhook_alert(self, alert: ProductionAlert):
        """Send webhook alert."""
        try:
            import aiohttp
            
            payload = {
                "alert": alert.to_dict(),
                "source": "pipeline_quality_monitor"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        self.logger.info("Webhook alert sent successfully")
                    else:
                        self.logger.error(f"Webhook alert failed with status {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
    
    def _log_alert(self, alert: ProductionAlert):
        """Log alert to file."""
        alert_log_file = Path("production_alerts.log")
        
        with open(alert_log_file, 'a') as f:
            f.write(f"{alert.timestamp.isoformat()} [{alert.level.upper()}] {alert.title}: {alert.message}\n")


class ProductionAutomationSystem:
    """Comprehensive production automation system."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        
        # Initialize components
        self.batch_reviewer = ComprehensiveBatchReviewer(
            BatchReviewConfig(
                max_concurrent_reviews=self.config.get("max_concurrent_reviews", 2),
                timeout_per_pipeline=self.config.get("timeout_per_pipeline", 300),
                enable_caching=self.config.get("enable_caching", True),
                output_directory=self.config.get("output_directory", "production_quality_reports")
            )
        )
        
        self.validation_system = IntegratedValidationSystem()
        self.notification_manager = NotificationManager(self.config.get("notifications", {}))
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker(
            Path(self.config.get("performance_data_file", "performance_metrics.json"))
        )
        
        # State management
        self.running = False
        self.last_check_time = None
        self.check_interval = self.config.get("check_interval_minutes", 60) * 60  # Convert to seconds
        self.consecutive_failures = 0
        self.max_consecutive_failures = self.config.get("max_consecutive_failures", 3)
        
        self.logger.info("Production Automation System initialized")
    
    def _load_config(self, config_file: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "check_interval_minutes": 60,
            "max_concurrent_reviews": 2,
            "timeout_per_pipeline": 300,
            "enable_caching": True,
            "output_directory": "production_quality_reports",
            "performance_data_file": "performance_metrics.json",
            "max_consecutive_failures": 3,
            "quality_thresholds": {
                "average_score_warning": 70,
                "average_score_critical": 50,
                "production_ready_warning": 60,
                "production_ready_critical": 30,
                "success_rate_warning": 80,
                "success_rate_critical": 50
            },
            "notifications": {
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "recipients": []
                },
                "webhook": {
                    "enabled": False,
                    "url": ""
                }
            }
        }
        
        if config_file and config_file.exists():
            try:
                with open(config_file) as f:
                    file_config = json.load(f)
                # Deep merge configurations
                default_config.update(file_config)
            except Exception as e:
                logging.error(f"Failed to load config file {config_file}: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging."""
        logger = logging.getLogger("ProductionAutomation")
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
        log_file = Path("production_automation.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    async def run_daemon(self):
        """Run continuous monitoring daemon."""
        self.logger.info("Starting production automation daemon")
        self.running = True
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            while self.running:
                check_start_time = time.time()
                
                try:
                    # Run quality check
                    await self._run_quality_check()
                    
                    # Reset failure counter on success
                    self.consecutive_failures = 0
                    
                except Exception as e:
                    self.consecutive_failures += 1
                    self.logger.error(f"Quality check failed: {e}")
                    
                    # Send alert for consecutive failures
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        await self.notification_manager.send_alert(
                            ProductionAlert(
                                level=AlertLevel.CRITICAL,
                                title="Quality Check System Failure",
                                message=f"Quality checks have failed {self.consecutive_failures} times consecutively. "
                                       f"Last error: {e}",
                                data={"consecutive_failures": self.consecutive_failures, "error": str(e)}
                            )
                        )
                
                # Calculate next check time
                check_duration = time.time() - check_start_time
                sleep_time = max(0, self.check_interval - check_duration)
                
                self.logger.info(f"Next quality check in {sleep_time/60:.1f} minutes")
                
                # Sleep with periodic wake-ups to check running status
                sleep_intervals = 30  # Check every 30 seconds
                while sleep_time > 0 and self.running:
                    actual_sleep = min(sleep_intervals, sleep_time)
                    await asyncio.sleep(actual_sleep)
                    sleep_time -= actual_sleep
                
        except Exception as e:
            self.logger.error(f"Daemon failed with critical error: {e}")
            await self.notification_manager.send_alert(
                ProductionAlert(
                    level=AlertLevel.FATAL,
                    title="Production Automation System Failure",
                    message=f"The production automation system has encountered a fatal error: {e}",
                    data={"error": str(e)}
                )
            )
            raise
        finally:
            self.logger.info("Production automation daemon stopped")
    
    async def _run_quality_check(self):
        """Run comprehensive quality check."""
        self.logger.info("Starting scheduled quality check")
        check_start_time = time.time()
        
        try:
            # Run batch review of all pipelines
            pipelines = self.batch_reviewer.available_pipelines
            batch_report = await self.batch_reviewer.batch_review_pipelines(
                pipelines,
                show_progress=False
            )
            
            check_duration = time.time() - check_start_time
            self.last_check_time = datetime.now()
            
            # Extract key metrics
            summary = batch_report["batch_review_summary"]
            quality_metrics = batch_report["quality_metrics"]
            
            # Record performance metrics
            performance_metrics = {
                "check_duration_seconds": check_duration,
                "total_pipelines": summary["total_pipelines"],
                "success_rate": summary["success_rate"],
                "average_quality_score": quality_metrics["average_score"],
                "production_ready_percentage": quality_metrics["production_ready_percentage"],
                "critical_issues": quality_metrics["total_critical_issues"],
                "major_issues": quality_metrics["total_major_issues"],
                "minor_issues": quality_metrics["total_minor_issues"]
            }
            
            self.performance_tracker.add_metrics(performance_metrics)
            
            # Check for alerts
            await self._check_quality_alerts(quality_metrics, summary)
            
            self.logger.info(f"Quality check completed in {check_duration:.1f}s")
            self.logger.info(f"  Success rate: {summary['success_rate']:.1f}%")
            self.logger.info(f"  Average quality: {quality_metrics['average_score']:.1f}/100")
            self.logger.info(f"  Production ready: {quality_metrics['production_ready_count']}/{summary['successful_reviews']}")
            
        except Exception as e:
            self.logger.error(f"Quality check failed: {e}")
            raise
    
    async def _check_quality_alerts(
        self,
        quality_metrics: Dict[str, Any],
        summary: Dict[str, Any]
    ):
        """Check for quality-based alerts."""
        thresholds = self.config["quality_thresholds"]
        
        # Average score alerts
        avg_score = quality_metrics["average_score"]
        if avg_score <= thresholds["average_score_critical"]:
            await self.notification_manager.send_alert(
                ProductionAlert(
                    level=AlertLevel.CRITICAL,
                    title="Critical Quality Score Drop",
                    message=f"Average quality score has dropped to {avg_score:.1f}/100 "
                           f"(threshold: {thresholds['average_score_critical']})",
                    data={"average_score": avg_score, "threshold": thresholds["average_score_critical"]}
                )
            )
        elif avg_score <= thresholds["average_score_warning"]:
            await self.notification_manager.send_alert(
                ProductionAlert(
                    level=AlertLevel.WARNING,
                    title="Quality Score Below Warning Threshold",
                    message=f"Average quality score is {avg_score:.1f}/100 "
                           f"(warning threshold: {thresholds['average_score_warning']})",
                    data={"average_score": avg_score, "threshold": thresholds["average_score_warning"]}
                )
            )
        
        # Production readiness alerts
        prod_ready = quality_metrics["production_ready_percentage"]
        if prod_ready <= thresholds["production_ready_critical"]:
            await self.notification_manager.send_alert(
                ProductionAlert(
                    level=AlertLevel.CRITICAL,
                    title="Critical Production Readiness Drop",
                    message=f"Only {prod_ready:.1f}% of pipelines are production ready "
                           f"(threshold: {thresholds['production_ready_critical']}%)",
                    data={"production_ready_percentage": prod_ready, "threshold": thresholds["production_ready_critical"]}
                )
            )
        elif prod_ready <= thresholds["production_ready_warning"]:
            await self.notification_manager.send_alert(
                ProductionAlert(
                    level=AlertLevel.WARNING,
                    title="Production Readiness Below Warning Threshold",
                    message=f"{prod_ready:.1f}% of pipelines are production ready "
                           f"(warning threshold: {thresholds['production_ready_warning']}%)",
                    data={"production_ready_percentage": prod_ready, "threshold": thresholds["production_ready_warning"]}
                )
            )
        
        # Success rate alerts
        success_rate = summary["success_rate"]
        if success_rate <= thresholds["success_rate_critical"]:
            await self.notification_manager.send_alert(
                ProductionAlert(
                    level=AlertLevel.CRITICAL,
                    title="Critical Review Success Rate Drop",
                    message=f"Review success rate has dropped to {success_rate:.1f}% "
                           f"(threshold: {thresholds['success_rate_critical']}%)",
                    data={"success_rate": success_rate, "threshold": thresholds["success_rate_critical"]}
                )
            )
        elif success_rate <= thresholds["success_rate_warning"]:
            await self.notification_manager.send_alert(
                ProductionAlert(
                    level=AlertLevel.WARNING,
                    title="Review Success Rate Below Warning Threshold",
                    message=f"Review success rate is {success_rate:.1f}% "
                           f"(warning threshold: {thresholds['success_rate_warning']}%)",
                    data={"success_rate": success_rate, "threshold": thresholds["success_rate_warning"]}
                )
            )
        
        # Critical issues alert
        critical_issues = quality_metrics["total_critical_issues"]
        if critical_issues > 0:
            await self.notification_manager.send_alert(
                ProductionAlert(
                    level=AlertLevel.WARNING,
                    title="Critical Issues Detected",
                    message=f"{critical_issues} critical issues found in pipeline outputs",
                    data={"critical_issues": critical_issues}
                )
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "healthy",
            "components": {},
            "metrics": {},
            "alerts": []
        }
        
        try:
            # Check batch reviewer
            batch_reviewer_health = len(self.batch_reviewer.available_pipelines) > 0
            health_status["components"]["batch_reviewer"] = {
                "status": "healthy" if batch_reviewer_health else "unhealthy",
                "available_pipelines": len(self.batch_reviewer.available_pipelines)
            }
            
            # Check performance trends
            avg_score_trend = self.performance_tracker.get_trend_analysis("average_quality_score", hours=24)
            success_rate_trend = self.performance_tracker.get_trend_analysis("success_rate", hours=24)
            
            health_status["metrics"] = {
                "average_quality_score_trend": avg_score_trend,
                "success_rate_trend": success_rate_trend,
                "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
                "consecutive_failures": self.consecutive_failures
            }
            
            # Overall health determination
            if self.consecutive_failures >= self.max_consecutive_failures:
                health_status["overall_health"] = "critical"
                health_status["alerts"].append("Too many consecutive failures")
            
            if not batch_reviewer_health:
                health_status["overall_health"] = "degraded"
                health_status["alerts"].append("No pipelines available for review")
            
            if avg_score_trend.get("trend") == "degrading":
                health_status["overall_health"] = "degraded"
                health_status["alerts"].append("Quality scores trending downward")
            
        except Exception as e:
            health_status["overall_health"] = "critical"
            health_status["error"] = str(e)
        
        return health_status
    
    async def generate_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "report_period_hours": hours,
            "generated_at": datetime.now().isoformat(),
            "metrics_analysis": {},
            "trends": {},
            "recommendations": []
        }
        
        # Analyze key metrics
        metrics_to_analyze = [
            "average_quality_score",
            "success_rate", 
            "production_ready_percentage",
            "check_duration_seconds"
        ]
        
        for metric in metrics_to_analyze:
            trend_analysis = self.performance_tracker.get_trend_analysis(metric, hours)
            report["trends"][metric] = trend_analysis
            
            # Generate metric-specific insights
            if trend_analysis["trend"] == "degrading":
                report["recommendations"].append(f"Investigate {metric} degradation")
            elif trend_analysis["trend"] == "improving":
                report["recommendations"].append(f"Document improvements in {metric}")
        
        # Overall assessment
        degrading_metrics = [
            metric for metric, trend in report["trends"].items()
            if trend.get("trend") == "degrading"
        ]
        
        if len(degrading_metrics) > len(metrics_to_analyze) / 2:
            report["overall_assessment"] = "concerning"
            report["recommendations"].append("Multiple metrics are degrading - comprehensive review needed")
        elif len(degrading_metrics) > 0:
            report["overall_assessment"] = "attention_needed"
            report["recommendations"].append(f"Monitor degrading metrics: {', '.join(degrading_metrics)}")
        else:
            report["overall_assessment"] = "good"
            report["recommendations"].append("System performance is stable")
        
        return report


async def main():
    """Main function for production automation."""
    parser = argparse.ArgumentParser(
        description="Production Automation System for Pipeline Quality Monitoring"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as continuous monitoring daemon"
    )
    parser.add_argument(
        "--check-now",
        action="store_true",
        help="Run single quality check and exit"
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Perform system health check"
    )
    parser.add_argument(
        "--performance-report",
        action="store_true",
        help="Generate performance report"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Configuration file path"
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Hours for performance report (default: 24)"
    )
    
    args = parser.parse_args()
    
    # Initialize system
    system = ProductionAutomationSystem(config_file=args.config)
    
    try:
        if args.daemon:
            print("🚀 Starting Production Automation Daemon")
            await system.run_daemon()
            
        elif args.check_now:
            print("🔍 Running Quality Check")
            await system._run_quality_check()
            print("✅ Quality check completed")
            
        elif args.health_check:
            print("🏥 Running Health Check")
            health = await system.health_check()
            print(json.dumps(health, indent=2))
            
        elif args.performance_report:
            print(f"📊 Generating Performance Report ({args.hours} hours)")
            report = await system.generate_performance_report(args.hours)
            print(json.dumps(report, indent=2))
            
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n🛑 Production automation interrupted by user")
    except Exception as e:
        print(f"\n❌ Production automation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())