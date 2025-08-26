"""
Production automation and scheduling system for pipeline testing infrastructure.

This module provides production-ready automation capabilities including scheduled
test execution, monitoring integration, alert management, and deployment validation.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Types of automated schedules."""
    CONTINUOUS = "continuous"    # Run continuously with intervals
    DAILY = "daily"             # Once per day
    WEEKLY = "weekly"           # Once per week  
    ON_DEMAND = "on_demand"     # Triggered by external events
    PRE_RELEASE = "pre_release" # Before releases
    POST_DEPLOY = "post_deploy" # After deployments


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AutomationStatus(Enum):
    """Automation execution status."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ScheduleConfig:
    """Configuration for scheduled test execution."""
    
    name: str
    schedule_type: ScheduleType
    test_mode: str = "core"  # quick, core, full
    
    # Timing configuration
    interval_minutes: Optional[int] = None
    daily_time: Optional[str] = None  # "14:30" format
    weekly_day: Optional[int] = None  # 0=Monday
    
    # Execution configuration
    max_execution_time_minutes: int = 60
    max_cost_dollars: float = 5.0
    retry_on_failure: bool = True
    max_retries: int = 2
    
    # Quality gates
    min_success_rate: float = 0.8
    min_quality_score: float = 80.0
    
    # Alert configuration
    alert_on_failure: bool = True
    alert_on_regression: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "success_rate": 0.8,
        "quality_score": 75.0,
        "execution_time_increase": 0.5  # 50% increase
    })
    
    # Integration settings
    enabled: bool = True
    environment: str = "production"
    tags: List[str] = field(default_factory=list)


@dataclass
class AutomationResult:
    """Result of automated test execution."""
    
    schedule_name: str
    execution_id: str
    start_time: datetime
    end_time: datetime
    status: AutomationStatus
    
    # Test results
    total_tests: int = 0
    successful_tests: int = 0
    failed_tests: int = 0
    success_rate: float = 0.0
    quality_score: float = 0.0
    execution_time_minutes: float = 0.0
    total_cost: float = 0.0
    
    # Issues and alerts
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    alerts_generated: List[Dict[str, Any]] = field(default_factory=list)
    
    # Artifacts
    report_path: Optional[Path] = None
    detailed_results_path: Optional[Path] = None


@dataclass
class AlertConfig:
    """Configuration for alert management."""
    
    # Alert channels
    enable_logging: bool = True
    enable_email: bool = False
    enable_slack: bool = False
    enable_webhooks: bool = False
    
    # Email settings
    email_recipients: List[str] = field(default_factory=list)
    email_smtp_host: Optional[str] = None
    email_smtp_port: int = 587
    
    # Slack settings
    slack_webhook_url: Optional[str] = None
    slack_channel: Optional[str] = None
    
    # Webhook settings
    webhook_urls: List[str] = field(default_factory=list)
    
    # Alert filtering
    min_severity: AlertSeverity = AlertSeverity.WARNING
    rate_limit_minutes: int = 60  # Minimum time between similar alerts
    max_alerts_per_hour: int = 10


class ProductionAutomationManager:
    """
    Production automation manager for pipeline testing infrastructure.
    
    Features:
    - Scheduled test execution with multiple schedules
    - Alert management and notifications  
    - Performance monitoring and regression detection
    - Integration with CI/CD systems
    - Production deployment validation
    - Health monitoring and recovery
    """
    
    def __init__(
        self,
        pipeline_test_suite,
        alert_config: Optional[AlertConfig] = None,
        data_dir: Path = None
    ):
        """Initialize production automation manager."""
        self.test_suite = pipeline_test_suite
        self.alert_config = alert_config or AlertConfig()
        self.data_dir = data_dir or Path("automation_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Automation state
        self.schedules: Dict[str, ScheduleConfig] = {}
        self.running_schedules: Dict[str, threading.Thread] = {}
        self.status = AutomationStatus.IDLE
        self.last_execution_results: Dict[str, AutomationResult] = {}
        
        # Alert state
        self.recent_alerts: List[Tuple[datetime, str, AlertSeverity]] = []
        self.alert_history: List[Dict[str, Any]] = []
        
        # Executor for background tasks
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="automation")
        self._shutdown_event = threading.Event()
        
    def add_schedule(self, config: ScheduleConfig) -> bool:
        """Add a new automated schedule."""
        if config.name in self.schedules:
            logger.warning(f"Schedule '{config.name}' already exists")
            return False
            
        # Validate schedule configuration
        if not self._validate_schedule_config(config):
            return False
            
        self.schedules[config.name] = config
        logger.info(f"Added schedule '{config.name}' ({config.schedule_type.value})")
        
        # Start schedule if automation is running and schedule is enabled
        if self.status == AutomationStatus.RUNNING and config.enabled:
            self._start_schedule(config)
            
        return True
    
    def remove_schedule(self, name: str) -> bool:
        """Remove an automated schedule."""
        if name not in self.schedules:
            logger.warning(f"Schedule '{name}' not found")
            return False
            
        # Stop running schedule
        if name in self.running_schedules:
            self._stop_schedule(name)
            
        del self.schedules[name]
        logger.info(f"Removed schedule '{name}'")
        return True
    
    def start_automation(self) -> bool:
        """Start the automation system."""
        if self.status == AutomationStatus.RUNNING:
            logger.warning("Automation already running")
            return False
            
        logger.info("Starting production automation system")
        self.status = AutomationStatus.RUNNING
        
        # Start all enabled schedules
        for config in self.schedules.values():
            if config.enabled:
                self._start_schedule(config)
                
        return True
    
    def stop_automation(self) -> bool:
        """Stop the automation system."""
        if self.status == AutomationStatus.STOPPED:
            logger.warning("Automation already stopped")
            return False
            
        logger.info("Stopping production automation system")
        self._shutdown_event.set()
        
        # Stop all running schedules
        for name in list(self.running_schedules.keys()):
            self._stop_schedule(name)
            
        self.status = AutomationStatus.STOPPED
        return True
    
    def pause_automation(self) -> bool:
        """Pause the automation system."""
        if self.status != AutomationStatus.RUNNING:
            logger.warning("Automation not running")
            return False
            
        logger.info("Pausing automation system")
        self.status = AutomationStatus.PAUSED
        
        # Don't stop threads, just mark as paused
        return True
    
    def resume_automation(self) -> bool:
        """Resume the automation system."""
        if self.status != AutomationStatus.PAUSED:
            logger.warning("Automation not paused")
            return False
            
        logger.info("Resuming automation system")
        self.status = AutomationStatus.RUNNING
        return True
    
    def _validate_schedule_config(self, config: ScheduleConfig) -> bool:
        """Validate schedule configuration."""
        if config.schedule_type == ScheduleType.CONTINUOUS and not config.interval_minutes:
            logger.error(f"Continuous schedule '{config.name}' requires interval_minutes")
            return False
            
        if config.schedule_type == ScheduleType.DAILY and not config.daily_time:
            logger.error(f"Daily schedule '{config.name}' requires daily_time")
            return False
            
        if config.schedule_type == ScheduleType.WEEKLY and (config.weekly_day is None or not config.daily_time):
            logger.error(f"Weekly schedule '{config.name}' requires weekly_day and daily_time")
            return False
            
        return True
    
    def _start_schedule(self, config: ScheduleConfig):
        """Start a specific schedule."""
        if config.name in self.running_schedules:
            logger.warning(f"Schedule '{config.name}' already running")
            return
            
        # Create and start thread for this schedule
        thread = threading.Thread(
            target=self._run_schedule,
            args=(config,),
            name=f"schedule_{config.name}",
            daemon=True
        )
        thread.start()
        self.running_schedules[config.name] = thread
        logger.info(f"Started schedule '{config.name}'")
    
    def _stop_schedule(self, name: str):
        """Stop a specific schedule."""
        if name not in self.running_schedules:
            return
            
        # Signal shutdown and wait for thread to finish
        thread = self.running_schedules[name]
        # Thread will check _shutdown_event periodically
        thread.join(timeout=30)  # Wait up to 30 seconds
        
        if thread.is_alive():
            logger.warning(f"Schedule '{name}' thread did not stop cleanly")
        else:
            logger.info(f"Stopped schedule '{name}'")
            
        del self.running_schedules[name]
    
    def _run_schedule(self, config: ScheduleConfig):
        """Run a schedule in a background thread."""
        logger.info(f"Schedule '{config.name}' thread started")
        
        while not self._shutdown_event.is_set():
            if self.status != AutomationStatus.RUNNING:
                # Sleep while paused, but check for shutdown
                if self._shutdown_event.wait(timeout=10):
                    break
                continue
                
            try:
                # Calculate next execution time
                next_execution = self._calculate_next_execution(config)
                
                if next_execution and next_execution <= datetime.now():
                    logger.info(f"Executing schedule '{config.name}'")
                    result = self._execute_scheduled_test(config)
                    
                    # Store result
                    self.last_execution_results[config.name] = result
                    
                    # Process alerts if needed
                    if result.alerts_generated:
                        for alert in result.alerts_generated:
                            self._send_alert(alert)
                    
                    # Save result to disk
                    self._save_execution_result(result)
                
                # Sleep until next check (but respond to shutdown quickly)
                sleep_time = min(60, config.interval_minutes or 60)  # Check at least every minute
                if self._shutdown_event.wait(timeout=sleep_time):
                    break
                    
            except Exception as e:
                logger.error(f"Error in schedule '{config.name}': {e}")
                # Create error result
                error_result = AutomationResult(
                    schedule_name=config.name,
                    execution_id=f"error_{int(time.time())}",
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    status=AutomationStatus.ERROR,
                    errors=[str(e)]
                )
                self.last_execution_results[config.name] = error_result
                
                # Send critical alert
                self._send_alert({
                    "severity": AlertSeverity.CRITICAL,
                    "title": f"Schedule '{config.name}' Error",
                    "message": f"Scheduled test execution failed: {e}",
                    "schedule_name": config.name
                })
                
                # Wait before retrying
                if self._shutdown_event.wait(timeout=300):  # 5 minute delay on error
                    break
        
        logger.info(f"Schedule '{config.name}' thread stopped")
    
    def _calculate_next_execution(self, config: ScheduleConfig) -> Optional[datetime]:
        """Calculate the next execution time for a schedule."""
        now = datetime.now()
        
        if config.schedule_type == ScheduleType.CONTINUOUS:
            # For continuous, check if enough time has passed since last execution
            if config.name in self.last_execution_results:
                last_execution = self.last_execution_results[config.name].end_time
                next_time = last_execution + timedelta(minutes=config.interval_minutes)
                return next_time
            else:
                return now  # First execution
                
        elif config.schedule_type == ScheduleType.DAILY:
            # Calculate next daily execution
            time_parts = config.daily_time.split(':')
            target_hour = int(time_parts[0])
            target_minute = int(time_parts[1])
            
            today_target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
            
            if now >= today_target:
                # Target time has passed today, schedule for tomorrow
                return today_target + timedelta(days=1)
            else:
                return today_target
                
        elif config.schedule_type == ScheduleType.WEEKLY:
            # Calculate next weekly execution
            time_parts = config.daily_time.split(':')
            target_hour = int(time_parts[0])
            target_minute = int(time_parts[1])
            
            # Calculate days until target weekday
            days_ahead = config.weekly_day - now.weekday()
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
                
            target_date = now + timedelta(days=days_ahead)
            return target_date.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        
        return None  # On-demand schedules don't have automatic execution times
    
    def _execute_scheduled_test(self, config: ScheduleConfig) -> AutomationResult:
        """Execute a scheduled test run."""
        execution_id = f"{config.name}_{int(time.time())}"
        start_time = datetime.now()
        
        result = AutomationResult(
            schedule_name=config.name,
            execution_id=execution_id,
            start_time=start_time,
            end_time=start_time,  # Will be updated
            status=AutomationStatus.RUNNING
        )
        
        try:
            logger.info(f"Starting test execution for schedule '{config.name}'")
            
            # Configure test suite
            self.test_suite.timeout_seconds = config.max_execution_time_minutes * 60
            self.test_suite.max_cost_per_pipeline = config.max_cost_dollars / 10  # Estimate per pipeline
            
            # Execute tests
            test_results = asyncio.run(
                self.test_suite.run_pipeline_tests(test_mode=config.test_mode)
            )
            
            # Update result with test outcomes
            result.end_time = datetime.now()
            result.status = AutomationStatus.IDLE
            result.total_tests = test_results.total_tests
            result.successful_tests = test_results.successful_tests
            result.failed_tests = test_results.failed_tests
            result.success_rate = test_results.success_rate / 100.0
            result.quality_score = test_results.average_quality_score
            result.execution_time_minutes = test_results.total_time / 60.0
            result.total_cost = test_results.total_cost
            
            # Check for alert conditions
            alerts = self._check_alert_conditions(config, result, test_results)
            result.alerts_generated = alerts
            
            # Generate warnings for issues
            if result.success_rate < config.min_success_rate:
                result.warnings.append(
                    f"Success rate {result.success_rate:.1%} below threshold {config.min_success_rate:.1%}"
                )
                
            if result.quality_score < config.min_quality_score:
                result.warnings.append(
                    f"Quality score {result.quality_score:.1f} below threshold {config.min_quality_score}"
                )
            
            logger.info(f"Completed test execution for schedule '{config.name}': "
                       f"{result.successful_tests}/{result.total_tests} passed")
            
        except Exception as e:
            result.end_time = datetime.now()
            result.status = AutomationStatus.ERROR
            result.errors.append(str(e))
            logger.error(f"Test execution failed for schedule '{config.name}': {e}")
        
        return result
    
    def _check_alert_conditions(
        self, 
        config: ScheduleConfig, 
        result: AutomationResult,
        test_results
    ) -> List[Dict[str, Any]]:
        """Check for conditions that should trigger alerts."""
        alerts = []
        
        # Failure alerts
        if config.alert_on_failure and result.failed_tests > 0:
            alerts.append({
                "severity": AlertSeverity.WARNING if result.success_rate > 0.5 else AlertSeverity.CRITICAL,
                "title": f"Test Failures in {config.name}",
                "message": f"{result.failed_tests}/{result.total_tests} tests failed "
                          f"(success rate: {result.success_rate:.1%})",
                "schedule_name": config.name,
                "execution_id": result.execution_id
            })
        
        # Quality alerts
        if result.quality_score < config.alert_thresholds.get("quality_score", 75.0):
            alerts.append({
                "severity": AlertSeverity.WARNING,
                "title": f"Quality Issues in {config.name}",
                "message": f"Quality score {result.quality_score:.1f} below threshold",
                "schedule_name": config.name,
                "execution_id": result.execution_id
            })
        
        # Performance regression alerts
        if config.alert_on_regression:
            # Check for significant execution time increases
            if config.name in self.last_execution_results:
                last_result = self.last_execution_results[config.name]
                if last_result.execution_time_minutes > 0:
                    time_increase = (result.execution_time_minutes - last_result.execution_time_minutes) / last_result.execution_time_minutes
                    
                    threshold = config.alert_thresholds.get("execution_time_increase", 0.5)
                    if time_increase > threshold:
                        alerts.append({
                            "severity": AlertSeverity.WARNING,
                            "title": f"Performance Regression in {config.name}",
                            "message": f"Execution time increased by {time_increase:.1%} "
                                      f"({result.execution_time_minutes:.1f} vs {last_result.execution_time_minutes:.1f} min)",
                            "schedule_name": config.name,
                            "execution_id": result.execution_id
                        })
        
        return alerts
    
    def _send_alert(self, alert: Dict[str, Any]):
        """Send an alert through configured channels."""
        severity = alert.get("severity", AlertSeverity.INFO)
        
        # Check if alert should be sent based on configuration
        if severity.value < self.alert_config.min_severity.value:
            return
        
        # Check rate limiting
        alert_key = f"{alert.get('title', 'Unknown')}:{alert.get('schedule_name', '')}"
        now = datetime.now()
        
        # Remove old alerts from recent list
        cutoff = now - timedelta(minutes=self.alert_config.rate_limit_minutes)
        self.recent_alerts = [
            (timestamp, key, sev) for timestamp, key, sev in self.recent_alerts
            if timestamp > cutoff
        ]
        
        # Check if similar alert was sent recently
        if any(key == alert_key for _, key, _ in self.recent_alerts):
            logger.debug(f"Skipping rate-limited alert: {alert_key}")
            return
        
        # Check hourly alert limit
        hour_ago = now - timedelta(hours=1)
        recent_alert_count = len([
            1 for timestamp, _, _ in self.recent_alerts
            if timestamp > hour_ago
        ])
        
        if recent_alert_count >= self.alert_config.max_alerts_per_hour:
            logger.warning(f"Alert rate limit exceeded ({recent_alert_count}/hour)")
            return
        
        # Add to recent alerts
        self.recent_alerts.append((now, alert_key, severity))
        
        # Send through configured channels
        alert_sent = False
        
        if self.alert_config.enable_logging:
            log_level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.CRITICAL: logging.CRITICAL,
                AlertSeverity.EMERGENCY: logging.CRITICAL
            }.get(severity, logging.INFO)
            
            logger.log(log_level, f"ALERT [{severity.value.upper()}] {alert['title']}: {alert['message']}")
            alert_sent = True
        
        # Email alerts (would need email implementation)
        if self.alert_config.enable_email and self.alert_config.email_recipients:
            # self._send_email_alert(alert)
            logger.info("Email alert would be sent (not implemented)")
            alert_sent = True
        
        # Slack alerts (would need Slack implementation) 
        if self.alert_config.enable_slack and self.alert_config.slack_webhook_url:
            # self._send_slack_alert(alert)
            logger.info("Slack alert would be sent (not implemented)")
            alert_sent = True
        
        # Webhook alerts (would need webhook implementation)
        if self.alert_config.enable_webhooks and self.alert_config.webhook_urls:
            # self._send_webhook_alert(alert)
            logger.info("Webhook alert would be sent (not implemented)")
            alert_sent = True
        
        if alert_sent:
            # Add to alert history
            self.alert_history.append({
                **alert,
                "timestamp": now.isoformat(),
                "sent": True
            })
            
    def _save_execution_result(self, result: AutomationResult):
        """Save execution result to disk."""
        result_file = self.data_dir / f"execution_{result.execution_id}.json"
        
        # Convert result to JSON-serializable format
        result_data = {
            "schedule_name": result.schedule_name,
            "execution_id": result.execution_id,
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat(),
            "status": result.status.value,
            "total_tests": result.total_tests,
            "successful_tests": result.successful_tests,
            "failed_tests": result.failed_tests,
            "success_rate": result.success_rate,
            "quality_score": result.quality_score,
            "execution_time_minutes": result.execution_time_minutes,
            "total_cost": result.total_cost,
            "errors": result.errors,
            "warnings": result.warnings,
            "alerts_generated": result.alerts_generated
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
    
    def get_automation_status(self) -> Dict[str, Any]:
        """Get current automation system status."""
        return {
            "status": self.status.value,
            "total_schedules": len(self.schedules),
            "running_schedules": len(self.running_schedules),
            "enabled_schedules": len([s for s in self.schedules.values() if s.enabled]),
            "last_executions": {
                name: {
                    "execution_id": result.execution_id,
                    "end_time": result.end_time.isoformat(),
                    "success_rate": result.success_rate,
                    "status": result.status.value
                }
                for name, result in self.last_execution_results.items()
            },
            "recent_alerts": len(self.recent_alerts),
            "alert_history_count": len(self.alert_history)
        }
    
    def trigger_on_demand_execution(self, schedule_name: str) -> bool:
        """Trigger an on-demand execution of a specific schedule."""
        if schedule_name not in self.schedules:
            logger.error(f"Schedule '{schedule_name}' not found")
            return False
            
        config = self.schedules[schedule_name]
        
        if not config.enabled:
            logger.warning(f"Schedule '{schedule_name}' is disabled")
            return False
            
        # Execute in background
        future = self.executor.submit(self._execute_scheduled_test, config)
        
        # Store future for monitoring (could be extended)
        logger.info(f"Triggered on-demand execution of '{schedule_name}'")
        return True


def create_default_production_schedules() -> List[ScheduleConfig]:
    """Create default production schedules for common use cases."""
    return [
        # Continuous monitoring with quick tests
        ScheduleConfig(
            name="continuous_monitoring",
            schedule_type=ScheduleType.CONTINUOUS,
            test_mode="quick",
            interval_minutes=60,  # Every hour
            max_execution_time_minutes=15,
            max_cost_dollars=1.0,
            alert_on_failure=True,
            alert_on_regression=True,
            tags=["monitoring", "continuous"]
        ),
        
        # Daily comprehensive testing
        ScheduleConfig(
            name="daily_comprehensive",
            schedule_type=ScheduleType.DAILY,
            test_mode="core",
            daily_time="06:00",  # 6 AM daily
            max_execution_time_minutes=45,
            max_cost_dollars=5.0,
            min_success_rate=0.9,
            min_quality_score=85.0,
            alert_on_failure=True,
            alert_on_regression=True,
            tags=["daily", "comprehensive"]
        ),
        
        # Weekly full validation
        ScheduleConfig(
            name="weekly_full_validation",
            schedule_type=ScheduleType.WEEKLY,
            test_mode="full",
            weekly_day=6,  # Sunday
            daily_time="02:00",  # 2 AM Sunday
            max_execution_time_minutes=120,
            max_cost_dollars=15.0,
            min_success_rate=0.95,
            min_quality_score=90.0,
            alert_on_failure=True,
            alert_on_regression=True,
            tags=["weekly", "full", "validation"]
        )
    ]


if __name__ == "__main__":
    # Example usage
    print("Production Automation Manager - Example Usage")
    print("=" * 50)
    
    # Create example configurations
    schedules = create_default_production_schedules()
    
    print("Default Production Schedules:")
    for schedule in schedules:
        print(f"  - {schedule.name}: {schedule.schedule_type.value} ({schedule.test_mode} mode)")
        if schedule.schedule_type == ScheduleType.CONTINUOUS:
            print(f"    Interval: {schedule.interval_minutes} minutes")
        elif schedule.schedule_type == ScheduleType.DAILY:
            print(f"    Time: {schedule.daily_time} daily")
        elif schedule.schedule_type == ScheduleType.WEEKLY:
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            print(f"    Time: {schedule.daily_time} on {days[schedule.weekly_day]}")
    
    print(f"\nTotal schedules: {len(schedules)}")
    print("Automation system ready for production deployment")