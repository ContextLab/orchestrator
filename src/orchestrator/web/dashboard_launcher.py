#!/usr/bin/env python3
"""
Monitoring Dashboard Launcher

Simple script to launch the monitoring dashboard server with proper configuration
and integration with the wrapper monitoring system.

Usage:
    python -m orchestrator.web.dashboard_launcher
    
    or with custom settings:
    
    python -m orchestrator.web.dashboard_launcher --host 0.0.0.0 --port 8080 --debug
"""

import argparse
import logging
import sys
from typing import Optional

from ..analytics.performance_monitor import PerformanceMonitor
from ..core.wrapper_monitoring import WrapperMonitoring
from .monitoring_dashboard import create_monitoring_dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_monitoring_instances() -> tuple[WrapperMonitoring, PerformanceMonitor]:
    """
    Create monitoring instances with proper configuration.
    
    Returns:
        Tuple of (WrapperMonitoring, PerformanceMonitor) instances
    """
    logger.info("Initializing monitoring systems...")
    
    # Create performance monitor
    performance_monitor = PerformanceMonitor()
    
    # Create wrapper monitoring with integration
    wrapper_monitoring = WrapperMonitoring(
        performance_monitor=performance_monitor,
        retention_days=7,  # Keep 7 days of data
        max_operations_in_memory=50000,  # Large memory buffer
        health_check_interval_minutes=2  # Frequent health checks
    )
    
    logger.info("Monitoring systems initialized successfully")
    return wrapper_monitoring, performance_monitor


def setup_sample_data(wrapper_monitoring: WrapperMonitoring) -> None:
    """
    Setup sample data for demonstration purposes.
    
    Args:
        wrapper_monitoring: Wrapper monitoring instance
    """
    logger.info("Setting up sample monitoring data...")
    
    import time
    from datetime import datetime, timedelta
    import random
    
    # Sample wrapper names
    wrappers = ['RouteLLM', 'POML', 'External_API', 'GPT_Wrapper']
    operations = ['chat_completion', 'embedding', 'function_call', 'streaming']
    
    # Create sample operations for the past hour
    for i in range(100):
        wrapper_name = random.choice(wrappers)
        operation_type = random.choice(operations)
        
        # Start operation
        op_id = f"demo_op_{i}"
        wrapper_monitoring.start_operation(op_id, wrapper_name, operation_type)
        
        # Simulate some processing time
        processing_time = random.uniform(50, 500)  # 50-500ms
        
        # Simulate success/failure (90% success rate)
        if random.random() < 0.9:
            # Success
            custom_metrics = {
                'tokens_used': random.randint(10, 1000),
                'cost_estimate': random.uniform(0.001, 0.1),
                'quality_score': random.uniform(0.7, 1.0)
            }
            wrapper_monitoring.record_success(op_id, custom_metrics=custom_metrics)
        else:
            # Error
            error_messages = [
                "Rate limit exceeded",
                "Network timeout",
                "Invalid API key", 
                "Service unavailable",
                "Request too large"
            ]
            wrapper_monitoring.record_error(op_id, random.choice(error_messages))
        
        # End operation
        wrapper_monitoring.end_operation(op_id)
    
    logger.info("Sample data setup complete")


def main():
    """Main entry point for dashboard launcher."""
    parser = argparse.ArgumentParser(
        description="Launch Orchestrator Monitoring Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Launch with defaults (localhost:5000)
  %(prog)s --port 8080             # Launch on port 8080
  %(prog)s --host 0.0.0.0          # Listen on all interfaces
  %(prog)s --debug                 # Enable debug mode
  %(prog)s --sample-data           # Include sample data for demo
        """
    )
    
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host address to bind to (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port number to bind to (default: 5000)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--sample-data',
        action='store_true',
        help='Generate sample monitoring data for demonstration'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Create monitoring instances
        wrapper_monitoring, performance_monitor = create_monitoring_instances()
        
        # Setup sample data if requested
        if args.sample_data:
            setup_sample_data(wrapper_monitoring)
        
        # Create dashboard
        dashboard = create_monitoring_dashboard(wrapper_monitoring, performance_monitor)
        
        logger.info(f"Starting monitoring dashboard on {args.host}:{args.port}")
        
        if args.debug:
            logger.warning("Debug mode enabled - do not use in production!")
        
        print("="*60)
        print("🚀 Orchestrator Monitoring Dashboard")
        print("="*60)
        print(f"📊 Dashboard URL: http://{args.host}:{args.port}")
        print(f"🔍 Health Check: http://{args.host}:{args.port}/health")
        print(f"📈 System API: http://{args.host}:{args.port}/api/system/health")
        print("="*60)
        
        if args.sample_data:
            print("📝 Sample data loaded for demonstration")
            print("="*60)
        
        print("Press Ctrl+C to stop the server")
        print()
        
        # Run the dashboard
        dashboard.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
        
    except KeyboardInterrupt:
        logger.info("Dashboard shutdown requested by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()