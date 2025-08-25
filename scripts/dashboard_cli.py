#!/usr/bin/env python3
"""
Dashboard Command-Line Interface for Issue #262.

Unified CLI for comprehensive pipeline validation dashboard system that provides:
- Dashboard generation and management
- Quality analysis and reporting
- Performance monitoring and analysis
- Historical trend analysis
- Export capabilities in multiple formats

Integration Point: Combines dashboard_generator.py, quality_analyzer.py, 
performance_monitor.py, and existing validation infrastructure.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add orchestrator to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scripts.dashboard_generator import PipelineDashboardGenerator, DashboardConfig
    from scripts.quality_analyzer import QualityAnalyzer
    from scripts.performance_monitor import PerformanceMonitor
    from scripts.organization_reporter import OrganizationReporter
except ImportError as e:
    logging.error(f"Failed to import dashboard modules: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DashboardCLI:
    """Unified command-line interface for dashboard operations."""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        
        # Initialize components
        self.dashboard_generator = None
        self.quality_analyzer = None
        self.performance_monitor = None
        self.organization_reporter = None
        
        logger.info(f"Dashboard CLI initialized for: {self.root_path}")

    def _initialize_components(self, config: Optional[DashboardConfig] = None):
        """Initialize dashboard components."""
        if not self.dashboard_generator:
            self.dashboard_generator = PipelineDashboardGenerator(str(self.root_path), config)
        
        if not self.quality_analyzer:
            self.quality_analyzer = QualityAnalyzer(str(self.root_path))
        
        if not self.performance_monitor:
            self.performance_monitor = PerformanceMonitor(str(self.root_path))
        
        if not self.organization_reporter:
            try:
                self.organization_reporter = OrganizationReporter(str(self.root_path))
            except Exception as e:
                logger.warning(f"Could not initialize organization reporter: {e}")

    async def generate_full_dashboard(self, formats: List[str] = None, 
                                    trend_days: int = 30, 
                                    output_dir: Optional[str] = None) -> Dict[str, Path]:
        """Generate comprehensive dashboard with all components."""
        print("🚀 Generating comprehensive pipeline validation dashboard...")
        
        # Configure dashboard
        config = DashboardConfig(
            export_formats=formats or ['json', 'html', 'csv'],
            trend_days=trend_days
        )
        
        self._initialize_components(config)
        
        # Override output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            self.dashboard_generator.dashboard_dir = output_path
        
        # Generate main dashboard
        dashboard_outputs = await self.dashboard_generator.generate_comprehensive_dashboard()
        
        print(f"✅ Dashboard generation complete!")
        return dashboard_outputs

    def analyze_quality(self, pipeline_name: Optional[str] = None, 
                       generate_report: bool = False,
                       output_file: Optional[str] = None) -> Dict[str, Any]:
        """Analyze quality metrics for pipelines."""
        print("🔍 Analyzing pipeline quality metrics...")
        
        self._initialize_components()
        
        if pipeline_name:
            # Analyze specific pipeline
            metrics = self.quality_analyzer.analyze_pipeline_quality(pipeline_name)
            result = {
                'pipeline': pipeline_name,
                'overall_score': metrics.overall_score,
                'category_scores': metrics.category_scores,
                'issues_count': len(metrics.issues),
                'top_issues': [issue.description for issue in metrics.issues[:5]]
            }
            
            if generate_report:
                result['detailed_metrics'] = {
                    'pipeline_name': metrics.pipeline_name,
                    'overall_score': metrics.overall_score,
                    'category_scores': metrics.category_scores,
                    'issues': [
                        {
                            'type': issue.issue_type,
                            'severity': issue.severity,
                            'description': issue.description,
                            'suggestion': issue.suggestion
                        } for issue in metrics.issues
                    ]
                }
        else:
            # Analyze all pipelines
            all_metrics = self.quality_analyzer.analyze_all_pipelines()
            
            if generate_report:
                result = self.quality_analyzer.generate_quality_report(all_metrics)
            else:
                # Simple summary
                scores = [m.overall_score for m in all_metrics.values()]
                result = {
                    'total_pipelines': len(all_metrics),
                    'average_quality_score': sum(scores) / len(scores) if scores else 0,
                    'best_pipeline': max(all_metrics.items(), key=lambda x: x[1].overall_score)[0] if all_metrics else None,
                    'worst_pipeline': min(all_metrics.items(), key=lambda x: x[1].overall_score)[0] if all_metrics else None,
                    'pipelines_needing_attention': len([s for s in scores if s < 70])
                }
        
        # Save output if requested
        if output_file and generate_report:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"📄 Quality report saved to: {output_file}")
        
        return result

    def monitor_performance(self, pipeline_name: str, action: str = "summary") -> Dict[str, Any]:
        """Monitor or analyze pipeline performance."""
        print(f"📊 Performance monitoring for pipeline: {pipeline_name}")
        
        self._initialize_components()
        
        if action == "summary":
            result = self.performance_monitor.get_performance_summary(pipeline_name)
        elif action == "start":
            monitor_id = self.performance_monitor.start_monitoring(pipeline_name)
            result = {'monitor_id': monitor_id, 'status': 'monitoring_started'}
        elif action == "stop":
            profile = self.performance_monitor.stop_monitoring(pipeline_name)
            result = {
                'status': 'monitoring_stopped',
                'profile': profile.__dict__ if profile else None
            }
        else:
            result = {'error': f'Unknown action: {action}'}
        
        return result

    def generate_organization_report(self) -> Dict[str, Any]:
        """Generate organization health report."""
        print("🏥 Generating organization health report...")
        
        self._initialize_components()
        
        if not self.organization_reporter:
            return {'error': 'Organization reporter not available'}
        
        # Generate dashboard report
        dashboard_report = self.organization_reporter.generate_dashboard_report()
        
        # Generate detailed JSON report
        json_report = self.organization_reporter.generate_detailed_json_report()
        
        # Save reports
        saved_reports = self.organization_reporter.save_all_reports()
        
        return {
            'dashboard_preview': dashboard_report.split('\n')[:20],  # First 20 lines
            'summary': json_report['current_metrics'],
            'saved_reports': {k: str(v) for k, v in saved_reports.items()}
        }

    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old dashboard data."""
        print(f"🧹 Cleaning up data older than {days_to_keep} days...")
        
        self._initialize_components()
        
        # Cleanup performance metrics
        self.performance_monitor.cleanup_old_metrics(days_to_keep)
        
        # Could add cleanup for other components here
        print("✅ Cleanup complete!")

    def show_status(self) -> Dict[str, Any]:
        """Show current dashboard system status."""
        print("📋 Dashboard system status...")
        
        self._initialize_components()
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'root_path': str(self.root_path),
            'components_initialized': {
                'dashboard_generator': self.dashboard_generator is not None,
                'quality_analyzer': self.quality_analyzer is not None,
                'performance_monitor': self.performance_monitor is not None,
                'organization_reporter': self.organization_reporter is not None
            }
        }
        
        # Check for existing outputs
        outputs_dir = self.root_path / "examples" / "outputs"
        if outputs_dir.exists():
            pipeline_dirs = [d.name for d in outputs_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            status['available_pipelines'] = len(pipeline_dirs)
            status['pipeline_examples'] = pipeline_dirs[:10]  # First 10
        else:
            status['available_pipelines'] = 0
            status['pipeline_examples'] = []
        
        # Check for dashboard outputs
        dashboard_dir = self.root_path / "temp" / "dashboards"
        if dashboard_dir.exists():
            dashboard_files = list(dashboard_dir.glob("pipeline_dashboard_*.html"))
            status['recent_dashboards'] = len(dashboard_files)
            status['latest_dashboard'] = str(max(dashboard_files)) if dashboard_files else None
        else:
            status['recent_dashboards'] = 0
            status['latest_dashboard'] = None
        
        return status

    def open_dashboard(self, dashboard_file: Optional[str] = None):
        """Open dashboard in web browser."""
        dashboard_dir = self.root_path / "temp" / "dashboards"
        
        if dashboard_file:
            dashboard_path = Path(dashboard_file)
        else:
            # Find most recent dashboard
            dashboard_files = list(dashboard_dir.glob("pipeline_dashboard_*.html"))
            if not dashboard_files:
                print("❌ No dashboard files found. Generate one first with --generate")
                return
            dashboard_path = max(dashboard_files)
        
        if dashboard_path.exists():
            print(f"🌐 Opening dashboard: {dashboard_path}")
            webbrowser.open(f"file://{dashboard_path.absolute()}")
        else:
            print(f"❌ Dashboard file not found: {dashboard_path}")

    async def export_data(self, format_type: str, output_file: str) -> bool:
        """Export dashboard data in specified format."""
        print(f"📤 Exporting dashboard data to {format_type} format...")
        
        self._initialize_components()
        
        try:
            if format_type.lower() == 'csv':
                # Export pipeline metrics to CSV
                metrics = await self.dashboard_generator.collect_all_pipeline_metrics()
                csv_content = self.dashboard_generator.export_to_csv(metrics)
                
                with open(output_file, 'w') as f:
                    f.write(csv_content)
                    
            elif format_type.lower() == 'json':
                # Export comprehensive JSON report
                metrics = await self.dashboard_generator.collect_all_pipeline_metrics()
                summary = self.dashboard_generator.generate_executive_summary(metrics)
                json_data = self.dashboard_generator.export_to_json(metrics, summary)
                
                with open(output_file, 'w') as f:
                    json.dump(json_data, f, indent=2, default=str)
                    
            else:
                print(f"❌ Unsupported export format: {format_type}")
                return False
            
            print(f"✅ Data exported successfully to: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return False


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Pipeline Validation Dashboard CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate comprehensive dashboard
  python scripts/dashboard_cli.py --generate
  
  # Generate dashboard with specific formats
  python scripts/dashboard_cli.py --generate --formats json html
  
  # Analyze quality for specific pipeline
  python scripts/dashboard_cli.py --quality --pipeline control_flow_advanced.yaml
  
  # Analyze quality for all pipelines with detailed report
  python scripts/dashboard_cli.py --quality --all --report --output quality_report.json
  
  # Monitor performance
  python scripts/dashboard_cli.py --performance pipeline_name --action summary
  
  # Generate organization report
  python scripts/dashboard_cli.py --organization
  
  # Show system status
  python scripts/dashboard_cli.py --status
  
  # Open latest dashboard in browser
  python scripts/dashboard_cli.py --open
  
  # Export data
  python scripts/dashboard_cli.py --export json --output dashboard_data.json
  
  # Cleanup old data
  python scripts/dashboard_cli.py --cleanup 60
"""
    )
    
    # Main actions
    parser.add_argument("--generate", action='store_true', 
                       help="Generate comprehensive dashboard")
    parser.add_argument("--quality", action='store_true',
                       help="Analyze pipeline quality metrics")
    parser.add_argument("--performance", type=str, metavar='PIPELINE',
                       help="Monitor/analyze pipeline performance")
    parser.add_argument("--organization", action='store_true',
                       help="Generate organization health report")
    parser.add_argument("--status", action='store_true',
                       help="Show dashboard system status")
    parser.add_argument("--open", action='store_true',
                       help="Open dashboard in web browser")
    parser.add_argument("--export", choices=['json', 'csv'],
                       help="Export dashboard data")
    parser.add_argument("--cleanup", type=int, metavar='DAYS',
                       help="Cleanup data older than N days")
    
    # Options for --generate
    parser.add_argument("--formats", nargs='+', 
                       choices=['json', 'html', 'csv'],
                       default=['json', 'html', 'csv'],
                       help="Output formats for dashboard")
    parser.add_argument("--trend-days", type=int, default=30,
                       help="Days for trend analysis")
    parser.add_argument("--output-dir", type=str,
                       help="Output directory for generated files")
    
    # Options for --quality
    parser.add_argument("--pipeline", type=str,
                       help="Specific pipeline to analyze")
    parser.add_argument("--all", action='store_true',
                       help="Analyze all available pipelines")
    parser.add_argument("--report", action='store_true',
                       help="Generate detailed report")
    
    # Options for --performance
    parser.add_argument("--action", choices=['summary', 'start', 'stop'],
                       default='summary',
                       help="Performance monitoring action")
    
    # General options
    parser.add_argument("--root", default=".",
                       help="Repository root path")
    parser.add_argument("--output", type=str,
                       help="Output file for reports")
    parser.add_argument("--verbose", action='store_true',
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize CLI
    cli = DashboardCLI(args.root)
    
    try:
        # Execute requested action
        if args.generate:
            outputs = await cli.generate_full_dashboard(
                formats=args.formats,
                trend_days=args.trend_days,
                output_dir=args.output_dir
            )
            
            print(f"📊 Generated {len(outputs)} output files:")
            for format_type, path in outputs.items():
                print(f"  {format_type.upper()}: {path}")
                
        elif args.quality:
            pipeline = args.pipeline if not args.all else None
            result = cli.analyze_quality(
                pipeline_name=pipeline,
                generate_report=args.report,
                output_file=args.output
            )
            
            if not args.report:
                print(f"🔍 Quality Analysis Results:")
                print(json.dumps(result, indent=2, default=str))
                
        elif args.performance:
            result = cli.monitor_performance(args.performance, args.action)
            print(f"📊 Performance Results:")
            print(json.dumps(result, indent=2, default=str))
            
        elif args.organization:
            result = cli.generate_organization_report()
            print(f"🏥 Organization Report:")
            print("\n".join(result.get('dashboard_preview', [])))
            if result.get('saved_reports'):
                print(f"\n📄 Reports saved:")
                for report_type, path in result['saved_reports'].items():
                    print(f"  {report_type}: {path}")
                    
        elif args.status:
            result = cli.show_status()
            print(f"📋 Dashboard System Status:")
            print(json.dumps(result, indent=2, default=str))
            
        elif args.open:
            cli.open_dashboard()
            
        elif args.export:
            if not args.output:
                print("❌ --output required for export")
                sys.exit(1)
            
            success = await cli.export_data(args.export, args.output)
            if not success:
                sys.exit(1)
                
        elif args.cleanup:
            cli.cleanup_old_data(args.cleanup)
            
        else:
            print("No action specified. Use --help for usage information.")
            parser.print_help()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())