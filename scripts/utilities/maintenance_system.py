#!/usr/bin/env python3
"""
Maintenance System for Issue #255 Stream C - Simplified Version.

Long-term maintenance procedures with self-healing capabilities.
Integrates all components from Streams A, B, and C.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import our infrastructure
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from repository_organization_monitor import RepositoryOrganizationMonitor
    from organization_validator import OrganizationValidator
    from organization_reporter import OrganizationReporter
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MaintenanceSystem:
    """Simplified maintenance system for repository organization."""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        
        # Initialize components
        self.monitor = RepositoryOrganizationMonitor(str(self.root_path))
        self.validator = OrganizationValidator(str(self.root_path))
        self.reporter = OrganizationReporter(str(self.root_path))
        
        # Setup directories
        self.maintenance_dir = self.root_path / "temp" / "maintenance"
        self.maintenance_dir.mkdir(parents=True, exist_ok=True)
        
        self.docs_dir = self.root_path / "docs" / "maintenance"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Maintenance System initialized for: {self.root_path}")

    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check."""
        logger.info("Running maintenance health check...")
        
        # Get current health
        health_report = self.monitor.get_current_health_report()
        
        # Run validation
        validation_report = self.validator.validate('ci_validation')
        
        # Generate summary
        health_summary = {
            'timestamp': datetime.now().isoformat(),
            'health_score': health_report['health_score'],
            'validation_status': validation_report.overall_status,
            'violations_count': health_report['stats']['violations_detected'],
            'monitoring_status': health_report['monitoring_status'],
            'recommendations': []
        }
        
        # Add recommendations based on health
        if health_report['health_score'] < 80:
            health_summary['recommendations'].append("Health score below 80 - run cleanup procedures")
        
        if validation_report.failed_tests > 0:
            health_summary['recommendations'].append(f"Validation failures detected - review {validation_report.failed_tests} failed tests")
        
        # Save health check result
        health_check_file = self.maintenance_dir / f"health_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(health_check_file, 'w') as f:
            json.dump(health_summary, f, indent=2, default=str)
        
        logger.info(f"Health check complete - Score: {health_report['health_score']:.1f}")
        return health_summary

    def generate_maintenance_documentation(self) -> Path:
        """Generate maintenance documentation."""
        logger.info("Generating maintenance documentation...")
        
        doc_content = [
            "# Repository Organization Maintenance System",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Repository: {self.root_path}",
            "",
            "## System Overview",
            "",
            "This repository uses an automated organization maintenance system with:",
            "",
            "### Core Components",
            "- **Monitoring System**: Real-time organization violation detection",
            "- **Validation System**: CI/CD integrated organization validation", 
            "- **Reporting System**: Automated compliance reporting and analytics",
            "- **Safety Framework**: Multi-layer safety checks and backup systems",
            "",
            "## Manual Maintenance Commands",
            "",
            "### Health Monitoring",
            "```bash",
            "# Check current health status",
            "python scripts/repository_organization_monitor.py --dashboard",
            "",
            "# Run health check", 
            "python scripts/maintenance_system.py --health-check",
            "```",
            "",
            "### Validation",
            "```bash",
            "# Run CI validation suite",
            "python scripts/organization_validator.py --suite ci_validation",
            "",
            "# Run full validation",
            "python scripts/organization_validator.py --suite full_validation",
            "```",
            "",
            "### Reporting", 
            "```bash",
            "# Generate all reports",
            "python scripts/organization_reporter.py --generate-all",
            "",
            "# Show dashboard",
            "python scripts/organization_reporter.py --dashboard",
            "```",
            "",
            "## Emergency Procedures",
            "",
            "### If Organization Issues Occur",
            "1. Check health status: `python scripts/maintenance_system.py --health-check`",
            "2. Run validation: `python scripts/organization_validator.py --suite full_validation`",
            "3. Review violations: `python scripts/repository_organization_monitor.py --scan-once`",
            "4. Generate reports: `python scripts/organization_reporter.py --generate-all`",
            "",
            "### System Status Monitoring",
            "- **Health Reports**: `temp/maintenance/`",
            "- **Validation Reports**: `temp/validation/`", 
            "- **Organization Reports**: `temp/reports/`",
            "",
            "## Maintenance Schedule Recommendations",
            "",
            "### Daily",
            "- Monitor health dashboard",
            "- Check for new violations",
            "",
            "### Weekly", 
            "- Run full validation suite",
            "- Review compliance reports",
            "",
            "### Monthly",
            "- Review archived files",
            "- Clean up old temporary files",
            "- Update maintenance documentation",
            ""
        ]
        
        # Add current status
        try:
            health_report = self.monitor.get_current_health_report()
            doc_content.extend([
                f"## Current Status (as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
                "",
                f"- **Health Score**: {health_report['health_score']:.1f}/100",
                f"- **Violations**: {health_report['stats']['violations_detected']}",
                f"- **Monitoring**: {health_report['monitoring_status']}",
                ""
            ])
        except:
            doc_content.extend([
                "## Current Status",
                "- Status check failed - manual verification needed",
                ""
            ])
        
        # Save documentation
        doc_path = self.docs_dir / "README.md"
        with open(doc_path, 'w') as f:
            f.write('\n'.join(doc_content))
        
        logger.info(f"Documentation generated: {doc_path}")
        return doc_path

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        try:
            health_report = self.monitor.get_current_health_report()
            validation_report = self.validator.validate('pre_commit')
            
            return {
                'timestamp': datetime.now().isoformat(),
                'health_score': health_report['health_score'],
                'validation_status': validation_report.overall_status,
                'violations_count': health_report['stats']['violations_detected'],
                'monitoring_active': health_report['monitoring_status'] == 'active',
                'system_operational': True
            }
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'system_operational': False
            }


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Repository Organization Maintenance System")
    parser.add_argument("--root", default=".", help="Repository root path")
    parser.add_argument("--health-check", action='store_true', help="Run health check")
    parser.add_argument("--status", action='store_true', help="Show system status") 
    parser.add_argument("--generate-docs", action='store_true', help="Generate documentation")
    parser.add_argument("--verbose", action='store_true', help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    maintenance = MaintenanceSystem(args.root)
    
    if args.health_check:
        result = maintenance.run_health_check()
        print("🏥 Health Check Results:")
        print(f"  Health Score: {result['health_score']:.1f}/100")
        print(f"  Validation Status: {result['validation_status']}")
        print(f"  Violations: {result['violations_count']}")
        if result['recommendations']:
            print("  Recommendations:")
            for rec in result['recommendations']:
                print(f"    - {rec}")
    
    elif args.status:
        status = maintenance.get_system_status()
        print("📊 System Status:")
        if status.get('system_operational', False):
            print(f"  Health Score: {status['health_score']:.1f}/100")
            print(f"  Validation: {status['validation_status']}")
            print(f"  Violations: {status['violations_count']}")
            print(f"  Monitoring: {'Active' if status['monitoring_active'] else 'Inactive'}")
        else:
            print(f"  Error: {status.get('error', 'Unknown error')}")
    
    elif args.generate_docs:
        doc_path = maintenance.generate_maintenance_documentation()
        print(f"📝 Documentation generated: {doc_path}")
    
    else:
        print("Use --health-check, --status, or --generate-docs")


if __name__ == "__main__":
    main()