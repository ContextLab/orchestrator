# Repository Organization Maintenance System
Generated: 2025-08-25 11:10:57
Repository: /Users/jmanning/orchestrator

## System Overview

This repository uses an automated organization maintenance system with:

### Core Components
- **Monitoring System**: Real-time organization violation detection
- **Validation System**: CI/CD integrated organization validation
- **Reporting System**: Automated compliance reporting and analytics
- **Safety Framework**: Multi-layer safety checks and backup systems

## Manual Maintenance Commands

### Health Monitoring
```bash
# Check current health status
python scripts/repository_organization_monitor.py --dashboard

# Run health check
python scripts/maintenance_system.py --health-check
```

### Validation
```bash
# Run CI validation suite
python scripts/organization_validator.py --suite ci_validation

# Run full validation
python scripts/organization_validator.py --suite full_validation
```

### Reporting
```bash
# Generate all reports
python scripts/organization_reporter.py --generate-all

# Show dashboard
python scripts/organization_reporter.py --dashboard
```

## Emergency Procedures

### If Organization Issues Occur
1. Check health status: `python scripts/maintenance_system.py --health-check`
2. Run validation: `python scripts/organization_validator.py --suite full_validation`
3. Review violations: `python scripts/repository_organization_monitor.py --scan-once`
4. Generate reports: `python scripts/organization_reporter.py --generate-all`

### System Status Monitoring
- **Health Reports**: `temp/maintenance/`
- **Validation Reports**: `temp/validation/`
- **Organization Reports**: `temp/reports/`

## Maintenance Schedule Recommendations

### Daily
- Monitor health dashboard
- Check for new violations

### Weekly
- Run full validation suite
- Review compliance reports

### Monthly
- Review archived files
- Clean up old temporary files
- Update maintenance documentation

## Current Status (as of 2025-08-25 11:10:57)

- **Health Score**: 100.0/100
- **Violations**: 0
- **Monitoring**: inactive
