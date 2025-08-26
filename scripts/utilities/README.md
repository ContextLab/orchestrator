# Utility Scripts

This directory contains utility scripts for repository maintenance, organization, monitoring, and general system utilities.

## Scripts

### Repository Organization

- **`repository_organizer.py`** - Main orchestrator script for repository organization
  - Comprehensive repository structure management
  - Safe, atomic operations with rollback capabilities
  - Usage: `python scripts/utilities/repository_organizer.py [--scan|--plan|--execute|--dry-run]`

- **`root_directory_organizer.py`** - Focused organizer for root directory cleanup
  - Handles scattered files in the root directory
  - Safer, more targeted approach than full reorganization
  - Usage: `python scripts/utilities/root_directory_organizer.py`

- **`directory_structure_analyzer.py`** - Analyzes directory structure and patterns
  - Provides insights into repository organization
  - Identifies structural issues and improvements
  - Usage: `python scripts/utilities/directory_structure_analyzer.py`

- **`directory_structure_standardizer.py`** - Standardizes directory structure
  - Ensures consistent directory naming and organization
  - Applies standard conventions across the repository
  - Usage: `python scripts/utilities/directory_structure_standardizer.py`

### Repository Monitoring and Maintenance

- **`repository_organization_monitor.py`** - Monitors repository organization health
  - Continuous monitoring of organization standards
  - Alerts for organizational violations
  - Usage: `python scripts/utilities/repository_organization_monitor.py`

- **`organization_maintenance.py`** - Long-term maintenance and self-healing system
  - Automated cleanup schedules and procedures
  - Self-healing capabilities for minor violations
  - Usage: `python scripts/utilities/organization_maintenance.py`

- **`maintenance_system.py`** - Main maintenance system script
  - Coordinates various maintenance activities
  - Scheduling and automation of maintenance tasks
  - Usage: `python scripts/utilities/maintenance_system.py`

### Validation and Safety

- **`organization_validator.py`** - Validates organization structure
  - Checks compliance with organization standards
  - Validates file placements and naming conventions
  - Usage: `python scripts/utilities/organization_validator.py`

- **`safety_validator.py`** - Validates repository safety before operations
  - Ensures safe file operations and movements
  - Prevents accidental data loss
  - Usage: `python scripts/utilities/safety_validator.py`

### Reporting and Analysis

- **`organization_reporter.py`** - Reports on organization status
  - Generates comprehensive organization reports
  - Provides metrics and recommendations
  - Usage: `python scripts/utilities/organization_reporter.py`

- **`repository_scanner.py`** - Scans repository for files and patterns
  - Comprehensive file discovery and analysis
  - Pattern matching and categorization
  - Usage: `python scripts/utilities/repository_scanner.py`

### System Utilities

- **`setup_api_keys.py`** - Interactive API key setup
  - Secure configuration of model API keys
  - Environment variable management
  - Usage: `python scripts/utilities/setup_api_keys.py`

- **`generate_sample_data.py`** - Generates sample data for testing
  - Creates test datasets for pipeline validation
  - Supports various data formats and structures
  - Usage: `python scripts/utilities/generate_sample_data.py`

- **`fix_tool_names.py`** - Fixes tool naming issues
  - Resolves tool name conflicts and inconsistencies
  - Updates tool references across the codebase
  - Usage: `python scripts/utilities/fix_tool_names.py`

### Health Monitoring

- **`template_resolution_health_monitor.py`** - Monitors template resolution health
  - Tracks template resolution performance
  - Identifies template-related issues
  - Usage: `python scripts/utilities/template_resolution_health_monitor.py`

## Usage Examples

### Repository Organization

```bash
# Scan repository structure
python scripts/utilities/repository_organizer.py --scan

# Generate organization plan
python scripts/utilities/repository_organizer.py --plan

# Execute organization (dry-run first)
python scripts/utilities/repository_organizer.py --dry-run
python scripts/utilities/repository_organizer.py --execute

# Clean up root directory
python scripts/utilities/root_directory_organizer.py
```

### Maintenance and Monitoring

```bash
# Run maintenance system
python scripts/utilities/maintenance_system.py

# Monitor organization health
python scripts/utilities/repository_organization_monitor.py

# Generate organization report
python scripts/utilities/organization_reporter.py
```

### System Setup

```bash
# Set up API keys interactively
python scripts/utilities/setup_api_keys.py

# Generate sample data
python scripts/utilities/generate_sample_data.py

# Fix tool naming issues
python scripts/utilities/fix_tool_names.py
```

## Integration

These utility scripts integrate with:
- Repository organization standards
- Automated maintenance workflows
- CI/CD pipeline processes
- Development environment setup
- Quality assurance procedures

## Safety Features

All utility scripts include safety features:
- Backup creation before major operations
- Dry-run modes for previewing changes
- Rollback capabilities for reversible operations
- Validation checks before execution
- Comprehensive logging and error handling

## Scheduling and Automation

Many utility scripts support:
- Scheduled execution via cron or similar
- Automated maintenance workflows
- Integration with monitoring systems
- Alert and notification capabilities
- Self-healing and recovery procedures