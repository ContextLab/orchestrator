# System Information Report

## Python Environment
```
{{ check_python.output }}
```

## Installed Packages
```
{{ check_packages.output | default('No data science packages found') }}
```

## System Details
```
{{ system_info.output }}
```

## Disk Usage
```
{{ disk_usage.output }}
```

Generated on: 2025-07-28 23:57:28.658657
