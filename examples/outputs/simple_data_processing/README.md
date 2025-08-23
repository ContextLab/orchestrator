# Simple Data Processing Output Examples

This directory contains real output examples from the Simple Data Processing Pipeline (`examples/simple_data_processing.yaml`).

## Pipeline Overview

The Simple Data Processing Pipeline demonstrates basic CSV data filtering and report generation. It reads a CSV file, filters records based on criteria, and produces both structured data and human-readable reports.

**Pipeline Documentation**: [simple_data_processing.md](../../../docs/examples/simple_data_processing.md)

## Generated Files

### Data Files
- **[filtered_output.csv](filtered_output.csv)** - Processed CSV containing only active projects
  - Original dataset: 10 project records
  - Filtered result: 5 active projects
  - Filter criteria: `status="active"`

- **[output_*.csv](.)** - Timestamped output files from multiple runs
  - Format: `output_YYYY-MM-DD-HHMMSS.csv`
  - Contains same filtered data with execution timestamps

### Reports
- **[analysis_report.md](analysis_report.md)** - Comprehensive processing report
  - Processing summary and metadata
  - Applied filter criteria
  - Data preview and results overview
  - Generated using pipeline templates

- **[report_*.md](.)** - Timestamped report files from multiple runs
  - Format: `report_YYYY-MM-DD-HHMMSS.md`
  - Contains same analysis with execution timestamps

## Sample Data Overview

### Original Input Data
The pipeline processes `examples/data/input.csv` with these projects:
- Project Alpha (active, $1,500)
- Project Beta (inactive, $2,300)
- Project Gamma (active, $3,100)
- Project Delta (inactive, $1,800)
- Project Epsilon (active, $2,700)
- Project Zeta (inactive, $3,500)
- Project Eta (active, $4,200)
- Project Theta (inactive, $2,100)
- Project Iota (active, $2,900)
- Project Kappa (inactive, $3,800)

### Filtered Results
After applying `status="active"` filter:
- Project Alpha, Gamma, Epsilon, Eta, Iota (5 projects)
- Total value: $14,500
- Date range: 2024-01-15 to 2024-06-22

## Usage Example

To reproduce these outputs:
```bash
python scripts/run_pipeline.py examples/simple_data_processing.yaml
```

Custom output directory:
```bash
python scripts/run_pipeline.py examples/simple_data_processing.yaml \
  -i output_path="my_custom_results"
```

## Technical Details

- **Processing Tool**: data-processing with filter action
- **Input Format**: CSV with headers (name, status, value, created_date)
- **Filter Logic**: Simple equality match on status column
- **Output Formats**: CSV (structured) + Markdown (human-readable)
- **Template Features**: Dynamic timestamps, data preview, metadata injection

## Related Examples
- [data_processing.md](../../../docs/examples/data_processing.md) - More advanced data processing
- [validation_pipeline.md](../../../docs/examples/validation_pipeline.md) - Data validation focus
- [statistical_analysis.md](../../../docs/examples/statistical_analysis.md) - Statistical processing

This example demonstrates the foundation patterns used in more complex data processing pipelines.