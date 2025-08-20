# Data Processing Report

## Processing Summary

- **Input File**: sales_data.csv
- **Output Path**: examples/outputs/data_processing_pipeline
- **Rows Processed**: 5
- **Data Profile**: 
  - Total Rows: 5
  - Total Columns: 7
  - Duplicate Rows: 0

## Data Validation Results

✅ **Validation Passed**: All data conforms to schema requirements

## Data Quality Assessment

### Quality Score: 0.95/1.0

### Issues Found
- ⚠️ High outlier rate: 'quantity' column has 20.0% outliers (exceeds 10% threshold).

## Column Statistics

| Column | Type | Missing % | Unique Values | Min | Max | Mean |
|--------|------|-----------|---------------|-----|-----|------|
| order_id | date | 0.0% | 5 | N/A | N/A | N/A |
| customer_id | date | 0.0% | 3 | N/A | N/A | N/A |
| product_name | string | 0.0% | 4 | N/A | N/A | N/A |
| quantity | numeric | 0.0% | 5 | 1.0 | 10.0 | 4.2 |
| unit_price | numeric | 0.0% | 4 | 19.99 | 49.99 | 31.99 |
| order_date | date | 0.0% | 5 | N/A | N/A | N/A |
| status | string | 0.0% | 4 | N/A | N/A | N/A |

## Monthly Aggregations

| Month | Total Quantity | Total Revenue | Avg Price | Order Count | Unique Customers |
|-------|----------------|---------------|-----------|-------------|------------------|
| 2024-01 | 15.0 | $299.85 | $19.99 | 2 | 2 |
| 2024-01 | 3.0 | $89.97 | $29.99 | 1 | 1 |
| 2024-01 | 1.0 | $49.99 | $49.99 | 1 | 1 |
| 2024-01 | 2.0 | $79.98 | $39.99 | 1 | 1 |

## Product Status Distribution (Pivot Table)

| Product | Pending | Processing | Shipped | Delivered | Total |
|---------|---------|------------|---------|-----------|-------|
| Widget A | 0 | 0 | 0 | 15.0 | 15.0 |
| Widget B | 0 | 0 | 3.0 | 0 | 3.0 |
| Widget C | 0 | 1.0 | 0 | 0 | 1.0 |
| Widget D | 2.0 | 0 | 0 | 0 | 2.0 |

## Statistical Analysis

### Growth Rate: 0%

### Top Products by Revenue
1. **Widget A** - $299.852. **Widget B** - $89.973. **Widget D** - $79.984. **Widget C** - $49.99
### Seasonal Patterns
- Single-month data provided (2024-01); cannot identify seasonal patterns across multiple months.

### Anomalies Detected

## Recommendations

- 📌 Investigate the 'quantity' outlier records to determine root cause (data entry error, unit/scale mismatch, legitimate extremes).
- 📌 Correct or remove records confirmed as errors; document any changes.
- 📌 If outliers are legitimate, apply robust treatments (e.g., winsorization, trimming, or a log transform) before analysis or modeling.
- 📌 Add/strengthen validation rules at data capture for 'quantity' (acceptable range, data type checks, required fields) to prevent future anomalous values.
- 📌 Implement monitoring/alerting on outlier rate for 'quantity' (and other key numeric fields) so spikes are caught early.

---

*Report generated on: 2025-08-19 23:41:49.044083*
*Pipeline ID: data-processing-pipeline*