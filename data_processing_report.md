# Data Processing Report

## Processing Summary

- **Input File**: {{ input_file }}
- **Output Path**: {{ output_path }}
- **Rows Processed**: {{ clean_data.row_count | default(0) }}
- **Data Profile**: 
  - Total Rows: {{ profile_data.processed_data.row_count | default(0) }}
  - Total Columns: {{ profile_data.processed_data.column_count | default(0) }}
  - Duplicate Rows: {{ profile_data.processed_data.duplicate_rows | default(0) }}

## Data Quality Assessment

### Quality Score: {{ quality_check.result.quality_score | default(0) | round(2) }}/1.0

### Issues Found
{% if quality_check.result.issues_found %}
{% for issue in quality_check.result.issues_found %}
- ⚠️ {{ issue }}
{% endfor %}
{% else %}
- ✅ No major issues detected
{% endif %}

## Column Statistics

| Column | Type | Missing % | Unique Values | Min | Max | Mean |
|--------|------|-----------|---------------|-----|-----|------|
{% for col_name, col_data in profile_data.processed_data.columns.items() %}
| {{ col_name }} | {{ col_data.data_type }} | {{ col_data.missing_percentage | round(1) }}% | {{ col_data.unique_count }} | {{ col_data.min | default('N/A') }} | {{ col_data.max | default('N/A') }} | {{ col_data.mean | default('N/A') | round(2) }} |
{% endfor %}

## Monthly Aggregations

{% if aggregate_monthly.processed_data.result %}
| Month | Total Quantity | Total Revenue | Avg Price | Order Count | Unique Customers |
|-------|----------------|---------------|-----------|-------------|------------------|
{% for row in aggregate_monthly.processed_data.result %}
| {{ row.order_month }} | {{ row.total_quantity | default(0) }} | ${{ row.total_revenue | default(0) | round(2) }} | ${{ row.average_price | default(0) | round(2) }} | {{ row.order_count | default(0) }} | {{ row.unique_customers | default(0) }} |
{% endfor %}
{% else %}
*No monthly aggregation data available*
{% endif %}

## Product Status Distribution (Pivot Table)

{% if pivot_analysis.processed_data.result %}
| Product | Pending | Processing | Shipped | Delivered | Total |
|---------|---------|------------|---------|-----------|-------|
{% for row in pivot_analysis.processed_data.result %}
| {{ row.product_name }} | {{ row.pending | default(0) }} | {{ row.processing | default(0) }} | {{ row.shipped | default(0) }} | {{ row.delivered | default(0) }} | {{ (row.pending | default(0)) + (row.processing | default(0)) + (row.shipped | default(0)) + (row.delivered | default(0)) }} |
{% endfor %}
{% else %}
*No pivot table data available*
{% endif %}

## Statistical Analysis

{% if analyze_trends.result %}
### Growth Rate: {{ analyze_trends.result.growth_rate | default('N/A') }}%

### Top Products by Revenue
{% for product in analyze_trends.result.top_products %}
{{ loop.index }}. **{{ product.product }}** - ${{ product.revenue | round(2) }}
{% endfor %}

### Seasonal Patterns
{% for pattern in analyze_trends.result.seasonal_patterns %}
- {{ pattern }}
{% endfor %}

### Anomalies Detected
{% for anomaly in analyze_trends.result.anomalies %}
- **{{ anomaly.month }}**: {{ anomaly.description }} ({{ anomaly.metric }})
{% endfor %}
{% else %}
*Statistical analysis pending*
{% endif %}

## Recommendations

{% if quality_check.result.recommendations %}
{% for rec in quality_check.result.recommendations %}
- 📌 {{ rec }}
{% endfor %}
{% endif %}

---

*Report generated on: {{ now() }}*
*Pipeline ID: {{ pipeline_id }}*
