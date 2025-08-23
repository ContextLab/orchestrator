# Interactive Sales Data Processing Pipeline

**Pipeline**: `examples/interactive_pipeline.yaml`  
**Category**: User Interaction  
**Complexity**: Advanced  
**Key Features**: User prompts, Approval gates, Feedback collection, Dynamic choices, Human-in-the-loop

## Overview

The Interactive Sales Data Processing Pipeline demonstrates advanced human-in-the-loop workflows with real sales data processing. It showcases user prompts, approval gates, feedback collection, and dynamic content generation based on user selections, making it perfect for scenarios requiring human oversight and decision-making.

## Key Features Demonstrated

### 1. Dynamic User Choice Collection
```yaml
- id: get_processing_options
  tool: user-prompt
  parameters:
    prompt: "Select data processing method"
    input_type: "choice"
    choices: ["aggregate", "filter", "transform", "analyze"]
```

### 2. Conditional Logic Based on User Input
```yaml
choices: |
  {% if get_processing_options.value == 'aggregate' %}
  ["by_category", "by_region", "by_date", "top_products"]
  {% elif get_processing_options.value == 'filter' %}
  ["high_value", "electronics_only", "recent_orders", "top_customers"]
  {% endif %}
```

### 3. Approval Gates with Modifications
```yaml
- id: approve_results
  tool: approval-gate
  parameters:
    allow_modifications: true
    require_reason: true
    content: |
      ## Processing Summary
      {{ create_summary }}
```

### 4. Feedback Collection System
```yaml
- id: collect_feedback
  tool: feedback-collection
  parameters:
    questions:
      - id: "data_quality"
        type: "rating"
        scale: 5
      - id: "suggestions"
        type: "text"
```

## Pipeline Architecture

### Input Parameters
- **input_file** (optional): Path to sales data CSV (default: "data/sales_data.csv")
- **output_dir** (optional): Output directory path (default: "examples/outputs/interactive_pipeline")

### User Interaction Flow

1. **Processing Method Selection** - Choose between aggregate, filter, transform, or analyze
2. **Specific Operation Selection** - Dynamic options based on method choice
3. **Output Format Selection** - Choose between CSV, JSON, or Markdown
4. **Data Processing** - Automated processing based on user selections
5. **Approval Gate** - Review results with modification capability
6. **Feedback Collection** - Gather user experience metrics

### Processing Methods Available

#### Aggregation Options
- **by_category**: Group by product category with revenue totals
- **by_region**: Regional sales analysis with key metrics
- **by_date**: Daily sales summaries with top products
- **top_products**: Top 5 products by revenue analysis

#### Filter Options  
- **high_value**: Orders over $500 value
- **electronics_only**: Electronics category items only
- **recent_orders**: Orders from January 20, 2024 onwards
- **top_customers**: Top 3 customers by purchase amount

#### Transform Options
- **add_totals**: Add calculated fields (total, tax, final amount)
- **calculate_margins**: Add profit margin calculations
- **normalize_prices**: Price normalization and indexing
- **pivot_data**: Pivot table creation (dates vs categories)

#### Analysis Options
- **summary_stats**: Overall sales summary statistics
- **sales_trends**: Trend analysis with growth rates
- **customer_analysis**: Customer behavior analysis
- **product_performance**: Product performance metrics

## Usage Examples

### Basic Interactive Session
```bash
python scripts/run_pipeline.py examples/interactive_pipeline.yaml
# Follow prompts to select:
# 1. Processing method: aggregate
# 2. Specific operation: by_category
# 3. Output format: csv
# 4. Review and approve results
# 5. Provide feedback
```

### Custom Data File
```bash
python scripts/run_pipeline.py examples/interactive_pipeline.yaml \
  -i input_file="custom_sales_data.csv" \
  -i output_dir="custom_outputs"
```

### Pre-configured Execution
```bash
# Note: Interactive elements still require user input
python scripts/run_pipeline.py examples/interactive_pipeline.yaml \
  -i input_file="examples/data/sales_data.csv"
```

## Sample User Interaction Flows

### Scenario 1: Category Analysis
1. **User selects**: "aggregate" → "by_category" → "markdown"
2. **System processes**: Groups sales by Electronics/Furniture categories
3. **User reviews**: Markdown table with revenue by category
4. **User approves**: Results saved as processed_by_category.md
5. **User provides feedback**: Rates quality and ease of use

### Scenario 2: High-Value Order Filtering
1. **User selects**: "filter" → "high_value" → "json"
2. **System processes**: Filters orders > $500 with total value calculation
3. **User reviews**: JSON array of high-value transactions
4. **User modifies**: Adjusts threshold in approval gate
5. **System saves**: Modified results to processed_high_value.json

### Scenario 3: Profit Margin Analysis
1. **User selects**: "transform" → "calculate_margins" → "csv"
2. **System processes**: Adds cost basis, profit, and margin columns
3. **User reviews**: CSV with detailed margin calculations
4. **User approves**: Results saved for further analysis
5. **User feedback**: Suggests additional metrics

## Advanced Interactive Features

### Dynamic Content Generation
```yaml
prompt: |
  {% if get_processing_options.value == 'aggregate' %}
    {% if get_specific_operation.value == 'by_category' %}
    Group by category and calculate:
    - Total quantity sold
    - Total revenue (quantity * unit_price)
    {% endif %}
  {% endif %}
```

### Conditional Processing
```yaml
condition: "{{ approve_results.approved }}"
```

### User Input Validation
```yaml
required_questions: ["data_quality", "ease_of_use", "would_use_again"]
```

## Approval Gate Features

### Modification Support
```yaml
allow_modifications: true
require_reason: true
```

### Content Preview
```yaml
content: |
  ## Processing Summary
  {{ create_summary }}
  
  ## Processed Data (first 1000 chars)
  {{ process_data | truncate(1000) }}
```

### Approval Decision Logic
```yaml
path: "{{ output_dir }}/data/processed_{{ get_specific_operation.value }}.{{ get_output_format.value }}"
content: "{{ approve_results.modified_content | default(process_data) }}"
```

## Feedback Collection System

### Question Types
- **Rating**: 1-5 scale feedback (data_quality, ease_of_use)
- **Boolean**: Yes/No questions (processing_useful, would_use_again)
- **Text**: Open-ended suggestions and comments

### Feedback Analysis
```yaml
## User Feedback
- **Data Quality**: {{ collect_feedback.summary.rating_average | round(1) }}/5
- **Would Use Again**: {{ 'Yes' if collect_feedback.summary.boolean_summary.would_use_again else 'No' }}
```

### Feedback Storage
```yaml
save_to_file: "{{ output_dir }}/feedback/pipeline_feedback.json"
anonymous: false
```

## Output Generation

### Summary Report
The pipeline generates a comprehensive summary including:
- Processing configuration choices
- Data processing metrics
- Business insights summary
- Approval status and reasoning
- User feedback aggregation
- Execution timestamp

### File Outputs
- **Processed Data**: `data/processed_{operation}.{format}`
- **Summary Report**: `summary.md`
- **Feedback Data**: `feedback/pipeline_feedback.json`

## Technical Implementation

### User Prompt Integration
```yaml
tool: user-prompt
context: "cli"
input_type: "choice"
```

### Template Logic
```yaml
{% if condition %}
  Option A
{% elif other_condition %}
  Option B
{% else %}
  Default Option
{% endif %}
```

### Dependency Management
```yaml
dependencies:
  - get_processing_options
  - get_specific_operation
```

## Best Practices Demonstrated

1. **Progressive User Engagement**: Build complexity through sequential choices
2. **Dynamic Content**: Adapt processing based on user selections
3. **Transparent Previews**: Show users what will be processed
4. **Modification Support**: Allow users to adjust results before saving
5. **Comprehensive Feedback**: Collect both quantitative and qualitative feedback
6. **Clear Documentation**: Provide detailed summaries of all actions

## Common Use Cases

- **Data Analysis Workflows**: Interactive data exploration and processing
- **Report Generation**: User-guided report creation with approval
- **Business Intelligence**: Self-service analytics with oversight
- **Quality Assurance**: Human validation of automated processing
- **Training Systems**: Interactive learning with feedback loops
- **Custom Processing**: User-driven data transformation pipelines

## Troubleshooting

### User Input Issues
- Verify CLI context is properly configured
- Check choice options are valid and accessible
- Ensure prompts are clear and actionable

### Approval Gate Problems
- Confirm content renders properly with template variables
- Check modification permissions and user capabilities
- Validate approval logic conditions

### Feedback Collection Errors
- Ensure output directory exists for feedback storage
- Check required questions are properly specified
- Verify feedback format and structure

## Related Examples
- [simple_error_handling.md](simple_error_handling.md) - Error handling in interactive workflows
- [data_processing_pipeline.md](data_processing_pipeline.md) - Automated data processing
- [validation_pipeline.md](validation_pipeline.md) - Data validation with user oversight

## Technical Requirements

- **User Interface**: CLI context for prompts and approvals
- **File System**: Read/write access for data and outputs
- **Template Engine**: Jinja2 for conditional logic and content generation
- **Interactive Tools**: user-prompt, approval-gate, feedback-collection tools
- **Data Processing**: Text generation capabilities for data analysis

This pipeline demonstrates enterprise-ready interactive workflows combining automated processing with human oversight, making it ideal for scenarios requiring user judgment and approval in data processing workflows.