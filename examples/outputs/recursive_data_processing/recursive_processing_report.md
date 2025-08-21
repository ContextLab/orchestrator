# Recursive Data Processing Results

## Processing Summary
- Total Iterations: {{ increment_iteration.new_value }}
- Final Quality Score: {{ update_quality_score.full_state.quality_score }}
- Quality Threshold: {{ parameters.quality_threshold }}
- Termination Reason: {{ 'Quality threshold met' if update_quality_score.full_state.quality_score >= parameters.quality_threshold else 'Maximum iterations reached' }}

## Quality Metrics History
{{ validate_quality.metrics | json }}

## Output
- Final processed data saved to: processed_data_final.csv
