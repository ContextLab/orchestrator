# Batch Processing Summary

Total files processed: {{ file_list | length }}

## Files
{% for file in file_list %}
- {{ file }}
{% endfor %}

## Results
All files have been processed and saved to {{ output_dir }}/
