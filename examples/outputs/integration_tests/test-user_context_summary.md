# Context Summary

**Generated on:** {{ execution['timestamp'] }}

## Current Context State

**Namespace**: conversation
**Active Keys**: {{ get_full_context['keys'] | join(', ') }}

### Details:
{% for key in get_full_context['keys'] %}
- **{{ key }}**: Stored in memory
{% endfor %}
