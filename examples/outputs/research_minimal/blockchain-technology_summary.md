# Research Report: Blockchain Technology

**Date:** {{current_date}}  
**Sources Reviewed:** {{search_web.total_results}}

---

## Overview

{{summarize_results.result}}

---

## Key Findings

{% for finding in summarize_results.key_findings %}
{{loop.index}}. {{finding}}
{% endfor %}

---

## Summary

{{summarize_results.summary}}

---

## Sources

{% for result in search_web.results %}
{{loop.index}}. [{{result.title}}]({{result.url}})
{% endfor %}