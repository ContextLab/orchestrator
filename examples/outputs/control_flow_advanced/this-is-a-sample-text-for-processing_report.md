# Multi-Stage Text Processing Report

## Original Text
{{ input_text }}

## Analysis
{{ analyze_text.result }}

## Enhancement
{% if enhance_text.result %}Text was enhanced{% else %}Original text was used{% endif %}

## Translations
Attempted translations to: {{ languages | join(', ') }}

Check the {{ output }}/translations/ directory for successful translations.
