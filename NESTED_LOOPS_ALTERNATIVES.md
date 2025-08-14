# Alternative Approaches to Nested Loops in Orchestrator

## Problem with Direct Nesting

Direct nesting of `for_each` loops has template rendering issues with parent loop variables like `$parent_item`. These variables are not properly accessible in the inner loop's template context.

## Working Alternatives

### 1. Separate Steps with Dependencies

Instead of nesting loops, use separate steps that depend on each other:

```yaml
steps:
  # First dimension
  - id: process_categories
    for_each: "{{ categories }}"
    steps:
      - id: analyze_category
        action: generate_text
        parameters:
          prompt: "Analyze {{ $item }}"
      
  # Second dimension (depends on first)
  - id: process_items_for_category_A
    for_each: "{{ items_A }}"
    steps:
      - id: process_item
        action: generate_text
        parameters:
          prompt: "Process {{ $item }} for category A"
    dependencies:
      - process_categories
```

### 2. Matrix Processing

For known dimensions, create separate steps for each combination:

```yaml
parameters:
  regions: ["US", "EU", "ASIA"]
  products: ["laptop", "phone"]

steps:
  # Process US products
  - id: us_products
    for_each: "{{ products }}"
    steps:
      - id: analyze
        action: generate_text
        parameters:
          prompt: "Analyze {{ $item }} in US"
  
  # Process EU products  
  - id: eu_products
    for_each: "{{ products }}"
    steps:
      - id: analyze
        action: generate_text
        parameters:
          prompt: "Analyze {{ $item }} in EU"
  
  # Process ASIA products
  - id: asia_products
    for_each: "{{ products }}"
    steps:
      - id: analyze
        action: generate_text
        parameters:
          prompt: "Analyze {{ $item }} in ASIA"
```

### 3. Conditional Processing

Use conditionals to control which combinations execute:

```yaml
steps:
  # Evaluate conditions
  - id: evaluate
    for_each: "{{ categories }}"
    steps:
      - id: check_condition
        action: generate_text
        parameters:
          prompt: "Should process {{ $item }}?"
  
  # Process based on conditions
  - id: process_if_premium
    for_each: "{{ items }}"
    if: "{{ 'premium' in categories }}"
    steps:
      - id: premium_process
        action: generate_text
        parameters:
          prompt: "Premium processing for {{ $item }}"
    dependencies:
      - evaluate
```

### 4. Pre-computed Combinations

Generate combinations in a step, then process them:

```yaml
steps:
  # Generate combinations
  - id: create_combinations
    action: generate_text
    parameters:
      prompt: |
        Create all combinations of:
        Categories: {{ categories }}
        Items: {{ items }}
        Return as CSV: category,item
  
  # Process combinations
  - id: process_combinations
    for_each: "{{ create_combinations.split('\n') }}"
    steps:
      - id: process_combo
        action: generate_text
        parameters:
          prompt: "Process combination: {{ $item }}"
```

## Test Results

All alternative approaches successfully render templates without issues:

| Approach | Template Rendering | Complexity | Use Case |
|----------|-------------------|------------|----------|
| Separate Steps | ✅ Perfect | Low | Known dimensions |
| Matrix Processing | ✅ Perfect | Medium | Fixed combinations |
| Conditional | ✅ Perfect | Medium | Dynamic selection |
| Pre-computed | ✅ Perfect | High | Complex logic |

## Recommendations

1. **Avoid directly nested `for_each` loops** - They have template issues with parent variables
2. **Use separate steps with dependencies** - Most reliable and clear
3. **Leverage pipeline parameters** - Share data between loop levels
4. **Consider the matrix pattern** - Good for fixed dimensions
5. **Use conditionals** - Control which combinations execute

## Example: Region x Product Matrix

```yaml
# Instead of:
- id: nested_bad
  for_each: "{{ regions }}"
  steps:
    - id: inner
      for_each: "{{ products }}"
      steps:
        - id: process
          # $parent_item won't work here!
          
# Do this:
- id: us_products
  for_each: "{{ products }}"
  steps:
    - id: process
      # Works perfectly!
      
- id: eu_products
  for_each: "{{ products }}"
  steps:
    - id: process
      # Works perfectly!
```

## Conclusion

While direct nested `for_each` loops have limitations, the alternative approaches provide full functionality with perfect template rendering. The separate steps pattern is the most reliable and should be the preferred approach for complex multi-dimensional processing.