# Loop Variables in Orchestrator

## Overview

When using `for_each` loops in Orchestrator pipelines, several variables are automatically available within the loop context for use in templates.

## Available Loop Variables

### Core Variables

- `{{ item }}` or `{{ $item }}` - The current item being processed
- `{{ index }}` or `{{ $index }}` - The zero-based index of the current iteration
- `{{ is_first }}` or `{{ $is_first }}` - Boolean indicating if this is the first iteration
- `{{ is_last }}` or `{{ $is_last }}` - Boolean indicating if this is the last iteration

### Additional Variables

- `{{ length }}` or `{{ $length }}` - Total number of items in the loop
- `{{ position }}` or `{{ $position }}` - One-based position (index + 1)
- `{{ remaining }}` or `{{ $remaining }}` - Number of items remaining after current
- `{{ has_next }}` or `{{ $has_next }}` - Boolean indicating if there's a next item
- `{{ has_prev }}` or `{{ $has_prev }}` - Boolean indicating if there's a previous item

## Examples

### Simple Loop

```yaml
steps:
  - id: process_items
    for_each: "{{ items }}"
    steps:
      - id: save
        tool: filesystem
        action: write
        parameters:
          path: "output/{{ item }}_{{ index }}.txt"
          content: |
            Processing item {{ item }} at position {{ position }}
            This is item {{ index }} of {{ length }}
            First item: {{ is_first }}
            Last item: {{ is_last }}
```

### Loop with Dependencies

```yaml
steps:
  - id: process
    for_each: "{{ data_items }}"
    steps:
      - id: transform
        action: generate_text
        parameters:
          prompt: "Transform: {{ item }}"
          
      - id: save
        tool: filesystem
        action: write
        parameters:
          path: "results/{{ item }}.txt"
          content: |
            Original: {{ item }}
            Transformed: {{ transform }}
            Index: {{ index }}
        dependencies:
          - transform
```

### Named Loops (Advanced)

When loops are nested or need explicit naming:

```yaml
steps:
  - id: outer
    for_each: "{{ categories }}"
    as: category_loop  # Optional: give the loop a name
    steps:
      - id: process
        action: generate_text
        parameters:
          prompt: "Process category {{ $category_loop.item }}"
```

## Template Formats

Loop variables support both formats:
- Dollar prefix: `{{ $item }}`, `{{ $index }}`
- Without prefix: `{{ item }}`, `{{ index }}`

Both formats work identically within loop contexts.

## Nested Loops

For nested loops, you can access parent loop variables using the loop name:

```yaml
steps:
  - id: outer
    for_each: "{{ categories }}"
    as: outer_loop
    steps:
      - id: inner
        for_each: "{{ items }}"
        steps:
          - id: process
            parameters:
              # Access both loops
              category: "{{ $outer_loop.item }}"
              item: "{{ item }}"  # Current (inner) loop item
```

## Troubleshooting

### Common Issues

1. **Unrendered Templates**: If you see `{{ item }}` in your output instead of the actual value, ensure:
   - You're within a `for_each` loop context
   - The variable name is spelled correctly
   - You're using the correct template syntax

2. **Index Starting at 0**: Remember that `index` is zero-based. Use `position` for one-based numbering.

3. **Filesystem Paths**: When using loop variables in file paths, ensure the values don't contain invalid characters for filenames.

## Implementation Notes

As of the latest update, loop variables are properly injected into the execution context during both compile-time loop expansion and runtime ForEachTask expansion. The variables are available in:
- Task parameters
- Filesystem operations
- Template rendering
- Nested dependencies