## Debugging control_flow_while_loop.yaml

Starting debug session for while loop control flow issues.

### Initial Findings

1. **Missing Jinja2 Filter**: The pipeline uses `regex_search` filter which is not available in the template engine:
   ```
   Error rendering template: No filter named 'regex_search'.
   Error type: TemplateAssertionError
   ```

2. **Variable Resolution Issues**: 
   - `read_guess.content` is not resolved (shows as `{{read_guess.content}}`)
   - `guessing_loop.iteration` is not available in loop context
   - Variables are prefixed with iteration number (e.g., `guessing_loop_0_read_guess`) but templates expect unprefixed names

3. **Iteration Count Issue**: With `max_attempts: 2`, the loop creates 4 iterations (0-3) instead of 2

### Test Environment
- Running with: `python scripts/run_pipeline.py examples/control_flow_while_loop.yaml -i '{"target_number": 42, "max_attempts": 2}'`
- Current branch: main (ahead by 4 commits)

### Next Steps
1. Add regex_search filter to template engine
2. Fix variable scoping in while loop iterations
3. Fix iteration count logic


### Progress Update - Commit b318645

Fixed two major issues:

1. **Added regex_search filter** - The template engine now supports the `regex_search` filter used throughout the pipeline
2. **Improved variable scoping** - Loop iteration results are now registered under both prefixed and unprefixed names

However, some template variables are still not resolving correctly:
- `{{ counter_loop.iteration }}` shows as literal text instead of the iteration number
- The loop metadata registration needs to happen earlier in the execution flow

### Next Steps
1. Fix template variable resolution timing for loop metadata
2. Investigate why `max_attempts: 1` creates 4 iterations in the full pipeline (though simple test works correctly)

