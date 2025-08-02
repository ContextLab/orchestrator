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

