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


### Progress Update - Commit 379f6de

Added debugging and file path template resolution:

1. **Added logging to track iteration counts** - Now logs when should_continue is called and why loops stop
2. **Fixed file path template resolution** - The FileSystemTool now renders path templates before using them
3. **Fixed action field processing** - The _process_loop_params now processes both action fields and parameters

Issues discovered:
- The write action is being interpreted by the model as a request to explain file writing, not actually executed
- The full pipeline still creates too many iterations (0, 1, 2, 3, 4...) even with max_attempts=1
- Simple test cases work correctly (only 1 iteration), suggesting issue is specific to the complex pipeline

### Next Steps
1. Investigate why write actions are being interpreted instead of executed
2. Debug the iteration count issue in the full pipeline


### Progress Update - Commit 26f77bb

Further improvements and testing:

1. **File path template resolution working** - Template variables in file paths (e.g., `{{guessing_loop.iteration}}`) are now resolved correctly
2. **Simple while loops work correctly** - Tests confirm that with max_attempts=1, only 1 iteration is executed
3. **Added comprehensive test cases** - Created multiple test files to isolate the issue

Test results:
- `test_while_simple.py`: ✅ Executes 1 iteration with max_iterations=1
- `test_while_debug.py`: ✅ Creates file with correct template resolution
- `test_while_full.py`: ✅ Executes 1 iteration with max_attempts=1
- `test_while_minimal.yaml`: ✅ Executes 1 iteration with max_attempts=1

The issue appears to be specific to the complex control_flow_while_loop.yaml pipeline. All simpler test cases work correctly.

### Next Steps
1. Investigate what makes the full pipeline different (possibly AUTO tags or complex dependencies)
2. Add more detailed logging to track where extra iterations are being created


### Progress Update - Commit 1bd6cca

Isolated the iteration count issue through comprehensive testing:

**Test Results Summary:**
- ✅ `test_iteration_count.yaml` - Simple echo loop works correctly (1 iteration)
- ✅ `test_while_simple_flow.yaml` - Loop with file writes works correctly (1 iteration)
- ✅ `test_while_trace.yaml` - Loop with checkpoints works correctly (1 iteration)
- ✅ `test_while_minimal.yaml` - Loop with generate_text works correctly (1 iteration)
- ❌ `control_flow_while_loop.yaml` - Times out during execution
- ❌ `test_while_no_auto.yaml` - Times out at evaluate_condition action

**Key Findings:**
1. **While loop iteration count is correct** - All simple tests execute exactly 1 iteration with max_attempts=1
2. **The timeout issue is not related to iteration count** - The pipeline gets stuck during execution
3. **The `evaluate_condition` action appears to be the problem** - Pipelines hang when reaching this action

The logs show the pipeline gets stuck after executing `guessing_loop_0_check_result` which uses the `evaluate_condition` action. This action doesn't appear to be a built-in action and may need to be implemented or replaced.

### Recommendation
Replace `evaluate_condition` with a proper condition check mechanism or implement the missing action handler.

