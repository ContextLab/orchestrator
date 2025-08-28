# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-27 23:56:50

## Analysis Results

I don’t see any Python code in your message (it shows “# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-27 23:56:50

## Analysis Results

I don’t see any Python code in your message (it shows “# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-27 23:56:50

## Analysis Results

I don’t see any Python code in your message (it shows “# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-27 23:56:50

## Analysis Results

I don’t see any Python code in your message (it shows “# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-27 23:56:50

## Analysis Results

I don’t see any Python code in your message (it shows “# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-27 23:56:50

## Analysis Results

I don’t see any Python code in your message (it shows “# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-27 23:56:50

## Analysis Results

I don’t see any Python code in your message (it shows “# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-27 23:56:50

## Analysis Results

I don’t see any Python code in your message (it shows “# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-27 23:56:50

## Analysis Results

I don’t see any Python code in your message (it shows “# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-27 23:56:50

## Analysis Results

I don’t see any Python code in your message (it shows “# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-27 23:56:50

## Analysis Results

I don’t see any Python code in your message (it shows “# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-27 23:56:50

## Analysis Results

I don’t see any Python code in your message (it shows “{{content}}”). Please paste the code (or a link/repro) so I can review it and identify:
1) Performance bottlenecks
2) Code quality issues
3) Best practice violations

To get the most useful analysis, include:
- Python version, key dependencies, and typical input sizes
- How you run it (CLI, web, batch), and the slow paths you’ve noticed
- Example inputs or a minimal reproducer

If you want to pre-profile before sharing:
- CPU profiling:
  - python -m cProfile -o prof.out your_script.py; then view with SnakeViz: snakeviz prof.out
  - Or sampling profiler: py-spy record -o prof.svg -- python your_script.py
- Line-level timing: pip install line_profiler; decorate hot functions with @profile and run kernprof -l -v your_script.py
- Memory: pip install memory_profiler and add @profile; or use memray

When you share the code, I’ll return findings in this structure:
1) Performance bottlenecks
- Issue: what’s slow and where (function/line), why it’s slow
- Evidence: profile metrics or complexity
- Fix: specific change, expected impact, complexity change

2) Code quality issues
- Issue: description and location
- Why it matters: readability, maintainability, correctness risk
- Fix: concrete refactor

3) Best practice violations
- Violation: standard/guideline (e.g., PEP 8/484, resource handling, error handling)
- Fix: example-compliant snippet

Common hotspots I’ll check for:
- Algorithmic: accidental O(n^2) loops, repeated scans, N+1 DB/API calls, unnecessary sorting, missing indexing/caching
- Data structures: using lists where sets/dicts/heaps would be faster; using deque for queues
- I/O: unbuffered or per-line writes, string concatenation in loops (use ''.join), synchronous calls in a loop (batch/async)
- Numeric/data processing: opportunities for vectorization (NumPy/Pandas), avoiding Python loops, preallocations
- Repeated work: recompiling regex, re-parsing JSON/CSV, re-reading files, re-opening connections
- Concurrency: CPU-bound work on threads (use multiprocessing), I/O-bound work not using threads/async
- Memory: building huge intermediates vs streaming/generators; leaking references
- Pythonic details: global lookups in tight loops, attribute lookups in hot paths, function-call overhead, use of lru_cache

Code quality and best practices I’ll review:
- Naming, cohesion, function size, DRY, cyclomatic complexity
- Type hints (PEP 484), docstrings, tests
- Error handling (no bare except, preserve tracebacks), logging over print
- Resource safety (context managers), timeouts/retries on I/O
- Immutable default args, pathlib, f-strings, enumerate/zip, dataclasses
- Security: parameterized SQL, safe subprocess (no shell=True), input validation

Paste the code and I’ll dive in with concrete, line-specific recommendations.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”). Please paste the code (or a link/repro) so I can review it and identify:
1) Performance bottlenecks
2) Code quality issues
3) Best practice violations

To get the most useful analysis, include:
- Python version, key dependencies, and typical input sizes
- How you run it (CLI, web, batch), and the slow paths you’ve noticed
- Example inputs or a minimal reproducer

If you want to pre-profile before sharing:
- CPU profiling:
  - python -m cProfile -o prof.out your_script.py; then view with SnakeViz: snakeviz prof.out
  - Or sampling profiler: py-spy record -o prof.svg -- python your_script.py
- Line-level timing: pip install line_profiler; decorate hot functions with @profile and run kernprof -l -v your_script.py
- Memory: pip install memory_profiler and add @profile; or use memray

When you share the code, I’ll return findings in this structure:
1) Performance bottlenecks
- Issue: what’s slow and where (function/line), why it’s slow
- Evidence: profile metrics or complexity
- Fix: specific change, expected impact, complexity change

2) Code quality issues
- Issue: description and location
- Why it matters: readability, maintainability, correctness risk
- Fix: concrete refactor

3) Best practice violations
- Violation: standard/guideline (e.g., PEP 8/484, resource handling, error handling)
- Fix: example-compliant snippet

Common hotspots I’ll check for:
- Algorithmic: accidental O(n^2) loops, repeated scans, N+1 DB/API calls, unnecessary sorting, missing indexing/caching
- Data structures: using lists where sets/dicts/heaps would be faster; using deque for queues
- I/O: unbuffered or per-line writes, string concatenation in loops (use ''.join), synchronous calls in a loop (batch/async)
- Numeric/data processing: opportunities for vectorization (NumPy/Pandas), avoiding Python loops, preallocations
- Repeated work: recompiling regex, re-parsing JSON/CSV, re-reading files, re-opening connections
- Concurrency: CPU-bound work on threads (use multiprocessing), I/O-bound work not using threads/async
- Memory: building huge intermediates vs streaming/generators; leaking references
- Pythonic details: global lookups in tight loops, attribute lookups in hot paths, function-call overhead, use of lru_cache

Code quality and best practices I’ll review:
- Naming, cohesion, function size, DRY, cyclomatic complexity
- Type hints (PEP 484), docstrings, tests
- Error handling (no bare except, preserve tracebacks), logging over print
- Resource safety (context managers), timeouts/retries on I/O
- Immutable default args, pathlib, f-strings, enumerate/zip, dataclasses
- Security: parameterized SQL, safe subprocess (no shell=True), input validation

Paste the code and I’ll dive in with concrete, line-specific recommendations.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”). Please paste the code (or a link/repro) so I can review it and identify:
1) Performance bottlenecks
2) Code quality issues
3) Best practice violations

To get the most useful analysis, include:
- Python version, key dependencies, and typical input sizes
- How you run it (CLI, web, batch), and the slow paths you’ve noticed
- Example inputs or a minimal reproducer

If you want to pre-profile before sharing:
- CPU profiling:
  - python -m cProfile -o prof.out your_script.py; then view with SnakeViz: snakeviz prof.out
  - Or sampling profiler: py-spy record -o prof.svg -- python your_script.py
- Line-level timing: pip install line_profiler; decorate hot functions with @profile and run kernprof -l -v your_script.py
- Memory: pip install memory_profiler and add @profile; or use memray

When you share the code, I’ll return findings in this structure:
1) Performance bottlenecks
- Issue: what’s slow and where (function/line), why it’s slow
- Evidence: profile metrics or complexity
- Fix: specific change, expected impact, complexity change

2) Code quality issues
- Issue: description and location
- Why it matters: readability, maintainability, correctness risk
- Fix: concrete refactor

3) Best practice violations
- Violation: standard/guideline (e.g., PEP 8/484, resource handling, error handling)
- Fix: example-compliant snippet

Common hotspots I’ll check for:
- Algorithmic: accidental O(n^2) loops, repeated scans, N+1 DB/API calls, unnecessary sorting, missing indexing/caching
- Data structures: using lists where sets/dicts/heaps would be faster; using deque for queues
- I/O: unbuffered or per-line writes, string concatenation in loops (use ''.join), synchronous calls in a loop (batch/async)
- Numeric/data processing: opportunities for vectorization (NumPy/Pandas), avoiding Python loops, preallocations
- Repeated work: recompiling regex, re-parsing JSON/CSV, re-reading files, re-opening connections
- Concurrency: CPU-bound work on threads (use multiprocessing), I/O-bound work not using threads/async
- Memory: building huge intermediates vs streaming/generators; leaking references
- Pythonic details: global lookups in tight loops, attribute lookups in hot paths, function-call overhead, use of lru_cache

Code quality and best practices I’ll review:
- Naming, cohesion, function size, DRY, cyclomatic complexity
- Type hints (PEP 484), docstrings, tests
- Error handling (no bare except, preserve tracebacks), logging over print
- Resource safety (context managers), timeouts/retries on I/O
- Immutable default args, pathlib, f-strings, enumerate/zip, dataclasses
- Security: parameterized SQL, safe subprocess (no shell=True), input validation

Paste the code and I’ll dive in with concrete, line-specific recommendations.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”). Please paste the code (or a link/repro) so I can review it and identify:
1) Performance bottlenecks
2) Code quality issues
3) Best practice violations

To get the most useful analysis, include:
- Python version, key dependencies, and typical input sizes
- How you run it (CLI, web, batch), and the slow paths you’ve noticed
- Example inputs or a minimal reproducer

If you want to pre-profile before sharing:
- CPU profiling:
  - python -m cProfile -o prof.out your_script.py; then view with SnakeViz: snakeviz prof.out
  - Or sampling profiler: py-spy record -o prof.svg -- python your_script.py
- Line-level timing: pip install line_profiler; decorate hot functions with @profile and run kernprof -l -v your_script.py
- Memory: pip install memory_profiler and add @profile; or use memray

When you share the code, I’ll return findings in this structure:
1) Performance bottlenecks
- Issue: what’s slow and where (function/line), why it’s slow
- Evidence: profile metrics or complexity
- Fix: specific change, expected impact, complexity change

2) Code quality issues
- Issue: description and location
- Why it matters: readability, maintainability, correctness risk
- Fix: concrete refactor

3) Best practice violations
- Violation: standard/guideline (e.g., PEP 8/484, resource handling, error handling)
- Fix: example-compliant snippet

Common hotspots I’ll check for:
- Algorithmic: accidental O(n^2) loops, repeated scans, N+1 DB/API calls, unnecessary sorting, missing indexing/caching
- Data structures: using lists where sets/dicts/heaps would be faster; using deque for queues
- I/O: unbuffered or per-line writes, string concatenation in loops (use ''.join), synchronous calls in a loop (batch/async)
- Numeric/data processing: opportunities for vectorization (NumPy/Pandas), avoiding Python loops, preallocations
- Repeated work: recompiling regex, re-parsing JSON/CSV, re-reading files, re-opening connections
- Concurrency: CPU-bound work on threads (use multiprocessing), I/O-bound work not using threads/async
- Memory: building huge intermediates vs streaming/generators; leaking references
- Pythonic details: global lookups in tight loops, attribute lookups in hot paths, function-call overhead, use of lru_cache

Code quality and best practices I’ll review:
- Naming, cohesion, function size, DRY, cyclomatic complexity
- Type hints (PEP 484), docstrings, tests
- Error handling (no bare except, preserve tracebacks), logging over print
- Resource safety (context managers), timeouts/retries on I/O
- Immutable default args, pathlib, f-strings, enumerate/zip, dataclasses
- Security: parameterized SQL, safe subprocess (no shell=True), input validation

Paste the code and I’ll dive in with concrete, line-specific recommendations.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”). Please paste the code (or a link/repro) so I can review it and identify:
1) Performance bottlenecks
2) Code quality issues
3) Best practice violations

To get the most useful analysis, include:
- Python version, key dependencies, and typical input sizes
- How you run it (CLI, web, batch), and the slow paths you’ve noticed
- Example inputs or a minimal reproducer

If you want to pre-profile before sharing:
- CPU profiling:
  - python -m cProfile -o prof.out your_script.py; then view with SnakeViz: snakeviz prof.out
  - Or sampling profiler: py-spy record -o prof.svg -- python your_script.py
- Line-level timing: pip install line_profiler; decorate hot functions with @profile and run kernprof -l -v your_script.py
- Memory: pip install memory_profiler and add @profile; or use memray

When you share the code, I’ll return findings in this structure:
1) Performance bottlenecks
- Issue: what’s slow and where (function/line), why it’s slow
- Evidence: profile metrics or complexity
- Fix: specific change, expected impact, complexity change

2) Code quality issues
- Issue: description and location
- Why it matters: readability, maintainability, correctness risk
- Fix: concrete refactor

3) Best practice violations
- Violation: standard/guideline (e.g., PEP 8/484, resource handling, error handling)
- Fix: example-compliant snippet

Common hotspots I’ll check for:
- Algorithmic: accidental O(n^2) loops, repeated scans, N+1 DB/API calls, unnecessary sorting, missing indexing/caching
- Data structures: using lists where sets/dicts/heaps would be faster; using deque for queues
- I/O: unbuffered or per-line writes, string concatenation in loops (use ''.join), synchronous calls in a loop (batch/async)
- Numeric/data processing: opportunities for vectorization (NumPy/Pandas), avoiding Python loops, preallocations
- Repeated work: recompiling regex, re-parsing JSON/CSV, re-reading files, re-opening connections
- Concurrency: CPU-bound work on threads (use multiprocessing), I/O-bound work not using threads/async
- Memory: building huge intermediates vs streaming/generators; leaking references
- Pythonic details: global lookups in tight loops, attribute lookups in hot paths, function-call overhead, use of lru_cache

Code quality and best practices I’ll review:
- Naming, cohesion, function size, DRY, cyclomatic complexity
- Type hints (PEP 484), docstrings, tests
- Error handling (no bare except, preserve tracebacks), logging over print
- Resource safety (context managers), timeouts/retries on I/O
- Immutable default args, pathlib, f-strings, enumerate/zip, dataclasses
- Security: parameterized SQL, safe subprocess (no shell=True), input validation

Paste the code and I’ll dive in with concrete, line-specific recommendations.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”). Please paste the code (or a link/repro) so I can review it and identify:
1) Performance bottlenecks
2) Code quality issues
3) Best practice violations

To get the most useful analysis, include:
- Python version, key dependencies, and typical input sizes
- How you run it (CLI, web, batch), and the slow paths you’ve noticed
- Example inputs or a minimal reproducer

If you want to pre-profile before sharing:
- CPU profiling:
  - python -m cProfile -o prof.out your_script.py; then view with SnakeViz: snakeviz prof.out
  - Or sampling profiler: py-spy record -o prof.svg -- python your_script.py
- Line-level timing: pip install line_profiler; decorate hot functions with @profile and run kernprof -l -v your_script.py
- Memory: pip install memory_profiler and add @profile; or use memray

When you share the code, I’ll return findings in this structure:
1) Performance bottlenecks
- Issue: what’s slow and where (function/line), why it’s slow
- Evidence: profile metrics or complexity
- Fix: specific change, expected impact, complexity change

2) Code quality issues
- Issue: description and location
- Why it matters: readability, maintainability, correctness risk
- Fix: concrete refactor

3) Best practice violations
- Violation: standard/guideline (e.g., PEP 8/484, resource handling, error handling)
- Fix: example-compliant snippet

Common hotspots I’ll check for:
- Algorithmic: accidental O(n^2) loops, repeated scans, N+1 DB/API calls, unnecessary sorting, missing indexing/caching
- Data structures: using lists where sets/dicts/heaps would be faster; using deque for queues
- I/O: unbuffered or per-line writes, string concatenation in loops (use ''.join), synchronous calls in a loop (batch/async)
- Numeric/data processing: opportunities for vectorization (NumPy/Pandas), avoiding Python loops, preallocations
- Repeated work: recompiling regex, re-parsing JSON/CSV, re-reading files, re-opening connections
- Concurrency: CPU-bound work on threads (use multiprocessing), I/O-bound work not using threads/async
- Memory: building huge intermediates vs streaming/generators; leaking references
- Pythonic details: global lookups in tight loops, attribute lookups in hot paths, function-call overhead, use of lru_cache

Code quality and best practices I’ll review:
- Naming, cohesion, function size, DRY, cyclomatic complexity
- Type hints (PEP 484), docstrings, tests
- Error handling (no bare except, preserve tracebacks), logging over print
- Resource safety (context managers), timeouts/retries on I/O
- Immutable default args, pathlib, f-strings, enumerate/zip, dataclasses
- Security: parameterized SQL, safe subprocess (no shell=True), input validation

Paste the code and I’ll dive in with concrete, line-specific recommendations.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”). Please paste the code (or a link/repro) so I can review it and identify:
1) Performance bottlenecks
2) Code quality issues
3) Best practice violations

To get the most useful analysis, include:
- Python version, key dependencies, and typical input sizes
- How you run it (CLI, web, batch), and the slow paths you’ve noticed
- Example inputs or a minimal reproducer

If you want to pre-profile before sharing:
- CPU profiling:
  - python -m cProfile -o prof.out your_script.py; then view with SnakeViz: snakeviz prof.out
  - Or sampling profiler: py-spy record -o prof.svg -- python your_script.py
- Line-level timing: pip install line_profiler; decorate hot functions with @profile and run kernprof -l -v your_script.py
- Memory: pip install memory_profiler and add @profile; or use memray

When you share the code, I’ll return findings in this structure:
1) Performance bottlenecks
- Issue: what’s slow and where (function/line), why it’s slow
- Evidence: profile metrics or complexity
- Fix: specific change, expected impact, complexity change

2) Code quality issues
- Issue: description and location
- Why it matters: readability, maintainability, correctness risk
- Fix: concrete refactor

3) Best practice violations
- Violation: standard/guideline (e.g., PEP 8/484, resource handling, error handling)
- Fix: example-compliant snippet

Common hotspots I’ll check for:
- Algorithmic: accidental O(n^2) loops, repeated scans, N+1 DB/API calls, unnecessary sorting, missing indexing/caching
- Data structures: using lists where sets/dicts/heaps would be faster; using deque for queues
- I/O: unbuffered or per-line writes, string concatenation in loops (use ''.join), synchronous calls in a loop (batch/async)
- Numeric/data processing: opportunities for vectorization (NumPy/Pandas), avoiding Python loops, preallocations
- Repeated work: recompiling regex, re-parsing JSON/CSV, re-reading files, re-opening connections
- Concurrency: CPU-bound work on threads (use multiprocessing), I/O-bound work not using threads/async
- Memory: building huge intermediates vs streaming/generators; leaking references
- Pythonic details: global lookups in tight loops, attribute lookups in hot paths, function-call overhead, use of lru_cache

Code quality and best practices I’ll review:
- Naming, cohesion, function size, DRY, cyclomatic complexity
- Type hints (PEP 484), docstrings, tests
- Error handling (no bare except, preserve tracebacks), logging over print
- Resource safety (context managers), timeouts/retries on I/O
- Immutable default args, pathlib, f-strings, enumerate/zip, dataclasses
- Security: parameterized SQL, safe subprocess (no shell=True), input validation

Paste the code and I’ll dive in with concrete, line-specific recommendations.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”). Please paste the code (or a link/repro) so I can review it and identify:
1) Performance bottlenecks
2) Code quality issues
3) Best practice violations

To get the most useful analysis, include:
- Python version, key dependencies, and typical input sizes
- How you run it (CLI, web, batch), and the slow paths you’ve noticed
- Example inputs or a minimal reproducer

If you want to pre-profile before sharing:
- CPU profiling:
  - python -m cProfile -o prof.out your_script.py; then view with SnakeViz: snakeviz prof.out
  - Or sampling profiler: py-spy record -o prof.svg -- python your_script.py
- Line-level timing: pip install line_profiler; decorate hot functions with @profile and run kernprof -l -v your_script.py
- Memory: pip install memory_profiler and add @profile; or use memray

When you share the code, I’ll return findings in this structure:
1) Performance bottlenecks
- Issue: what’s slow and where (function/line), why it’s slow
- Evidence: profile metrics or complexity
- Fix: specific change, expected impact, complexity change

2) Code quality issues
- Issue: description and location
- Why it matters: readability, maintainability, correctness risk
- Fix: concrete refactor

3) Best practice violations
- Violation: standard/guideline (e.g., PEP 8/484, resource handling, error handling)
- Fix: example-compliant snippet

Common hotspots I’ll check for:
- Algorithmic: accidental O(n^2) loops, repeated scans, N+1 DB/API calls, unnecessary sorting, missing indexing/caching
- Data structures: using lists where sets/dicts/heaps would be faster; using deque for queues
- I/O: unbuffered or per-line writes, string concatenation in loops (use ''.join), synchronous calls in a loop (batch/async)
- Numeric/data processing: opportunities for vectorization (NumPy/Pandas), avoiding Python loops, preallocations
- Repeated work: recompiling regex, re-parsing JSON/CSV, re-reading files, re-opening connections
- Concurrency: CPU-bound work on threads (use multiprocessing), I/O-bound work not using threads/async
- Memory: building huge intermediates vs streaming/generators; leaking references
- Pythonic details: global lookups in tight loops, attribute lookups in hot paths, function-call overhead, use of lru_cache

Code quality and best practices I’ll review:
- Naming, cohesion, function size, DRY, cyclomatic complexity
- Type hints (PEP 484), docstrings, tests
- Error handling (no bare except, preserve tracebacks), logging over print
- Resource safety (context managers), timeouts/retries on I/O
- Immutable default args, pathlib, f-strings, enumerate/zip, dataclasses
- Security: parameterized SQL, safe subprocess (no shell=True), input validation

Paste the code and I’ll dive in with concrete, line-specific recommendations.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”). Please paste the code (or a link/repro) so I can review it and identify:
1) Performance bottlenecks
2) Code quality issues
3) Best practice violations

To get the most useful analysis, include:
- Python version, key dependencies, and typical input sizes
- How you run it (CLI, web, batch), and the slow paths you’ve noticed
- Example inputs or a minimal reproducer

If you want to pre-profile before sharing:
- CPU profiling:
  - python -m cProfile -o prof.out your_script.py; then view with SnakeViz: snakeviz prof.out
  - Or sampling profiler: py-spy record -o prof.svg -- python your_script.py
- Line-level timing: pip install line_profiler; decorate hot functions with @profile and run kernprof -l -v your_script.py
- Memory: pip install memory_profiler and add @profile; or use memray

When you share the code, I’ll return findings in this structure:
1) Performance bottlenecks
- Issue: what’s slow and where (function/line), why it’s slow
- Evidence: profile metrics or complexity
- Fix: specific change, expected impact, complexity change

2) Code quality issues
- Issue: description and location
- Why it matters: readability, maintainability, correctness risk
- Fix: concrete refactor

3) Best practice violations
- Violation: standard/guideline (e.g., PEP 8/484, resource handling, error handling)
- Fix: example-compliant snippet

Common hotspots I’ll check for:
- Algorithmic: accidental O(n^2) loops, repeated scans, N+1 DB/API calls, unnecessary sorting, missing indexing/caching
- Data structures: using lists where sets/dicts/heaps would be faster; using deque for queues
- I/O: unbuffered or per-line writes, string concatenation in loops (use ''.join), synchronous calls in a loop (batch/async)
- Numeric/data processing: opportunities for vectorization (NumPy/Pandas), avoiding Python loops, preallocations
- Repeated work: recompiling regex, re-parsing JSON/CSV, re-reading files, re-opening connections
- Concurrency: CPU-bound work on threads (use multiprocessing), I/O-bound work not using threads/async
- Memory: building huge intermediates vs streaming/generators; leaking references
- Pythonic details: global lookups in tight loops, attribute lookups in hot paths, function-call overhead, use of lru_cache

Code quality and best practices I’ll review:
- Naming, cohesion, function size, DRY, cyclomatic complexity
- Type hints (PEP 484), docstrings, tests
- Error handling (no bare except, preserve tracebacks), logging over print
- Resource safety (context managers), timeouts/retries on I/O
- Immutable default args, pathlib, f-strings, enumerate/zip, dataclasses
- Security: parameterized SQL, safe subprocess (no shell=True), input validation

Paste the code and I’ll dive in with concrete, line-specific recommendations.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”). Please paste the code (or a link/repro) so I can review it and identify:
1) Performance bottlenecks
2) Code quality issues
3) Best practice violations

To get the most useful analysis, include:
- Python version, key dependencies, and typical input sizes
- How you run it (CLI, web, batch), and the slow paths you’ve noticed
- Example inputs or a minimal reproducer

If you want to pre-profile before sharing:
- CPU profiling:
  - python -m cProfile -o prof.out your_script.py; then view with SnakeViz: snakeviz prof.out
  - Or sampling profiler: py-spy record -o prof.svg -- python your_script.py
- Line-level timing: pip install line_profiler; decorate hot functions with @profile and run kernprof -l -v your_script.py
- Memory: pip install memory_profiler and add @profile; or use memray

When you share the code, I’ll return findings in this structure:
1) Performance bottlenecks
- Issue: what’s slow and where (function/line), why it’s slow
- Evidence: profile metrics or complexity
- Fix: specific change, expected impact, complexity change

2) Code quality issues
- Issue: description and location
- Why it matters: readability, maintainability, correctness risk
- Fix: concrete refactor

3) Best practice violations
- Violation: standard/guideline (e.g., PEP 8/484, resource handling, error handling)
- Fix: example-compliant snippet

Common hotspots I’ll check for:
- Algorithmic: accidental O(n^2) loops, repeated scans, N+1 DB/API calls, unnecessary sorting, missing indexing/caching
- Data structures: using lists where sets/dicts/heaps would be faster; using deque for queues
- I/O: unbuffered or per-line writes, string concatenation in loops (use ''.join), synchronous calls in a loop (batch/async)
- Numeric/data processing: opportunities for vectorization (NumPy/Pandas), avoiding Python loops, preallocations
- Repeated work: recompiling regex, re-parsing JSON/CSV, re-reading files, re-opening connections
- Concurrency: CPU-bound work on threads (use multiprocessing), I/O-bound work not using threads/async
- Memory: building huge intermediates vs streaming/generators; leaking references
- Pythonic details: global lookups in tight loops, attribute lookups in hot paths, function-call overhead, use of lru_cache

Code quality and best practices I’ll review:
- Naming, cohesion, function size, DRY, cyclomatic complexity
- Type hints (PEP 484), docstrings, tests
- Error handling (no bare except, preserve tracebacks), logging over print
- Resource safety (context managers), timeouts/retries on I/O
- Immutable default args, pathlib, f-strings, enumerate/zip, dataclasses
- Security: parameterized SQL, safe subprocess (no shell=True), input validation

Paste the code and I’ll dive in with concrete, line-specific recommendations.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”). Please paste the code (or a link/repro) so I can review it and identify:
1) Performance bottlenecks
2) Code quality issues
3) Best practice violations

To get the most useful analysis, include:
- Python version, key dependencies, and typical input sizes
- How you run it (CLI, web, batch), and the slow paths you’ve noticed
- Example inputs or a minimal reproducer

If you want to pre-profile before sharing:
- CPU profiling:
  - python -m cProfile -o prof.out your_script.py; then view with SnakeViz: snakeviz prof.out
  - Or sampling profiler: py-spy record -o prof.svg -- python your_script.py
- Line-level timing: pip install line_profiler; decorate hot functions with @profile and run kernprof -l -v your_script.py
- Memory: pip install memory_profiler and add @profile; or use memray

When you share the code, I’ll return findings in this structure:
1) Performance bottlenecks
- Issue: what’s slow and where (function/line), why it’s slow
- Evidence: profile metrics or complexity
- Fix: specific change, expected impact, complexity change

2) Code quality issues
- Issue: description and location
- Why it matters: readability, maintainability, correctness risk
- Fix: concrete refactor

3) Best practice violations
- Violation: standard/guideline (e.g., PEP 8/484, resource handling, error handling)
- Fix: example-compliant snippet

Common hotspots I’ll check for:
- Algorithmic: accidental O(n^2) loops, repeated scans, N+1 DB/API calls, unnecessary sorting, missing indexing/caching
- Data structures: using lists where sets/dicts/heaps would be faster; using deque for queues
- I/O: unbuffered or per-line writes, string concatenation in loops (use ''.join), synchronous calls in a loop (batch/async)
- Numeric/data processing: opportunities for vectorization (NumPy/Pandas), avoiding Python loops, preallocations
- Repeated work: recompiling regex, re-parsing JSON/CSV, re-reading files, re-opening connections
- Concurrency: CPU-bound work on threads (use multiprocessing), I/O-bound work not using threads/async
- Memory: building huge intermediates vs streaming/generators; leaking references
- Pythonic details: global lookups in tight loops, attribute lookups in hot paths, function-call overhead, use of lru_cache

Code quality and best practices I’ll review:
- Naming, cohesion, function size, DRY, cyclomatic complexity
- Type hints (PEP 484), docstrings, tests
- Error handling (no bare except, preserve tracebacks), logging over print
- Resource safety (context managers), timeouts/retries on I/O
- Immutable default args, pathlib, f-strings, enumerate/zip, dataclasses
- Security: parameterized SQL, safe subprocess (no shell=True), input validation

Paste the code and I’ll dive in with concrete, line-specific recommendations.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”). Please paste the code (or a link/repro) so I can review it and identify:
1) Performance bottlenecks
2) Code quality issues
3) Best practice violations

To get the most useful analysis, include:
- Python version, key dependencies, and typical input sizes
- How you run it (CLI, web, batch), and the slow paths you’ve noticed
- Example inputs or a minimal reproducer

If you want to pre-profile before sharing:
- CPU profiling:
  - python -m cProfile -o prof.out your_script.py; then view with SnakeViz: snakeviz prof.out
  - Or sampling profiler: py-spy record -o prof.svg -- python your_script.py
- Line-level timing: pip install line_profiler; decorate hot functions with @profile and run kernprof -l -v your_script.py
- Memory: pip install memory_profiler and add @profile; or use memray

When you share the code, I’ll return findings in this structure:
1) Performance bottlenecks
- Issue: what’s slow and where (function/line), why it’s slow
- Evidence: profile metrics or complexity
- Fix: specific change, expected impact, complexity change

2) Code quality issues
- Issue: description and location
- Why it matters: readability, maintainability, correctness risk
- Fix: concrete refactor

3) Best practice violations
- Violation: standard/guideline (e.g., PEP 8/484, resource handling, error handling)
- Fix: example-compliant snippet

Common hotspots I’ll check for:
- Algorithmic: accidental O(n^2) loops, repeated scans, N+1 DB/API calls, unnecessary sorting, missing indexing/caching
- Data structures: using lists where sets/dicts/heaps would be faster; using deque for queues
- I/O: unbuffered or per-line writes, string concatenation in loops (use ''.join), synchronous calls in a loop (batch/async)
- Numeric/data processing: opportunities for vectorization (NumPy/Pandas), avoiding Python loops, preallocations
- Repeated work: recompiling regex, re-parsing JSON/CSV, re-reading files, re-opening connections
- Concurrency: CPU-bound work on threads (use multiprocessing), I/O-bound work not using threads/async
- Memory: building huge intermediates vs streaming/generators; leaking references
- Pythonic details: global lookups in tight loops, attribute lookups in hot paths, function-call overhead, use of lru_cache

Code quality and best practices I’ll review:
- Naming, cohesion, function size, DRY, cyclomatic complexity
- Type hints (PEP 484), docstrings, tests
- Error handling (no bare except, preserve tracebacks), logging over print
- Resource safety (context managers), timeouts/retries on I/O
- Immutable default args, pathlib, f-strings, enumerate/zip, dataclasses
- Security: parameterized SQL, safe subprocess (no shell=True), input validation

Paste the code and I’ll dive in with concrete, line-specific recommendations.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions