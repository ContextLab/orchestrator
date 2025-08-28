# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:24:41

## Analysis Results

I don’t see any Python code in your message—the code block only contains # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:24:41

## Analysis Results

I don’t see any Python code in your message—the code block only contains # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:24:41

## Analysis Results

I don’t see any Python code in your message—the code block only contains # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:24:41

## Analysis Results

I don’t see any Python code in your message—the code block only contains # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:24:41

## Analysis Results

I don’t see any Python code in your message—the code block only contains # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:24:41

## Analysis Results

I don’t see any Python code in your message—the code block only contains # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:24:41

## Analysis Results

I don’t see any Python code in your message—the code block only contains # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:24:41

## Analysis Results

I don’t see any Python code in your message—the code block only contains # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:24:41

## Analysis Results

I don’t see any Python code in your message—the code block only contains # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:24:41

## Analysis Results

I don’t see any Python code in your message—the code block only contains # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:24:41

## Analysis Results

I don’t see any Python code in your message—the code block only contains # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:24:41

## Analysis Results

I don’t see any Python code in your message—the code block only contains {{content}}. Please paste the actual code so I can analyze it.

To provide the most useful review, include:
- Python version and key libraries
- Typical input sizes/data shapes
- Performance symptoms (slow steps, CPU vs I/O bound)
- Environment (local/serverless/container), OS
- Any constraints (memory/time limits), and target improvements

What I will deliver once I have the code:
1) Performance bottlenecks
- Hot loops, N^2 patterns, unnecessary work
- Repeated I/O or network calls in loops
- Inefficient data structures/algorithms
- Non-vectorized NumPy/Pandas usage, inefficient groupby/apply
- Redundant allocations, copies, conversions
- Regex compilation in loops, string concatenation in loops
- Serialization or logging overhead
- Sync network calls where batching/async helps
- CPU-bound work on threads (GIL issues), use of processes/numba/cython where apt

2) Code quality issues
- Readability, naming, structure, duplication
- Function length/complexity, missing docstrings/comments
- Misuse of exceptions, lack of context managers for resources
- Default-mutable args, shadowing builtins, global state
- Inconsistent or missing type hints, weak validation
- Logging vs print, missing tests, lack of separation of concerns

3) Best practice violations
- PEP 8/257 style, import hygiene, f-strings
- Data classes/namedtuples for simple containers
- pathlib for paths, timezone-aware datetimes
- Caching (functools.lru_cache), memoization where applicable
- Safe APIs (avoid eval/exec, pickle on untrusted data, yaml.safe_load)
- Deterministic randomness seeding, reproducibility
- __main__ guard, packaging/layout, dependency pinning

Optional quick checks you can run before sharing:
- Profiling CPU: python -m cProfile -o prof.out your_script.py args... and visualize with snakeviz prof.out
- Line profiling: line_profiler or scalene
- Memory: pip install memray or tracemalloc
- Pandas: df.info(), df.memory_usage(deep=True), sample timings with %timeit

Paste the code and any context, and I’ll produce a targeted report with concrete fixes and before/after examples.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions. Please paste the actual code so I can analyze it.

To provide the most useful review, include:
- Python version and key libraries
- Typical input sizes/data shapes
- Performance symptoms (slow steps, CPU vs I/O bound)
- Environment (local/serverless/container), OS
- Any constraints (memory/time limits), and target improvements

What I will deliver once I have the code:
1) Performance bottlenecks
- Hot loops, N^2 patterns, unnecessary work
- Repeated I/O or network calls in loops
- Inefficient data structures/algorithms
- Non-vectorized NumPy/Pandas usage, inefficient groupby/apply
- Redundant allocations, copies, conversions
- Regex compilation in loops, string concatenation in loops
- Serialization or logging overhead
- Sync network calls where batching/async helps
- CPU-bound work on threads (GIL issues), use of processes/numba/cython where apt

2) Code quality issues
- Readability, naming, structure, duplication
- Function length/complexity, missing docstrings/comments
- Misuse of exceptions, lack of context managers for resources
- Default-mutable args, shadowing builtins, global state
- Inconsistent or missing type hints, weak validation
- Logging vs print, missing tests, lack of separation of concerns

3) Best practice violations
- PEP 8/257 style, import hygiene, f-strings
- Data classes/namedtuples for simple containers
- pathlib for paths, timezone-aware datetimes
- Caching (functools.lru_cache), memoization where applicable
- Safe APIs (avoid eval/exec, pickle on untrusted data, yaml.safe_load)
- Deterministic randomness seeding, reproducibility
- __main__ guard, packaging/layout, dependency pinning

Optional quick checks you can run before sharing:
- Profiling CPU: python -m cProfile -o prof.out your_script.py args... and visualize with snakeviz prof.out
- Line profiling: line_profiler or scalene
- Memory: pip install memray or tracemalloc
- Pandas: df.info(), df.memory_usage(deep=True), sample timings with %timeit

Paste the code and any context, and I’ll produce a targeted report with concrete fixes and before/after examples.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions. Please paste the actual code so I can analyze it.

To provide the most useful review, include:
- Python version and key libraries
- Typical input sizes/data shapes
- Performance symptoms (slow steps, CPU vs I/O bound)
- Environment (local/serverless/container), OS
- Any constraints (memory/time limits), and target improvements

What I will deliver once I have the code:
1) Performance bottlenecks
- Hot loops, N^2 patterns, unnecessary work
- Repeated I/O or network calls in loops
- Inefficient data structures/algorithms
- Non-vectorized NumPy/Pandas usage, inefficient groupby/apply
- Redundant allocations, copies, conversions
- Regex compilation in loops, string concatenation in loops
- Serialization or logging overhead
- Sync network calls where batching/async helps
- CPU-bound work on threads (GIL issues), use of processes/numba/cython where apt

2) Code quality issues
- Readability, naming, structure, duplication
- Function length/complexity, missing docstrings/comments
- Misuse of exceptions, lack of context managers for resources
- Default-mutable args, shadowing builtins, global state
- Inconsistent or missing type hints, weak validation
- Logging vs print, missing tests, lack of separation of concerns

3) Best practice violations
- PEP 8/257 style, import hygiene, f-strings
- Data classes/namedtuples for simple containers
- pathlib for paths, timezone-aware datetimes
- Caching (functools.lru_cache), memoization where applicable
- Safe APIs (avoid eval/exec, pickle on untrusted data, yaml.safe_load)
- Deterministic randomness seeding, reproducibility
- __main__ guard, packaging/layout, dependency pinning

Optional quick checks you can run before sharing:
- Profiling CPU: python -m cProfile -o prof.out your_script.py args... and visualize with snakeviz prof.out
- Line profiling: line_profiler or scalene
- Memory: pip install memray or tracemalloc
- Pandas: df.info(), df.memory_usage(deep=True), sample timings with %timeit

Paste the code and any context, and I’ll produce a targeted report with concrete fixes and before/after examples.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions. Please paste the actual code so I can analyze it.

To provide the most useful review, include:
- Python version and key libraries
- Typical input sizes/data shapes
- Performance symptoms (slow steps, CPU vs I/O bound)
- Environment (local/serverless/container), OS
- Any constraints (memory/time limits), and target improvements

What I will deliver once I have the code:
1) Performance bottlenecks
- Hot loops, N^2 patterns, unnecessary work
- Repeated I/O or network calls in loops
- Inefficient data structures/algorithms
- Non-vectorized NumPy/Pandas usage, inefficient groupby/apply
- Redundant allocations, copies, conversions
- Regex compilation in loops, string concatenation in loops
- Serialization or logging overhead
- Sync network calls where batching/async helps
- CPU-bound work on threads (GIL issues), use of processes/numba/cython where apt

2) Code quality issues
- Readability, naming, structure, duplication
- Function length/complexity, missing docstrings/comments
- Misuse of exceptions, lack of context managers for resources
- Default-mutable args, shadowing builtins, global state
- Inconsistent or missing type hints, weak validation
- Logging vs print, missing tests, lack of separation of concerns

3) Best practice violations
- PEP 8/257 style, import hygiene, f-strings
- Data classes/namedtuples for simple containers
- pathlib for paths, timezone-aware datetimes
- Caching (functools.lru_cache), memoization where applicable
- Safe APIs (avoid eval/exec, pickle on untrusted data, yaml.safe_load)
- Deterministic randomness seeding, reproducibility
- __main__ guard, packaging/layout, dependency pinning

Optional quick checks you can run before sharing:
- Profiling CPU: python -m cProfile -o prof.out your_script.py args... and visualize with snakeviz prof.out
- Line profiling: line_profiler or scalene
- Memory: pip install memray or tracemalloc
- Pandas: df.info(), df.memory_usage(deep=True), sample timings with %timeit

Paste the code and any context, and I’ll produce a targeted report with concrete fixes and before/after examples.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions. Please paste the actual code so I can analyze it.

To provide the most useful review, include:
- Python version and key libraries
- Typical input sizes/data shapes
- Performance symptoms (slow steps, CPU vs I/O bound)
- Environment (local/serverless/container), OS
- Any constraints (memory/time limits), and target improvements

What I will deliver once I have the code:
1) Performance bottlenecks
- Hot loops, N^2 patterns, unnecessary work
- Repeated I/O or network calls in loops
- Inefficient data structures/algorithms
- Non-vectorized NumPy/Pandas usage, inefficient groupby/apply
- Redundant allocations, copies, conversions
- Regex compilation in loops, string concatenation in loops
- Serialization or logging overhead
- Sync network calls where batching/async helps
- CPU-bound work on threads (GIL issues), use of processes/numba/cython where apt

2) Code quality issues
- Readability, naming, structure, duplication
- Function length/complexity, missing docstrings/comments
- Misuse of exceptions, lack of context managers for resources
- Default-mutable args, shadowing builtins, global state
- Inconsistent or missing type hints, weak validation
- Logging vs print, missing tests, lack of separation of concerns

3) Best practice violations
- PEP 8/257 style, import hygiene, f-strings
- Data classes/namedtuples for simple containers
- pathlib for paths, timezone-aware datetimes
- Caching (functools.lru_cache), memoization where applicable
- Safe APIs (avoid eval/exec, pickle on untrusted data, yaml.safe_load)
- Deterministic randomness seeding, reproducibility
- __main__ guard, packaging/layout, dependency pinning

Optional quick checks you can run before sharing:
- Profiling CPU: python -m cProfile -o prof.out your_script.py args... and visualize with snakeviz prof.out
- Line profiling: line_profiler or scalene
- Memory: pip install memray or tracemalloc
- Pandas: df.info(), df.memory_usage(deep=True), sample timings with %timeit

Paste the code and any context, and I’ll produce a targeted report with concrete fixes and before/after examples.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions. Please paste the actual code so I can analyze it.

To provide the most useful review, include:
- Python version and key libraries
- Typical input sizes/data shapes
- Performance symptoms (slow steps, CPU vs I/O bound)
- Environment (local/serverless/container), OS
- Any constraints (memory/time limits), and target improvements

What I will deliver once I have the code:
1) Performance bottlenecks
- Hot loops, N^2 patterns, unnecessary work
- Repeated I/O or network calls in loops
- Inefficient data structures/algorithms
- Non-vectorized NumPy/Pandas usage, inefficient groupby/apply
- Redundant allocations, copies, conversions
- Regex compilation in loops, string concatenation in loops
- Serialization or logging overhead
- Sync network calls where batching/async helps
- CPU-bound work on threads (GIL issues), use of processes/numba/cython where apt

2) Code quality issues
- Readability, naming, structure, duplication
- Function length/complexity, missing docstrings/comments
- Misuse of exceptions, lack of context managers for resources
- Default-mutable args, shadowing builtins, global state
- Inconsistent or missing type hints, weak validation
- Logging vs print, missing tests, lack of separation of concerns

3) Best practice violations
- PEP 8/257 style, import hygiene, f-strings
- Data classes/namedtuples for simple containers
- pathlib for paths, timezone-aware datetimes
- Caching (functools.lru_cache), memoization where applicable
- Safe APIs (avoid eval/exec, pickle on untrusted data, yaml.safe_load)
- Deterministic randomness seeding, reproducibility
- __main__ guard, packaging/layout, dependency pinning

Optional quick checks you can run before sharing:
- Profiling CPU: python -m cProfile -o prof.out your_script.py args... and visualize with snakeviz prof.out
- Line profiling: line_profiler or scalene
- Memory: pip install memray or tracemalloc
- Pandas: df.info(), df.memory_usage(deep=True), sample timings with %timeit

Paste the code and any context, and I’ll produce a targeted report with concrete fixes and before/after examples.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions. Please paste the actual code so I can analyze it.

To provide the most useful review, include:
- Python version and key libraries
- Typical input sizes/data shapes
- Performance symptoms (slow steps, CPU vs I/O bound)
- Environment (local/serverless/container), OS
- Any constraints (memory/time limits), and target improvements

What I will deliver once I have the code:
1) Performance bottlenecks
- Hot loops, N^2 patterns, unnecessary work
- Repeated I/O or network calls in loops
- Inefficient data structures/algorithms
- Non-vectorized NumPy/Pandas usage, inefficient groupby/apply
- Redundant allocations, copies, conversions
- Regex compilation in loops, string concatenation in loops
- Serialization or logging overhead
- Sync network calls where batching/async helps
- CPU-bound work on threads (GIL issues), use of processes/numba/cython where apt

2) Code quality issues
- Readability, naming, structure, duplication
- Function length/complexity, missing docstrings/comments
- Misuse of exceptions, lack of context managers for resources
- Default-mutable args, shadowing builtins, global state
- Inconsistent or missing type hints, weak validation
- Logging vs print, missing tests, lack of separation of concerns

3) Best practice violations
- PEP 8/257 style, import hygiene, f-strings
- Data classes/namedtuples for simple containers
- pathlib for paths, timezone-aware datetimes
- Caching (functools.lru_cache), memoization where applicable
- Safe APIs (avoid eval/exec, pickle on untrusted data, yaml.safe_load)
- Deterministic randomness seeding, reproducibility
- __main__ guard, packaging/layout, dependency pinning

Optional quick checks you can run before sharing:
- Profiling CPU: python -m cProfile -o prof.out your_script.py args... and visualize with snakeviz prof.out
- Line profiling: line_profiler or scalene
- Memory: pip install memray or tracemalloc
- Pandas: df.info(), df.memory_usage(deep=True), sample timings with %timeit

Paste the code and any context, and I’ll produce a targeted report with concrete fixes and before/after examples.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions. Please paste the actual code so I can analyze it.

To provide the most useful review, include:
- Python version and key libraries
- Typical input sizes/data shapes
- Performance symptoms (slow steps, CPU vs I/O bound)
- Environment (local/serverless/container), OS
- Any constraints (memory/time limits), and target improvements

What I will deliver once I have the code:
1) Performance bottlenecks
- Hot loops, N^2 patterns, unnecessary work
- Repeated I/O or network calls in loops
- Inefficient data structures/algorithms
- Non-vectorized NumPy/Pandas usage, inefficient groupby/apply
- Redundant allocations, copies, conversions
- Regex compilation in loops, string concatenation in loops
- Serialization or logging overhead
- Sync network calls where batching/async helps
- CPU-bound work on threads (GIL issues), use of processes/numba/cython where apt

2) Code quality issues
- Readability, naming, structure, duplication
- Function length/complexity, missing docstrings/comments
- Misuse of exceptions, lack of context managers for resources
- Default-mutable args, shadowing builtins, global state
- Inconsistent or missing type hints, weak validation
- Logging vs print, missing tests, lack of separation of concerns

3) Best practice violations
- PEP 8/257 style, import hygiene, f-strings
- Data classes/namedtuples for simple containers
- pathlib for paths, timezone-aware datetimes
- Caching (functools.lru_cache), memoization where applicable
- Safe APIs (avoid eval/exec, pickle on untrusted data, yaml.safe_load)
- Deterministic randomness seeding, reproducibility
- __main__ guard, packaging/layout, dependency pinning

Optional quick checks you can run before sharing:
- Profiling CPU: python -m cProfile -o prof.out your_script.py args... and visualize with snakeviz prof.out
- Line profiling: line_profiler or scalene
- Memory: pip install memray or tracemalloc
- Pandas: df.info(), df.memory_usage(deep=True), sample timings with %timeit

Paste the code and any context, and I’ll produce a targeted report with concrete fixes and before/after examples.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions. Please paste the actual code so I can analyze it.

To provide the most useful review, include:
- Python version and key libraries
- Typical input sizes/data shapes
- Performance symptoms (slow steps, CPU vs I/O bound)
- Environment (local/serverless/container), OS
- Any constraints (memory/time limits), and target improvements

What I will deliver once I have the code:
1) Performance bottlenecks
- Hot loops, N^2 patterns, unnecessary work
- Repeated I/O or network calls in loops
- Inefficient data structures/algorithms
- Non-vectorized NumPy/Pandas usage, inefficient groupby/apply
- Redundant allocations, copies, conversions
- Regex compilation in loops, string concatenation in loops
- Serialization or logging overhead
- Sync network calls where batching/async helps
- CPU-bound work on threads (GIL issues), use of processes/numba/cython where apt

2) Code quality issues
- Readability, naming, structure, duplication
- Function length/complexity, missing docstrings/comments
- Misuse of exceptions, lack of context managers for resources
- Default-mutable args, shadowing builtins, global state
- Inconsistent or missing type hints, weak validation
- Logging vs print, missing tests, lack of separation of concerns

3) Best practice violations
- PEP 8/257 style, import hygiene, f-strings
- Data classes/namedtuples for simple containers
- pathlib for paths, timezone-aware datetimes
- Caching (functools.lru_cache), memoization where applicable
- Safe APIs (avoid eval/exec, pickle on untrusted data, yaml.safe_load)
- Deterministic randomness seeding, reproducibility
- __main__ guard, packaging/layout, dependency pinning

Optional quick checks you can run before sharing:
- Profiling CPU: python -m cProfile -o prof.out your_script.py args... and visualize with snakeviz prof.out
- Line profiling: line_profiler or scalene
- Memory: pip install memray or tracemalloc
- Pandas: df.info(), df.memory_usage(deep=True), sample timings with %timeit

Paste the code and any context, and I’ll produce a targeted report with concrete fixes and before/after examples.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions. Please paste the actual code so I can analyze it.

To provide the most useful review, include:
- Python version and key libraries
- Typical input sizes/data shapes
- Performance symptoms (slow steps, CPU vs I/O bound)
- Environment (local/serverless/container), OS
- Any constraints (memory/time limits), and target improvements

What I will deliver once I have the code:
1) Performance bottlenecks
- Hot loops, N^2 patterns, unnecessary work
- Repeated I/O or network calls in loops
- Inefficient data structures/algorithms
- Non-vectorized NumPy/Pandas usage, inefficient groupby/apply
- Redundant allocations, copies, conversions
- Regex compilation in loops, string concatenation in loops
- Serialization or logging overhead
- Sync network calls where batching/async helps
- CPU-bound work on threads (GIL issues), use of processes/numba/cython where apt

2) Code quality issues
- Readability, naming, structure, duplication
- Function length/complexity, missing docstrings/comments
- Misuse of exceptions, lack of context managers for resources
- Default-mutable args, shadowing builtins, global state
- Inconsistent or missing type hints, weak validation
- Logging vs print, missing tests, lack of separation of concerns

3) Best practice violations
- PEP 8/257 style, import hygiene, f-strings
- Data classes/namedtuples for simple containers
- pathlib for paths, timezone-aware datetimes
- Caching (functools.lru_cache), memoization where applicable
- Safe APIs (avoid eval/exec, pickle on untrusted data, yaml.safe_load)
- Deterministic randomness seeding, reproducibility
- __main__ guard, packaging/layout, dependency pinning

Optional quick checks you can run before sharing:
- Profiling CPU: python -m cProfile -o prof.out your_script.py args... and visualize with snakeviz prof.out
- Line profiling: line_profiler or scalene
- Memory: pip install memray or tracemalloc
- Pandas: df.info(), df.memory_usage(deep=True), sample timings with %timeit

Paste the code and any context, and I’ll produce a targeted report with concrete fixes and before/after examples.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions. Please paste the actual code so I can analyze it.

To provide the most useful review, include:
- Python version and key libraries
- Typical input sizes/data shapes
- Performance symptoms (slow steps, CPU vs I/O bound)
- Environment (local/serverless/container), OS
- Any constraints (memory/time limits), and target improvements

What I will deliver once I have the code:
1) Performance bottlenecks
- Hot loops, N^2 patterns, unnecessary work
- Repeated I/O or network calls in loops
- Inefficient data structures/algorithms
- Non-vectorized NumPy/Pandas usage, inefficient groupby/apply
- Redundant allocations, copies, conversions
- Regex compilation in loops, string concatenation in loops
- Serialization or logging overhead
- Sync network calls where batching/async helps
- CPU-bound work on threads (GIL issues), use of processes/numba/cython where apt

2) Code quality issues
- Readability, naming, structure, duplication
- Function length/complexity, missing docstrings/comments
- Misuse of exceptions, lack of context managers for resources
- Default-mutable args, shadowing builtins, global state
- Inconsistent or missing type hints, weak validation
- Logging vs print, missing tests, lack of separation of concerns

3) Best practice violations
- PEP 8/257 style, import hygiene, f-strings
- Data classes/namedtuples for simple containers
- pathlib for paths, timezone-aware datetimes
- Caching (functools.lru_cache), memoization where applicable
- Safe APIs (avoid eval/exec, pickle on untrusted data, yaml.safe_load)
- Deterministic randomness seeding, reproducibility
- __main__ guard, packaging/layout, dependency pinning

Optional quick checks you can run before sharing:
- Profiling CPU: python -m cProfile -o prof.out your_script.py args... and visualize with snakeviz prof.out
- Line profiling: line_profiler or scalene
- Memory: pip install memray or tracemalloc
- Pandas: df.info(), df.memory_usage(deep=True), sample timings with %timeit

Paste the code and any context, and I’ll produce a targeted report with concrete fixes and before/after examples.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions. Please paste the actual code so I can analyze it.

To provide the most useful review, include:
- Python version and key libraries
- Typical input sizes/data shapes
- Performance symptoms (slow steps, CPU vs I/O bound)
- Environment (local/serverless/container), OS
- Any constraints (memory/time limits), and target improvements

What I will deliver once I have the code:
1) Performance bottlenecks
- Hot loops, N^2 patterns, unnecessary work
- Repeated I/O or network calls in loops
- Inefficient data structures/algorithms
- Non-vectorized NumPy/Pandas usage, inefficient groupby/apply
- Redundant allocations, copies, conversions
- Regex compilation in loops, string concatenation in loops
- Serialization or logging overhead
- Sync network calls where batching/async helps
- CPU-bound work on threads (GIL issues), use of processes/numba/cython where apt

2) Code quality issues
- Readability, naming, structure, duplication
- Function length/complexity, missing docstrings/comments
- Misuse of exceptions, lack of context managers for resources
- Default-mutable args, shadowing builtins, global state
- Inconsistent or missing type hints, weak validation
- Logging vs print, missing tests, lack of separation of concerns

3) Best practice violations
- PEP 8/257 style, import hygiene, f-strings
- Data classes/namedtuples for simple containers
- pathlib for paths, timezone-aware datetimes
- Caching (functools.lru_cache), memoization where applicable
- Safe APIs (avoid eval/exec, pickle on untrusted data, yaml.safe_load)
- Deterministic randomness seeding, reproducibility
- __main__ guard, packaging/layout, dependency pinning

Optional quick checks you can run before sharing:
- Profiling CPU: python -m cProfile -o prof.out your_script.py args... and visualize with snakeviz prof.out
- Line profiling: line_profiler or scalene
- Memory: pip install memray or tracemalloc
- Pandas: df.info(), df.memory_usage(deep=True), sample timings with %timeit

Paste the code and any context, and I’ll produce a targeted report with concrete fixes and before/after examples.

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