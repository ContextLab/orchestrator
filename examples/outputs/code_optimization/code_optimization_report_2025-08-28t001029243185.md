# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:10:29

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:10:29

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:10:29

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:10:29

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:10:29

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:10:29

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:10:29

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:10:29

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:10:29

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:10:29

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:10:29

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:10:29

## Analysis Results

It looks like the code block contains a placeholder (“{{content}}”) rather than actual Python code. Please paste the code (or a link to a gist/repo) so I can analyze it specifically.

To give you the most useful review, include:
- Python version and key dependencies
- Typical input sizes/data shapes and target performance goals
- Runtime context (CLI, web endpoint, batch job, lambda, etc.)
- Known hot spots or failures

What I will check (and what you can pre-check):

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested comprehensions, N+1 queries)
- Inefficient data structures (lists vs sets/dicts for membership; using DataFrame.apply in pandas instead of vectorization)
- Repeated work (recomputing constants/regex, redundant parsing/serialization, lack of caching/memoization)
- I/O patterns (chatty DB/network calls; small synchronous reads/writes; no batching; blocking in async code)
- String/bytes handling (repeated concatenation in loops; unnecessary encoding/decoding)
- Loop hot paths (attribute/global lookups in tight loops; avoidable conversions; unnecessary deepcopy)
- Concurrency/parallelism (CPU-bound work on CPython without native extensions; missing multiprocessing/numba/vectorization)
- Memory pressure (holding entire datasets; unnecessary copies; pandas chained operations materializing intermediates)

2) Code quality issues
- Complex or long functions; high cyclomatic complexity
- Poor naming, missing docstrings/comments, unclear responsibilities
- Lack of type hints or inconsistent types
- Mutable default arguments
- Exception misuse (broad except, control flow via exceptions, swallowed errors)
- Inconsistent error handling/logging; excessive prints vs logging
- Dead code, duplicated logic, magic numbers, tight coupling

3) Best practice violations
- Style/PEP 8 issues; inconsistent formatting (recommend black/isort/ruff)
- Resource handling without context managers (files, locks, DB connections)
- Not using with for locks/sessions; not closing resources
- Security: hardcoded secrets, unsafe eval/exec, shell=True, SQL injection risk
- Testing: no unit/integration tests; no benchmarks for hotspots
- Packaging: relative imports across packages, missing __init__.py, environment assumptions
- Thread/process safety issues; non-determinism without seeding
- Missing input validation; undefined behavior on edge cases

If you want, I can start with a quick profiling plan once you share the code:
- Macro: run with cProfile + snakeviz/py-spy/scalene to find hotspots
- Micro: timeit/pytest-benchmark for critical functions
- Memory: tracemalloc or scalene, line_profiler/memory_profiler for lines

Paste the code and any context, and I’ll return a focused list of bottlenecks, fixes, and best-practice corrections.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”) rather than actual Python code. Please paste the code (or a link to a gist/repo) so I can analyze it specifically.

To give you the most useful review, include:
- Python version and key dependencies
- Typical input sizes/data shapes and target performance goals
- Runtime context (CLI, web endpoint, batch job, lambda, etc.)
- Known hot spots or failures

What I will check (and what you can pre-check):

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested comprehensions, N+1 queries)
- Inefficient data structures (lists vs sets/dicts for membership; using DataFrame.apply in pandas instead of vectorization)
- Repeated work (recomputing constants/regex, redundant parsing/serialization, lack of caching/memoization)
- I/O patterns (chatty DB/network calls; small synchronous reads/writes; no batching; blocking in async code)
- String/bytes handling (repeated concatenation in loops; unnecessary encoding/decoding)
- Loop hot paths (attribute/global lookups in tight loops; avoidable conversions; unnecessary deepcopy)
- Concurrency/parallelism (CPU-bound work on CPython without native extensions; missing multiprocessing/numba/vectorization)
- Memory pressure (holding entire datasets; unnecessary copies; pandas chained operations materializing intermediates)

2) Code quality issues
- Complex or long functions; high cyclomatic complexity
- Poor naming, missing docstrings/comments, unclear responsibilities
- Lack of type hints or inconsistent types
- Mutable default arguments
- Exception misuse (broad except, control flow via exceptions, swallowed errors)
- Inconsistent error handling/logging; excessive prints vs logging
- Dead code, duplicated logic, magic numbers, tight coupling

3) Best practice violations
- Style/PEP 8 issues; inconsistent formatting (recommend black/isort/ruff)
- Resource handling without context managers (files, locks, DB connections)
- Not using with for locks/sessions; not closing resources
- Security: hardcoded secrets, unsafe eval/exec, shell=True, SQL injection risk
- Testing: no unit/integration tests; no benchmarks for hotspots
- Packaging: relative imports across packages, missing __init__.py, environment assumptions
- Thread/process safety issues; non-determinism without seeding
- Missing input validation; undefined behavior on edge cases

If you want, I can start with a quick profiling plan once you share the code:
- Macro: run with cProfile + snakeviz/py-spy/scalene to find hotspots
- Micro: timeit/pytest-benchmark for critical functions
- Memory: tracemalloc or scalene, line_profiler/memory_profiler for lines

Paste the code and any context, and I’ll return a focused list of bottlenecks, fixes, and best-practice corrections.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”) rather than actual Python code. Please paste the code (or a link to a gist/repo) so I can analyze it specifically.

To give you the most useful review, include:
- Python version and key dependencies
- Typical input sizes/data shapes and target performance goals
- Runtime context (CLI, web endpoint, batch job, lambda, etc.)
- Known hot spots or failures

What I will check (and what you can pre-check):

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested comprehensions, N+1 queries)
- Inefficient data structures (lists vs sets/dicts for membership; using DataFrame.apply in pandas instead of vectorization)
- Repeated work (recomputing constants/regex, redundant parsing/serialization, lack of caching/memoization)
- I/O patterns (chatty DB/network calls; small synchronous reads/writes; no batching; blocking in async code)
- String/bytes handling (repeated concatenation in loops; unnecessary encoding/decoding)
- Loop hot paths (attribute/global lookups in tight loops; avoidable conversions; unnecessary deepcopy)
- Concurrency/parallelism (CPU-bound work on CPython without native extensions; missing multiprocessing/numba/vectorization)
- Memory pressure (holding entire datasets; unnecessary copies; pandas chained operations materializing intermediates)

2) Code quality issues
- Complex or long functions; high cyclomatic complexity
- Poor naming, missing docstrings/comments, unclear responsibilities
- Lack of type hints or inconsistent types
- Mutable default arguments
- Exception misuse (broad except, control flow via exceptions, swallowed errors)
- Inconsistent error handling/logging; excessive prints vs logging
- Dead code, duplicated logic, magic numbers, tight coupling

3) Best practice violations
- Style/PEP 8 issues; inconsistent formatting (recommend black/isort/ruff)
- Resource handling without context managers (files, locks, DB connections)
- Not using with for locks/sessions; not closing resources
- Security: hardcoded secrets, unsafe eval/exec, shell=True, SQL injection risk
- Testing: no unit/integration tests; no benchmarks for hotspots
- Packaging: relative imports across packages, missing __init__.py, environment assumptions
- Thread/process safety issues; non-determinism without seeding
- Missing input validation; undefined behavior on edge cases

If you want, I can start with a quick profiling plan once you share the code:
- Macro: run with cProfile + snakeviz/py-spy/scalene to find hotspots
- Micro: timeit/pytest-benchmark for critical functions
- Memory: tracemalloc or scalene, line_profiler/memory_profiler for lines

Paste the code and any context, and I’ll return a focused list of bottlenecks, fixes, and best-practice corrections.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”) rather than actual Python code. Please paste the code (or a link to a gist/repo) so I can analyze it specifically.

To give you the most useful review, include:
- Python version and key dependencies
- Typical input sizes/data shapes and target performance goals
- Runtime context (CLI, web endpoint, batch job, lambda, etc.)
- Known hot spots or failures

What I will check (and what you can pre-check):

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested comprehensions, N+1 queries)
- Inefficient data structures (lists vs sets/dicts for membership; using DataFrame.apply in pandas instead of vectorization)
- Repeated work (recomputing constants/regex, redundant parsing/serialization, lack of caching/memoization)
- I/O patterns (chatty DB/network calls; small synchronous reads/writes; no batching; blocking in async code)
- String/bytes handling (repeated concatenation in loops; unnecessary encoding/decoding)
- Loop hot paths (attribute/global lookups in tight loops; avoidable conversions; unnecessary deepcopy)
- Concurrency/parallelism (CPU-bound work on CPython without native extensions; missing multiprocessing/numba/vectorization)
- Memory pressure (holding entire datasets; unnecessary copies; pandas chained operations materializing intermediates)

2) Code quality issues
- Complex or long functions; high cyclomatic complexity
- Poor naming, missing docstrings/comments, unclear responsibilities
- Lack of type hints or inconsistent types
- Mutable default arguments
- Exception misuse (broad except, control flow via exceptions, swallowed errors)
- Inconsistent error handling/logging; excessive prints vs logging
- Dead code, duplicated logic, magic numbers, tight coupling

3) Best practice violations
- Style/PEP 8 issues; inconsistent formatting (recommend black/isort/ruff)
- Resource handling without context managers (files, locks, DB connections)
- Not using with for locks/sessions; not closing resources
- Security: hardcoded secrets, unsafe eval/exec, shell=True, SQL injection risk
- Testing: no unit/integration tests; no benchmarks for hotspots
- Packaging: relative imports across packages, missing __init__.py, environment assumptions
- Thread/process safety issues; non-determinism without seeding
- Missing input validation; undefined behavior on edge cases

If you want, I can start with a quick profiling plan once you share the code:
- Macro: run with cProfile + snakeviz/py-spy/scalene to find hotspots
- Micro: timeit/pytest-benchmark for critical functions
- Memory: tracemalloc or scalene, line_profiler/memory_profiler for lines

Paste the code and any context, and I’ll return a focused list of bottlenecks, fixes, and best-practice corrections.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”) rather than actual Python code. Please paste the code (or a link to a gist/repo) so I can analyze it specifically.

To give you the most useful review, include:
- Python version and key dependencies
- Typical input sizes/data shapes and target performance goals
- Runtime context (CLI, web endpoint, batch job, lambda, etc.)
- Known hot spots or failures

What I will check (and what you can pre-check):

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested comprehensions, N+1 queries)
- Inefficient data structures (lists vs sets/dicts for membership; using DataFrame.apply in pandas instead of vectorization)
- Repeated work (recomputing constants/regex, redundant parsing/serialization, lack of caching/memoization)
- I/O patterns (chatty DB/network calls; small synchronous reads/writes; no batching; blocking in async code)
- String/bytes handling (repeated concatenation in loops; unnecessary encoding/decoding)
- Loop hot paths (attribute/global lookups in tight loops; avoidable conversions; unnecessary deepcopy)
- Concurrency/parallelism (CPU-bound work on CPython without native extensions; missing multiprocessing/numba/vectorization)
- Memory pressure (holding entire datasets; unnecessary copies; pandas chained operations materializing intermediates)

2) Code quality issues
- Complex or long functions; high cyclomatic complexity
- Poor naming, missing docstrings/comments, unclear responsibilities
- Lack of type hints or inconsistent types
- Mutable default arguments
- Exception misuse (broad except, control flow via exceptions, swallowed errors)
- Inconsistent error handling/logging; excessive prints vs logging
- Dead code, duplicated logic, magic numbers, tight coupling

3) Best practice violations
- Style/PEP 8 issues; inconsistent formatting (recommend black/isort/ruff)
- Resource handling without context managers (files, locks, DB connections)
- Not using with for locks/sessions; not closing resources
- Security: hardcoded secrets, unsafe eval/exec, shell=True, SQL injection risk
- Testing: no unit/integration tests; no benchmarks for hotspots
- Packaging: relative imports across packages, missing __init__.py, environment assumptions
- Thread/process safety issues; non-determinism without seeding
- Missing input validation; undefined behavior on edge cases

If you want, I can start with a quick profiling plan once you share the code:
- Macro: run with cProfile + snakeviz/py-spy/scalene to find hotspots
- Micro: timeit/pytest-benchmark for critical functions
- Memory: tracemalloc or scalene, line_profiler/memory_profiler for lines

Paste the code and any context, and I’ll return a focused list of bottlenecks, fixes, and best-practice corrections.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”) rather than actual Python code. Please paste the code (or a link to a gist/repo) so I can analyze it specifically.

To give you the most useful review, include:
- Python version and key dependencies
- Typical input sizes/data shapes and target performance goals
- Runtime context (CLI, web endpoint, batch job, lambda, etc.)
- Known hot spots or failures

What I will check (and what you can pre-check):

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested comprehensions, N+1 queries)
- Inefficient data structures (lists vs sets/dicts for membership; using DataFrame.apply in pandas instead of vectorization)
- Repeated work (recomputing constants/regex, redundant parsing/serialization, lack of caching/memoization)
- I/O patterns (chatty DB/network calls; small synchronous reads/writes; no batching; blocking in async code)
- String/bytes handling (repeated concatenation in loops; unnecessary encoding/decoding)
- Loop hot paths (attribute/global lookups in tight loops; avoidable conversions; unnecessary deepcopy)
- Concurrency/parallelism (CPU-bound work on CPython without native extensions; missing multiprocessing/numba/vectorization)
- Memory pressure (holding entire datasets; unnecessary copies; pandas chained operations materializing intermediates)

2) Code quality issues
- Complex or long functions; high cyclomatic complexity
- Poor naming, missing docstrings/comments, unclear responsibilities
- Lack of type hints or inconsistent types
- Mutable default arguments
- Exception misuse (broad except, control flow via exceptions, swallowed errors)
- Inconsistent error handling/logging; excessive prints vs logging
- Dead code, duplicated logic, magic numbers, tight coupling

3) Best practice violations
- Style/PEP 8 issues; inconsistent formatting (recommend black/isort/ruff)
- Resource handling without context managers (files, locks, DB connections)
- Not using with for locks/sessions; not closing resources
- Security: hardcoded secrets, unsafe eval/exec, shell=True, SQL injection risk
- Testing: no unit/integration tests; no benchmarks for hotspots
- Packaging: relative imports across packages, missing __init__.py, environment assumptions
- Thread/process safety issues; non-determinism without seeding
- Missing input validation; undefined behavior on edge cases

If you want, I can start with a quick profiling plan once you share the code:
- Macro: run with cProfile + snakeviz/py-spy/scalene to find hotspots
- Micro: timeit/pytest-benchmark for critical functions
- Memory: tracemalloc or scalene, line_profiler/memory_profiler for lines

Paste the code and any context, and I’ll return a focused list of bottlenecks, fixes, and best-practice corrections.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”) rather than actual Python code. Please paste the code (or a link to a gist/repo) so I can analyze it specifically.

To give you the most useful review, include:
- Python version and key dependencies
- Typical input sizes/data shapes and target performance goals
- Runtime context (CLI, web endpoint, batch job, lambda, etc.)
- Known hot spots or failures

What I will check (and what you can pre-check):

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested comprehensions, N+1 queries)
- Inefficient data structures (lists vs sets/dicts for membership; using DataFrame.apply in pandas instead of vectorization)
- Repeated work (recomputing constants/regex, redundant parsing/serialization, lack of caching/memoization)
- I/O patterns (chatty DB/network calls; small synchronous reads/writes; no batching; blocking in async code)
- String/bytes handling (repeated concatenation in loops; unnecessary encoding/decoding)
- Loop hot paths (attribute/global lookups in tight loops; avoidable conversions; unnecessary deepcopy)
- Concurrency/parallelism (CPU-bound work on CPython without native extensions; missing multiprocessing/numba/vectorization)
- Memory pressure (holding entire datasets; unnecessary copies; pandas chained operations materializing intermediates)

2) Code quality issues
- Complex or long functions; high cyclomatic complexity
- Poor naming, missing docstrings/comments, unclear responsibilities
- Lack of type hints or inconsistent types
- Mutable default arguments
- Exception misuse (broad except, control flow via exceptions, swallowed errors)
- Inconsistent error handling/logging; excessive prints vs logging
- Dead code, duplicated logic, magic numbers, tight coupling

3) Best practice violations
- Style/PEP 8 issues; inconsistent formatting (recommend black/isort/ruff)
- Resource handling without context managers (files, locks, DB connections)
- Not using with for locks/sessions; not closing resources
- Security: hardcoded secrets, unsafe eval/exec, shell=True, SQL injection risk
- Testing: no unit/integration tests; no benchmarks for hotspots
- Packaging: relative imports across packages, missing __init__.py, environment assumptions
- Thread/process safety issues; non-determinism without seeding
- Missing input validation; undefined behavior on edge cases

If you want, I can start with a quick profiling plan once you share the code:
- Macro: run with cProfile + snakeviz/py-spy/scalene to find hotspots
- Micro: timeit/pytest-benchmark for critical functions
- Memory: tracemalloc or scalene, line_profiler/memory_profiler for lines

Paste the code and any context, and I’ll return a focused list of bottlenecks, fixes, and best-practice corrections.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”) rather than actual Python code. Please paste the code (or a link to a gist/repo) so I can analyze it specifically.

To give you the most useful review, include:
- Python version and key dependencies
- Typical input sizes/data shapes and target performance goals
- Runtime context (CLI, web endpoint, batch job, lambda, etc.)
- Known hot spots or failures

What I will check (and what you can pre-check):

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested comprehensions, N+1 queries)
- Inefficient data structures (lists vs sets/dicts for membership; using DataFrame.apply in pandas instead of vectorization)
- Repeated work (recomputing constants/regex, redundant parsing/serialization, lack of caching/memoization)
- I/O patterns (chatty DB/network calls; small synchronous reads/writes; no batching; blocking in async code)
- String/bytes handling (repeated concatenation in loops; unnecessary encoding/decoding)
- Loop hot paths (attribute/global lookups in tight loops; avoidable conversions; unnecessary deepcopy)
- Concurrency/parallelism (CPU-bound work on CPython without native extensions; missing multiprocessing/numba/vectorization)
- Memory pressure (holding entire datasets; unnecessary copies; pandas chained operations materializing intermediates)

2) Code quality issues
- Complex or long functions; high cyclomatic complexity
- Poor naming, missing docstrings/comments, unclear responsibilities
- Lack of type hints or inconsistent types
- Mutable default arguments
- Exception misuse (broad except, control flow via exceptions, swallowed errors)
- Inconsistent error handling/logging; excessive prints vs logging
- Dead code, duplicated logic, magic numbers, tight coupling

3) Best practice violations
- Style/PEP 8 issues; inconsistent formatting (recommend black/isort/ruff)
- Resource handling without context managers (files, locks, DB connections)
- Not using with for locks/sessions; not closing resources
- Security: hardcoded secrets, unsafe eval/exec, shell=True, SQL injection risk
- Testing: no unit/integration tests; no benchmarks for hotspots
- Packaging: relative imports across packages, missing __init__.py, environment assumptions
- Thread/process safety issues; non-determinism without seeding
- Missing input validation; undefined behavior on edge cases

If you want, I can start with a quick profiling plan once you share the code:
- Macro: run with cProfile + snakeviz/py-spy/scalene to find hotspots
- Micro: timeit/pytest-benchmark for critical functions
- Memory: tracemalloc or scalene, line_profiler/memory_profiler for lines

Paste the code and any context, and I’ll return a focused list of bottlenecks, fixes, and best-practice corrections.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”) rather than actual Python code. Please paste the code (or a link to a gist/repo) so I can analyze it specifically.

To give you the most useful review, include:
- Python version and key dependencies
- Typical input sizes/data shapes and target performance goals
- Runtime context (CLI, web endpoint, batch job, lambda, etc.)
- Known hot spots or failures

What I will check (and what you can pre-check):

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested comprehensions, N+1 queries)
- Inefficient data structures (lists vs sets/dicts for membership; using DataFrame.apply in pandas instead of vectorization)
- Repeated work (recomputing constants/regex, redundant parsing/serialization, lack of caching/memoization)
- I/O patterns (chatty DB/network calls; small synchronous reads/writes; no batching; blocking in async code)
- String/bytes handling (repeated concatenation in loops; unnecessary encoding/decoding)
- Loop hot paths (attribute/global lookups in tight loops; avoidable conversions; unnecessary deepcopy)
- Concurrency/parallelism (CPU-bound work on CPython without native extensions; missing multiprocessing/numba/vectorization)
- Memory pressure (holding entire datasets; unnecessary copies; pandas chained operations materializing intermediates)

2) Code quality issues
- Complex or long functions; high cyclomatic complexity
- Poor naming, missing docstrings/comments, unclear responsibilities
- Lack of type hints or inconsistent types
- Mutable default arguments
- Exception misuse (broad except, control flow via exceptions, swallowed errors)
- Inconsistent error handling/logging; excessive prints vs logging
- Dead code, duplicated logic, magic numbers, tight coupling

3) Best practice violations
- Style/PEP 8 issues; inconsistent formatting (recommend black/isort/ruff)
- Resource handling without context managers (files, locks, DB connections)
- Not using with for locks/sessions; not closing resources
- Security: hardcoded secrets, unsafe eval/exec, shell=True, SQL injection risk
- Testing: no unit/integration tests; no benchmarks for hotspots
- Packaging: relative imports across packages, missing __init__.py, environment assumptions
- Thread/process safety issues; non-determinism without seeding
- Missing input validation; undefined behavior on edge cases

If you want, I can start with a quick profiling plan once you share the code:
- Macro: run with cProfile + snakeviz/py-spy/scalene to find hotspots
- Micro: timeit/pytest-benchmark for critical functions
- Memory: tracemalloc or scalene, line_profiler/memory_profiler for lines

Paste the code and any context, and I’ll return a focused list of bottlenecks, fixes, and best-practice corrections.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”) rather than actual Python code. Please paste the code (or a link to a gist/repo) so I can analyze it specifically.

To give you the most useful review, include:
- Python version and key dependencies
- Typical input sizes/data shapes and target performance goals
- Runtime context (CLI, web endpoint, batch job, lambda, etc.)
- Known hot spots or failures

What I will check (and what you can pre-check):

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested comprehensions, N+1 queries)
- Inefficient data structures (lists vs sets/dicts for membership; using DataFrame.apply in pandas instead of vectorization)
- Repeated work (recomputing constants/regex, redundant parsing/serialization, lack of caching/memoization)
- I/O patterns (chatty DB/network calls; small synchronous reads/writes; no batching; blocking in async code)
- String/bytes handling (repeated concatenation in loops; unnecessary encoding/decoding)
- Loop hot paths (attribute/global lookups in tight loops; avoidable conversions; unnecessary deepcopy)
- Concurrency/parallelism (CPU-bound work on CPython without native extensions; missing multiprocessing/numba/vectorization)
- Memory pressure (holding entire datasets; unnecessary copies; pandas chained operations materializing intermediates)

2) Code quality issues
- Complex or long functions; high cyclomatic complexity
- Poor naming, missing docstrings/comments, unclear responsibilities
- Lack of type hints or inconsistent types
- Mutable default arguments
- Exception misuse (broad except, control flow via exceptions, swallowed errors)
- Inconsistent error handling/logging; excessive prints vs logging
- Dead code, duplicated logic, magic numbers, tight coupling

3) Best practice violations
- Style/PEP 8 issues; inconsistent formatting (recommend black/isort/ruff)
- Resource handling without context managers (files, locks, DB connections)
- Not using with for locks/sessions; not closing resources
- Security: hardcoded secrets, unsafe eval/exec, shell=True, SQL injection risk
- Testing: no unit/integration tests; no benchmarks for hotspots
- Packaging: relative imports across packages, missing __init__.py, environment assumptions
- Thread/process safety issues; non-determinism without seeding
- Missing input validation; undefined behavior on edge cases

If you want, I can start with a quick profiling plan once you share the code:
- Macro: run with cProfile + snakeviz/py-spy/scalene to find hotspots
- Micro: timeit/pytest-benchmark for critical functions
- Memory: tracemalloc or scalene, line_profiler/memory_profiler for lines

Paste the code and any context, and I’ll return a focused list of bottlenecks, fixes, and best-practice corrections.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”) rather than actual Python code. Please paste the code (or a link to a gist/repo) so I can analyze it specifically.

To give you the most useful review, include:
- Python version and key dependencies
- Typical input sizes/data shapes and target performance goals
- Runtime context (CLI, web endpoint, batch job, lambda, etc.)
- Known hot spots or failures

What I will check (and what you can pre-check):

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested comprehensions, N+1 queries)
- Inefficient data structures (lists vs sets/dicts for membership; using DataFrame.apply in pandas instead of vectorization)
- Repeated work (recomputing constants/regex, redundant parsing/serialization, lack of caching/memoization)
- I/O patterns (chatty DB/network calls; small synchronous reads/writes; no batching; blocking in async code)
- String/bytes handling (repeated concatenation in loops; unnecessary encoding/decoding)
- Loop hot paths (attribute/global lookups in tight loops; avoidable conversions; unnecessary deepcopy)
- Concurrency/parallelism (CPU-bound work on CPython without native extensions; missing multiprocessing/numba/vectorization)
- Memory pressure (holding entire datasets; unnecessary copies; pandas chained operations materializing intermediates)

2) Code quality issues
- Complex or long functions; high cyclomatic complexity
- Poor naming, missing docstrings/comments, unclear responsibilities
- Lack of type hints or inconsistent types
- Mutable default arguments
- Exception misuse (broad except, control flow via exceptions, swallowed errors)
- Inconsistent error handling/logging; excessive prints vs logging
- Dead code, duplicated logic, magic numbers, tight coupling

3) Best practice violations
- Style/PEP 8 issues; inconsistent formatting (recommend black/isort/ruff)
- Resource handling without context managers (files, locks, DB connections)
- Not using with for locks/sessions; not closing resources
- Security: hardcoded secrets, unsafe eval/exec, shell=True, SQL injection risk
- Testing: no unit/integration tests; no benchmarks for hotspots
- Packaging: relative imports across packages, missing __init__.py, environment assumptions
- Thread/process safety issues; non-determinism without seeding
- Missing input validation; undefined behavior on edge cases

If you want, I can start with a quick profiling plan once you share the code:
- Macro: run with cProfile + snakeviz/py-spy/scalene to find hotspots
- Micro: timeit/pytest-benchmark for critical functions
- Memory: tracemalloc or scalene, line_profiler/memory_profiler for lines

Paste the code and any context, and I’ll return a focused list of bottlenecks, fixes, and best-practice corrections.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”) rather than actual Python code. Please paste the code (or a link to a gist/repo) so I can analyze it specifically.

To give you the most useful review, include:
- Python version and key dependencies
- Typical input sizes/data shapes and target performance goals
- Runtime context (CLI, web endpoint, batch job, lambda, etc.)
- Known hot spots or failures

What I will check (and what you can pre-check):

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested comprehensions, N+1 queries)
- Inefficient data structures (lists vs sets/dicts for membership; using DataFrame.apply in pandas instead of vectorization)
- Repeated work (recomputing constants/regex, redundant parsing/serialization, lack of caching/memoization)
- I/O patterns (chatty DB/network calls; small synchronous reads/writes; no batching; blocking in async code)
- String/bytes handling (repeated concatenation in loops; unnecessary encoding/decoding)
- Loop hot paths (attribute/global lookups in tight loops; avoidable conversions; unnecessary deepcopy)
- Concurrency/parallelism (CPU-bound work on CPython without native extensions; missing multiprocessing/numba/vectorization)
- Memory pressure (holding entire datasets; unnecessary copies; pandas chained operations materializing intermediates)

2) Code quality issues
- Complex or long functions; high cyclomatic complexity
- Poor naming, missing docstrings/comments, unclear responsibilities
- Lack of type hints or inconsistent types
- Mutable default arguments
- Exception misuse (broad except, control flow via exceptions, swallowed errors)
- Inconsistent error handling/logging; excessive prints vs logging
- Dead code, duplicated logic, magic numbers, tight coupling

3) Best practice violations
- Style/PEP 8 issues; inconsistent formatting (recommend black/isort/ruff)
- Resource handling without context managers (files, locks, DB connections)
- Not using with for locks/sessions; not closing resources
- Security: hardcoded secrets, unsafe eval/exec, shell=True, SQL injection risk
- Testing: no unit/integration tests; no benchmarks for hotspots
- Packaging: relative imports across packages, missing __init__.py, environment assumptions
- Thread/process safety issues; non-determinism without seeding
- Missing input validation; undefined behavior on edge cases

If you want, I can start with a quick profiling plan once you share the code:
- Macro: run with cProfile + snakeviz/py-spy/scalene to find hotspots
- Micro: timeit/pytest-benchmark for critical functions
- Memory: tracemalloc or scalene, line_profiler/memory_profiler for lines

Paste the code and any context, and I’ll return a focused list of bottlenecks, fixes, and best-practice corrections.

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