# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:02:41

## Analysis Results

I don’t see any code in the block (it contains the placeholder # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:02:41

## Analysis Results

I don’t see any code in the block (it contains the placeholder # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:02:41

## Analysis Results

I don’t see any code in the block (it contains the placeholder # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:02:41

## Analysis Results

I don’t see any code in the block (it contains the placeholder # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:02:41

## Analysis Results

I don’t see any code in the block (it contains the placeholder # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:02:41

## Analysis Results

I don’t see any code in the block (it contains the placeholder # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:02:41

## Analysis Results

I don’t see any code in the block (it contains the placeholder # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:02:41

## Analysis Results

I don’t see any code in the block (it contains the placeholder # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:02:41

## Analysis Results

I don’t see any code in the block (it contains the placeholder # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:02:41

## Analysis Results

I don’t see any code in the block (it contains the placeholder # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:02:41

## Analysis Results

I don’t see any code in the block (it contains the placeholder # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:02:41

## Analysis Results

I don’t see any code in the block (it contains the placeholder {{content}}). Please paste the Python code you want reviewed, or share a link/gist. If it’s large, include the hot path or the parts that are slow/problematic, plus context:

- Python version, key libraries (e.g., numpy, pandas, asyncio)
- Typical input sizes and target performance
- Environment (local, serverless, container), OS
- Any profiler output (cProfile, py-spy, scalene) or failing tests

What I will analyze

1) Performance bottlenecks
- Algorithmic complexity and data-structure choices
- Hot loops and opportunities for vectorization or batching
- Excess I/O, chatty network/DB access (N+1 queries), synchronous waits
- Repeated work: redundant computations, missing caching/memoization
- Inefficient pandas/numpy usage (row-wise loops, non-broadcasted ops)
- Memory churn: unnecessary copies, large intermediates, JSON/pickle overhead
- Concurrency model issues (GIL-bound CPU work on threads, asyncio misuse)
- Inefficient regex, recursion depth, repeated object/attribute lookups in hot paths

2) Code quality issues
- Readability: naming, function length, cohesion, duplication
- Error handling and logging quality
- Tests and coverage; determinism of tests
- Type hints, docstrings, comments, public API clarity
- Resource management (context managers), side effects, immutability where helpful
- Module/package structure, separation of concerns, configuration handling

3) Best practice violations
- PEP 8/257 style, PEP 484 typing, f-strings
- Mutable default arguments, broad excepts, bare excepts
- Insecure patterns: eval/exec, pickle on untrusted data, subprocess shell=True, SQL injection
- Hardcoded secrets, weak randomness, timezone-naive datetimes
- Path handling (pathlib), file encodings, locale assumptions
- Dependency pinning, packaging, reproducibility

Optional helpers you can run and share output from
- Profiling: python -m cProfile -o out.prof your_script.py; visualize with snakeviz or py-spy
- Static checks: ruff, mypy, bandit; formatting with black
- Benchmarks: pytest-benchmark

Paste the code and I’ll provide a focused list of bottlenecks, issues, and fixes.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions). Please paste the Python code you want reviewed, or share a link/gist. If it’s large, include the hot path or the parts that are slow/problematic, plus context:

- Python version, key libraries (e.g., numpy, pandas, asyncio)
- Typical input sizes and target performance
- Environment (local, serverless, container), OS
- Any profiler output (cProfile, py-spy, scalene) or failing tests

What I will analyze

1) Performance bottlenecks
- Algorithmic complexity and data-structure choices
- Hot loops and opportunities for vectorization or batching
- Excess I/O, chatty network/DB access (N+1 queries), synchronous waits
- Repeated work: redundant computations, missing caching/memoization
- Inefficient pandas/numpy usage (row-wise loops, non-broadcasted ops)
- Memory churn: unnecessary copies, large intermediates, JSON/pickle overhead
- Concurrency model issues (GIL-bound CPU work on threads, asyncio misuse)
- Inefficient regex, recursion depth, repeated object/attribute lookups in hot paths

2) Code quality issues
- Readability: naming, function length, cohesion, duplication
- Error handling and logging quality
- Tests and coverage; determinism of tests
- Type hints, docstrings, comments, public API clarity
- Resource management (context managers), side effects, immutability where helpful
- Module/package structure, separation of concerns, configuration handling

3) Best practice violations
- PEP 8/257 style, PEP 484 typing, f-strings
- Mutable default arguments, broad excepts, bare excepts
- Insecure patterns: eval/exec, pickle on untrusted data, subprocess shell=True, SQL injection
- Hardcoded secrets, weak randomness, timezone-naive datetimes
- Path handling (pathlib), file encodings, locale assumptions
- Dependency pinning, packaging, reproducibility

Optional helpers you can run and share output from
- Profiling: python -m cProfile -o out.prof your_script.py; visualize with snakeviz or py-spy
- Static checks: ruff, mypy, bandit; formatting with black
- Benchmarks: pytest-benchmark

Paste the code and I’ll provide a focused list of bottlenecks, issues, and fixes.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions). Please paste the Python code you want reviewed, or share a link/gist. If it’s large, include the hot path or the parts that are slow/problematic, plus context:

- Python version, key libraries (e.g., numpy, pandas, asyncio)
- Typical input sizes and target performance
- Environment (local, serverless, container), OS
- Any profiler output (cProfile, py-spy, scalene) or failing tests

What I will analyze

1) Performance bottlenecks
- Algorithmic complexity and data-structure choices
- Hot loops and opportunities for vectorization or batching
- Excess I/O, chatty network/DB access (N+1 queries), synchronous waits
- Repeated work: redundant computations, missing caching/memoization
- Inefficient pandas/numpy usage (row-wise loops, non-broadcasted ops)
- Memory churn: unnecessary copies, large intermediates, JSON/pickle overhead
- Concurrency model issues (GIL-bound CPU work on threads, asyncio misuse)
- Inefficient regex, recursion depth, repeated object/attribute lookups in hot paths

2) Code quality issues
- Readability: naming, function length, cohesion, duplication
- Error handling and logging quality
- Tests and coverage; determinism of tests
- Type hints, docstrings, comments, public API clarity
- Resource management (context managers), side effects, immutability where helpful
- Module/package structure, separation of concerns, configuration handling

3) Best practice violations
- PEP 8/257 style, PEP 484 typing, f-strings
- Mutable default arguments, broad excepts, bare excepts
- Insecure patterns: eval/exec, pickle on untrusted data, subprocess shell=True, SQL injection
- Hardcoded secrets, weak randomness, timezone-naive datetimes
- Path handling (pathlib), file encodings, locale assumptions
- Dependency pinning, packaging, reproducibility

Optional helpers you can run and share output from
- Profiling: python -m cProfile -o out.prof your_script.py; visualize with snakeviz or py-spy
- Static checks: ruff, mypy, bandit; formatting with black
- Benchmarks: pytest-benchmark

Paste the code and I’ll provide a focused list of bottlenecks, issues, and fixes.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions). Please paste the Python code you want reviewed, or share a link/gist. If it’s large, include the hot path or the parts that are slow/problematic, plus context:

- Python version, key libraries (e.g., numpy, pandas, asyncio)
- Typical input sizes and target performance
- Environment (local, serverless, container), OS
- Any profiler output (cProfile, py-spy, scalene) or failing tests

What I will analyze

1) Performance bottlenecks
- Algorithmic complexity and data-structure choices
- Hot loops and opportunities for vectorization or batching
- Excess I/O, chatty network/DB access (N+1 queries), synchronous waits
- Repeated work: redundant computations, missing caching/memoization
- Inefficient pandas/numpy usage (row-wise loops, non-broadcasted ops)
- Memory churn: unnecessary copies, large intermediates, JSON/pickle overhead
- Concurrency model issues (GIL-bound CPU work on threads, asyncio misuse)
- Inefficient regex, recursion depth, repeated object/attribute lookups in hot paths

2) Code quality issues
- Readability: naming, function length, cohesion, duplication
- Error handling and logging quality
- Tests and coverage; determinism of tests
- Type hints, docstrings, comments, public API clarity
- Resource management (context managers), side effects, immutability where helpful
- Module/package structure, separation of concerns, configuration handling

3) Best practice violations
- PEP 8/257 style, PEP 484 typing, f-strings
- Mutable default arguments, broad excepts, bare excepts
- Insecure patterns: eval/exec, pickle on untrusted data, subprocess shell=True, SQL injection
- Hardcoded secrets, weak randomness, timezone-naive datetimes
- Path handling (pathlib), file encodings, locale assumptions
- Dependency pinning, packaging, reproducibility

Optional helpers you can run and share output from
- Profiling: python -m cProfile -o out.prof your_script.py; visualize with snakeviz or py-spy
- Static checks: ruff, mypy, bandit; formatting with black
- Benchmarks: pytest-benchmark

Paste the code and I’ll provide a focused list of bottlenecks, issues, and fixes.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions). Please paste the Python code you want reviewed, or share a link/gist. If it’s large, include the hot path or the parts that are slow/problematic, plus context:

- Python version, key libraries (e.g., numpy, pandas, asyncio)
- Typical input sizes and target performance
- Environment (local, serverless, container), OS
- Any profiler output (cProfile, py-spy, scalene) or failing tests

What I will analyze

1) Performance bottlenecks
- Algorithmic complexity and data-structure choices
- Hot loops and opportunities for vectorization or batching
- Excess I/O, chatty network/DB access (N+1 queries), synchronous waits
- Repeated work: redundant computations, missing caching/memoization
- Inefficient pandas/numpy usage (row-wise loops, non-broadcasted ops)
- Memory churn: unnecessary copies, large intermediates, JSON/pickle overhead
- Concurrency model issues (GIL-bound CPU work on threads, asyncio misuse)
- Inefficient regex, recursion depth, repeated object/attribute lookups in hot paths

2) Code quality issues
- Readability: naming, function length, cohesion, duplication
- Error handling and logging quality
- Tests and coverage; determinism of tests
- Type hints, docstrings, comments, public API clarity
- Resource management (context managers), side effects, immutability where helpful
- Module/package structure, separation of concerns, configuration handling

3) Best practice violations
- PEP 8/257 style, PEP 484 typing, f-strings
- Mutable default arguments, broad excepts, bare excepts
- Insecure patterns: eval/exec, pickle on untrusted data, subprocess shell=True, SQL injection
- Hardcoded secrets, weak randomness, timezone-naive datetimes
- Path handling (pathlib), file encodings, locale assumptions
- Dependency pinning, packaging, reproducibility

Optional helpers you can run and share output from
- Profiling: python -m cProfile -o out.prof your_script.py; visualize with snakeviz or py-spy
- Static checks: ruff, mypy, bandit; formatting with black
- Benchmarks: pytest-benchmark

Paste the code and I’ll provide a focused list of bottlenecks, issues, and fixes.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions). Please paste the Python code you want reviewed, or share a link/gist. If it’s large, include the hot path or the parts that are slow/problematic, plus context:

- Python version, key libraries (e.g., numpy, pandas, asyncio)
- Typical input sizes and target performance
- Environment (local, serverless, container), OS
- Any profiler output (cProfile, py-spy, scalene) or failing tests

What I will analyze

1) Performance bottlenecks
- Algorithmic complexity and data-structure choices
- Hot loops and opportunities for vectorization or batching
- Excess I/O, chatty network/DB access (N+1 queries), synchronous waits
- Repeated work: redundant computations, missing caching/memoization
- Inefficient pandas/numpy usage (row-wise loops, non-broadcasted ops)
- Memory churn: unnecessary copies, large intermediates, JSON/pickle overhead
- Concurrency model issues (GIL-bound CPU work on threads, asyncio misuse)
- Inefficient regex, recursion depth, repeated object/attribute lookups in hot paths

2) Code quality issues
- Readability: naming, function length, cohesion, duplication
- Error handling and logging quality
- Tests and coverage; determinism of tests
- Type hints, docstrings, comments, public API clarity
- Resource management (context managers), side effects, immutability where helpful
- Module/package structure, separation of concerns, configuration handling

3) Best practice violations
- PEP 8/257 style, PEP 484 typing, f-strings
- Mutable default arguments, broad excepts, bare excepts
- Insecure patterns: eval/exec, pickle on untrusted data, subprocess shell=True, SQL injection
- Hardcoded secrets, weak randomness, timezone-naive datetimes
- Path handling (pathlib), file encodings, locale assumptions
- Dependency pinning, packaging, reproducibility

Optional helpers you can run and share output from
- Profiling: python -m cProfile -o out.prof your_script.py; visualize with snakeviz or py-spy
- Static checks: ruff, mypy, bandit; formatting with black
- Benchmarks: pytest-benchmark

Paste the code and I’ll provide a focused list of bottlenecks, issues, and fixes.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions). Please paste the Python code you want reviewed, or share a link/gist. If it’s large, include the hot path or the parts that are slow/problematic, plus context:

- Python version, key libraries (e.g., numpy, pandas, asyncio)
- Typical input sizes and target performance
- Environment (local, serverless, container), OS
- Any profiler output (cProfile, py-spy, scalene) or failing tests

What I will analyze

1) Performance bottlenecks
- Algorithmic complexity and data-structure choices
- Hot loops and opportunities for vectorization or batching
- Excess I/O, chatty network/DB access (N+1 queries), synchronous waits
- Repeated work: redundant computations, missing caching/memoization
- Inefficient pandas/numpy usage (row-wise loops, non-broadcasted ops)
- Memory churn: unnecessary copies, large intermediates, JSON/pickle overhead
- Concurrency model issues (GIL-bound CPU work on threads, asyncio misuse)
- Inefficient regex, recursion depth, repeated object/attribute lookups in hot paths

2) Code quality issues
- Readability: naming, function length, cohesion, duplication
- Error handling and logging quality
- Tests and coverage; determinism of tests
- Type hints, docstrings, comments, public API clarity
- Resource management (context managers), side effects, immutability where helpful
- Module/package structure, separation of concerns, configuration handling

3) Best practice violations
- PEP 8/257 style, PEP 484 typing, f-strings
- Mutable default arguments, broad excepts, bare excepts
- Insecure patterns: eval/exec, pickle on untrusted data, subprocess shell=True, SQL injection
- Hardcoded secrets, weak randomness, timezone-naive datetimes
- Path handling (pathlib), file encodings, locale assumptions
- Dependency pinning, packaging, reproducibility

Optional helpers you can run and share output from
- Profiling: python -m cProfile -o out.prof your_script.py; visualize with snakeviz or py-spy
- Static checks: ruff, mypy, bandit; formatting with black
- Benchmarks: pytest-benchmark

Paste the code and I’ll provide a focused list of bottlenecks, issues, and fixes.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions). Please paste the Python code you want reviewed, or share a link/gist. If it’s large, include the hot path or the parts that are slow/problematic, plus context:

- Python version, key libraries (e.g., numpy, pandas, asyncio)
- Typical input sizes and target performance
- Environment (local, serverless, container), OS
- Any profiler output (cProfile, py-spy, scalene) or failing tests

What I will analyze

1) Performance bottlenecks
- Algorithmic complexity and data-structure choices
- Hot loops and opportunities for vectorization or batching
- Excess I/O, chatty network/DB access (N+1 queries), synchronous waits
- Repeated work: redundant computations, missing caching/memoization
- Inefficient pandas/numpy usage (row-wise loops, non-broadcasted ops)
- Memory churn: unnecessary copies, large intermediates, JSON/pickle overhead
- Concurrency model issues (GIL-bound CPU work on threads, asyncio misuse)
- Inefficient regex, recursion depth, repeated object/attribute lookups in hot paths

2) Code quality issues
- Readability: naming, function length, cohesion, duplication
- Error handling and logging quality
- Tests and coverage; determinism of tests
- Type hints, docstrings, comments, public API clarity
- Resource management (context managers), side effects, immutability where helpful
- Module/package structure, separation of concerns, configuration handling

3) Best practice violations
- PEP 8/257 style, PEP 484 typing, f-strings
- Mutable default arguments, broad excepts, bare excepts
- Insecure patterns: eval/exec, pickle on untrusted data, subprocess shell=True, SQL injection
- Hardcoded secrets, weak randomness, timezone-naive datetimes
- Path handling (pathlib), file encodings, locale assumptions
- Dependency pinning, packaging, reproducibility

Optional helpers you can run and share output from
- Profiling: python -m cProfile -o out.prof your_script.py; visualize with snakeviz or py-spy
- Static checks: ruff, mypy, bandit; formatting with black
- Benchmarks: pytest-benchmark

Paste the code and I’ll provide a focused list of bottlenecks, issues, and fixes.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions). Please paste the Python code you want reviewed, or share a link/gist. If it’s large, include the hot path or the parts that are slow/problematic, plus context:

- Python version, key libraries (e.g., numpy, pandas, asyncio)
- Typical input sizes and target performance
- Environment (local, serverless, container), OS
- Any profiler output (cProfile, py-spy, scalene) or failing tests

What I will analyze

1) Performance bottlenecks
- Algorithmic complexity and data-structure choices
- Hot loops and opportunities for vectorization or batching
- Excess I/O, chatty network/DB access (N+1 queries), synchronous waits
- Repeated work: redundant computations, missing caching/memoization
- Inefficient pandas/numpy usage (row-wise loops, non-broadcasted ops)
- Memory churn: unnecessary copies, large intermediates, JSON/pickle overhead
- Concurrency model issues (GIL-bound CPU work on threads, asyncio misuse)
- Inefficient regex, recursion depth, repeated object/attribute lookups in hot paths

2) Code quality issues
- Readability: naming, function length, cohesion, duplication
- Error handling and logging quality
- Tests and coverage; determinism of tests
- Type hints, docstrings, comments, public API clarity
- Resource management (context managers), side effects, immutability where helpful
- Module/package structure, separation of concerns, configuration handling

3) Best practice violations
- PEP 8/257 style, PEP 484 typing, f-strings
- Mutable default arguments, broad excepts, bare excepts
- Insecure patterns: eval/exec, pickle on untrusted data, subprocess shell=True, SQL injection
- Hardcoded secrets, weak randomness, timezone-naive datetimes
- Path handling (pathlib), file encodings, locale assumptions
- Dependency pinning, packaging, reproducibility

Optional helpers you can run and share output from
- Profiling: python -m cProfile -o out.prof your_script.py; visualize with snakeviz or py-spy
- Static checks: ruff, mypy, bandit; formatting with black
- Benchmarks: pytest-benchmark

Paste the code and I’ll provide a focused list of bottlenecks, issues, and fixes.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions). Please paste the Python code you want reviewed, or share a link/gist. If it’s large, include the hot path or the parts that are slow/problematic, plus context:

- Python version, key libraries (e.g., numpy, pandas, asyncio)
- Typical input sizes and target performance
- Environment (local, serverless, container), OS
- Any profiler output (cProfile, py-spy, scalene) or failing tests

What I will analyze

1) Performance bottlenecks
- Algorithmic complexity and data-structure choices
- Hot loops and opportunities for vectorization or batching
- Excess I/O, chatty network/DB access (N+1 queries), synchronous waits
- Repeated work: redundant computations, missing caching/memoization
- Inefficient pandas/numpy usage (row-wise loops, non-broadcasted ops)
- Memory churn: unnecessary copies, large intermediates, JSON/pickle overhead
- Concurrency model issues (GIL-bound CPU work on threads, asyncio misuse)
- Inefficient regex, recursion depth, repeated object/attribute lookups in hot paths

2) Code quality issues
- Readability: naming, function length, cohesion, duplication
- Error handling and logging quality
- Tests and coverage; determinism of tests
- Type hints, docstrings, comments, public API clarity
- Resource management (context managers), side effects, immutability where helpful
- Module/package structure, separation of concerns, configuration handling

3) Best practice violations
- PEP 8/257 style, PEP 484 typing, f-strings
- Mutable default arguments, broad excepts, bare excepts
- Insecure patterns: eval/exec, pickle on untrusted data, subprocess shell=True, SQL injection
- Hardcoded secrets, weak randomness, timezone-naive datetimes
- Path handling (pathlib), file encodings, locale assumptions
- Dependency pinning, packaging, reproducibility

Optional helpers you can run and share output from
- Profiling: python -m cProfile -o out.prof your_script.py; visualize with snakeviz or py-spy
- Static checks: ruff, mypy, bandit; formatting with black
- Benchmarks: pytest-benchmark

Paste the code and I’ll provide a focused list of bottlenecks, issues, and fixes.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions). Please paste the Python code you want reviewed, or share a link/gist. If it’s large, include the hot path or the parts that are slow/problematic, plus context:

- Python version, key libraries (e.g., numpy, pandas, asyncio)
- Typical input sizes and target performance
- Environment (local, serverless, container), OS
- Any profiler output (cProfile, py-spy, scalene) or failing tests

What I will analyze

1) Performance bottlenecks
- Algorithmic complexity and data-structure choices
- Hot loops and opportunities for vectorization or batching
- Excess I/O, chatty network/DB access (N+1 queries), synchronous waits
- Repeated work: redundant computations, missing caching/memoization
- Inefficient pandas/numpy usage (row-wise loops, non-broadcasted ops)
- Memory churn: unnecessary copies, large intermediates, JSON/pickle overhead
- Concurrency model issues (GIL-bound CPU work on threads, asyncio misuse)
- Inefficient regex, recursion depth, repeated object/attribute lookups in hot paths

2) Code quality issues
- Readability: naming, function length, cohesion, duplication
- Error handling and logging quality
- Tests and coverage; determinism of tests
- Type hints, docstrings, comments, public API clarity
- Resource management (context managers), side effects, immutability where helpful
- Module/package structure, separation of concerns, configuration handling

3) Best practice violations
- PEP 8/257 style, PEP 484 typing, f-strings
- Mutable default arguments, broad excepts, bare excepts
- Insecure patterns: eval/exec, pickle on untrusted data, subprocess shell=True, SQL injection
- Hardcoded secrets, weak randomness, timezone-naive datetimes
- Path handling (pathlib), file encodings, locale assumptions
- Dependency pinning, packaging, reproducibility

Optional helpers you can run and share output from
- Profiling: python -m cProfile -o out.prof your_script.py; visualize with snakeviz or py-spy
- Static checks: ruff, mypy, bandit; formatting with black
- Benchmarks: pytest-benchmark

Paste the code and I’ll provide a focused list of bottlenecks, issues, and fixes.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions). Please paste the Python code you want reviewed, or share a link/gist. If it’s large, include the hot path or the parts that are slow/problematic, plus context:

- Python version, key libraries (e.g., numpy, pandas, asyncio)
- Typical input sizes and target performance
- Environment (local, serverless, container), OS
- Any profiler output (cProfile, py-spy, scalene) or failing tests

What I will analyze

1) Performance bottlenecks
- Algorithmic complexity and data-structure choices
- Hot loops and opportunities for vectorization or batching
- Excess I/O, chatty network/DB access (N+1 queries), synchronous waits
- Repeated work: redundant computations, missing caching/memoization
- Inefficient pandas/numpy usage (row-wise loops, non-broadcasted ops)
- Memory churn: unnecessary copies, large intermediates, JSON/pickle overhead
- Concurrency model issues (GIL-bound CPU work on threads, asyncio misuse)
- Inefficient regex, recursion depth, repeated object/attribute lookups in hot paths

2) Code quality issues
- Readability: naming, function length, cohesion, duplication
- Error handling and logging quality
- Tests and coverage; determinism of tests
- Type hints, docstrings, comments, public API clarity
- Resource management (context managers), side effects, immutability where helpful
- Module/package structure, separation of concerns, configuration handling

3) Best practice violations
- PEP 8/257 style, PEP 484 typing, f-strings
- Mutable default arguments, broad excepts, bare excepts
- Insecure patterns: eval/exec, pickle on untrusted data, subprocess shell=True, SQL injection
- Hardcoded secrets, weak randomness, timezone-naive datetimes
- Path handling (pathlib), file encodings, locale assumptions
- Dependency pinning, packaging, reproducibility

Optional helpers you can run and share output from
- Profiling: python -m cProfile -o out.prof your_script.py; visualize with snakeviz or py-spy
- Static checks: ruff, mypy, bandit; formatting with black
- Benchmarks: pytest-benchmark

Paste the code and I’ll provide a focused list of bottlenecks, issues, and fixes.

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