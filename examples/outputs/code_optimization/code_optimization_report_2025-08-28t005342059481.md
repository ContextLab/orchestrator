# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:53:42

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:53:42

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:53:42

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:53:42

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:53:42

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:53:42

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:53:42

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:53:42

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:53:42

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:53:42

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:53:42

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:53:42

## Analysis Results

It looks like the code block contains a placeholder (“{{content}}”) rather than actual Python code. Please paste the code (or a link/gist) so I can run a specific review.

Helpful context to include:
- Python version and runtime (CPython/PyPy), OS
- Typical input sizes and performance goals (e.g., “process 1M rows under 5s”)
- Memory limits and third-party library constraints (e.g., can use numpy/pandas?)
- Whether public APIs/IO formats are fixed or can change

What I will examine once you provide the code:

1) Performance bottlenecks
- Algorithmic complexity (nested loops, O(n^2) patterns), unnecessary sorting or repeated work
- Inefficient data structures (list scans vs set/dict membership; using list where deque/heap is better)
- Hot-loop inefficiencies (per-iteration allocations, repeated attribute lookups, regex compilation, string concatenation with +)
- I/O and serialization overhead (sync file/network calls in tight loops, N+1 DB queries, lack of batching, no HTTP session reuse)
- Missed vectorization/broadcast (numpy/pandas), misuse of pandas apply/iterrows
- Missing caching/memoization for pure or expensive functions; redundant computations

2) Code quality issues
- Large, monolithic functions; duplication; unclear naming; magic numbers
- Mutable default arguments; hidden global state/side effects
- Fragile error handling (bare except, swallowed exceptions), missing context managers for resources
- Resource leaks (files/sockets/processes not closed), blocking calls in async code
- Inconsistent formatting, missing docstrings/type hints, unused imports/variables
- Security footguns (eval/exec, pickle on untrusted data, shell=True, SQL string concatenation)

3) Best practice violations
- PEP 8/257 style, import ordering, wildcard imports
- Type hints and dataclasses where appropriate; clear module boundaries
- Logging instead of print; appropriate log levels and configuration
- Using pathlib, subprocess.run, requests.Session, contextlib for context managers
- Configuration via environment/params instead of hard-coding; avoid secrets in code
- Deterministic randomness and secure primitives where needed (secrets, hmac.compare_digest)

If the code is lengthy, point me to the suspected hotspots or the specific functions/modules to prioritize.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”) rather than actual Python code. Please paste the code (or a link/gist) so I can run a specific review.

Helpful context to include:
- Python version and runtime (CPython/PyPy), OS
- Typical input sizes and performance goals (e.g., “process 1M rows under 5s”)
- Memory limits and third-party library constraints (e.g., can use numpy/pandas?)
- Whether public APIs/IO formats are fixed or can change

What I will examine once you provide the code:

1) Performance bottlenecks
- Algorithmic complexity (nested loops, O(n^2) patterns), unnecessary sorting or repeated work
- Inefficient data structures (list scans vs set/dict membership; using list where deque/heap is better)
- Hot-loop inefficiencies (per-iteration allocations, repeated attribute lookups, regex compilation, string concatenation with +)
- I/O and serialization overhead (sync file/network calls in tight loops, N+1 DB queries, lack of batching, no HTTP session reuse)
- Missed vectorization/broadcast (numpy/pandas), misuse of pandas apply/iterrows
- Missing caching/memoization for pure or expensive functions; redundant computations

2) Code quality issues
- Large, monolithic functions; duplication; unclear naming; magic numbers
- Mutable default arguments; hidden global state/side effects
- Fragile error handling (bare except, swallowed exceptions), missing context managers for resources
- Resource leaks (files/sockets/processes not closed), blocking calls in async code
- Inconsistent formatting, missing docstrings/type hints, unused imports/variables
- Security footguns (eval/exec, pickle on untrusted data, shell=True, SQL string concatenation)

3) Best practice violations
- PEP 8/257 style, import ordering, wildcard imports
- Type hints and dataclasses where appropriate; clear module boundaries
- Logging instead of print; appropriate log levels and configuration
- Using pathlib, subprocess.run, requests.Session, contextlib for context managers
- Configuration via environment/params instead of hard-coding; avoid secrets in code
- Deterministic randomness and secure primitives where needed (secrets, hmac.compare_digest)

If the code is lengthy, point me to the suspected hotspots or the specific functions/modules to prioritize.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”) rather than actual Python code. Please paste the code (or a link/gist) so I can run a specific review.

Helpful context to include:
- Python version and runtime (CPython/PyPy), OS
- Typical input sizes and performance goals (e.g., “process 1M rows under 5s”)
- Memory limits and third-party library constraints (e.g., can use numpy/pandas?)
- Whether public APIs/IO formats are fixed or can change

What I will examine once you provide the code:

1) Performance bottlenecks
- Algorithmic complexity (nested loops, O(n^2) patterns), unnecessary sorting or repeated work
- Inefficient data structures (list scans vs set/dict membership; using list where deque/heap is better)
- Hot-loop inefficiencies (per-iteration allocations, repeated attribute lookups, regex compilation, string concatenation with +)
- I/O and serialization overhead (sync file/network calls in tight loops, N+1 DB queries, lack of batching, no HTTP session reuse)
- Missed vectorization/broadcast (numpy/pandas), misuse of pandas apply/iterrows
- Missing caching/memoization for pure or expensive functions; redundant computations

2) Code quality issues
- Large, monolithic functions; duplication; unclear naming; magic numbers
- Mutable default arguments; hidden global state/side effects
- Fragile error handling (bare except, swallowed exceptions), missing context managers for resources
- Resource leaks (files/sockets/processes not closed), blocking calls in async code
- Inconsistent formatting, missing docstrings/type hints, unused imports/variables
- Security footguns (eval/exec, pickle on untrusted data, shell=True, SQL string concatenation)

3) Best practice violations
- PEP 8/257 style, import ordering, wildcard imports
- Type hints and dataclasses where appropriate; clear module boundaries
- Logging instead of print; appropriate log levels and configuration
- Using pathlib, subprocess.run, requests.Session, contextlib for context managers
- Configuration via environment/params instead of hard-coding; avoid secrets in code
- Deterministic randomness and secure primitives where needed (secrets, hmac.compare_digest)

If the code is lengthy, point me to the suspected hotspots or the specific functions/modules to prioritize.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”) rather than actual Python code. Please paste the code (or a link/gist) so I can run a specific review.

Helpful context to include:
- Python version and runtime (CPython/PyPy), OS
- Typical input sizes and performance goals (e.g., “process 1M rows under 5s”)
- Memory limits and third-party library constraints (e.g., can use numpy/pandas?)
- Whether public APIs/IO formats are fixed or can change

What I will examine once you provide the code:

1) Performance bottlenecks
- Algorithmic complexity (nested loops, O(n^2) patterns), unnecessary sorting or repeated work
- Inefficient data structures (list scans vs set/dict membership; using list where deque/heap is better)
- Hot-loop inefficiencies (per-iteration allocations, repeated attribute lookups, regex compilation, string concatenation with +)
- I/O and serialization overhead (sync file/network calls in tight loops, N+1 DB queries, lack of batching, no HTTP session reuse)
- Missed vectorization/broadcast (numpy/pandas), misuse of pandas apply/iterrows
- Missing caching/memoization for pure or expensive functions; redundant computations

2) Code quality issues
- Large, monolithic functions; duplication; unclear naming; magic numbers
- Mutable default arguments; hidden global state/side effects
- Fragile error handling (bare except, swallowed exceptions), missing context managers for resources
- Resource leaks (files/sockets/processes not closed), blocking calls in async code
- Inconsistent formatting, missing docstrings/type hints, unused imports/variables
- Security footguns (eval/exec, pickle on untrusted data, shell=True, SQL string concatenation)

3) Best practice violations
- PEP 8/257 style, import ordering, wildcard imports
- Type hints and dataclasses where appropriate; clear module boundaries
- Logging instead of print; appropriate log levels and configuration
- Using pathlib, subprocess.run, requests.Session, contextlib for context managers
- Configuration via environment/params instead of hard-coding; avoid secrets in code
- Deterministic randomness and secure primitives where needed (secrets, hmac.compare_digest)

If the code is lengthy, point me to the suspected hotspots or the specific functions/modules to prioritize.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”) rather than actual Python code. Please paste the code (or a link/gist) so I can run a specific review.

Helpful context to include:
- Python version and runtime (CPython/PyPy), OS
- Typical input sizes and performance goals (e.g., “process 1M rows under 5s”)
- Memory limits and third-party library constraints (e.g., can use numpy/pandas?)
- Whether public APIs/IO formats are fixed or can change

What I will examine once you provide the code:

1) Performance bottlenecks
- Algorithmic complexity (nested loops, O(n^2) patterns), unnecessary sorting or repeated work
- Inefficient data structures (list scans vs set/dict membership; using list where deque/heap is better)
- Hot-loop inefficiencies (per-iteration allocations, repeated attribute lookups, regex compilation, string concatenation with +)
- I/O and serialization overhead (sync file/network calls in tight loops, N+1 DB queries, lack of batching, no HTTP session reuse)
- Missed vectorization/broadcast (numpy/pandas), misuse of pandas apply/iterrows
- Missing caching/memoization for pure or expensive functions; redundant computations

2) Code quality issues
- Large, monolithic functions; duplication; unclear naming; magic numbers
- Mutable default arguments; hidden global state/side effects
- Fragile error handling (bare except, swallowed exceptions), missing context managers for resources
- Resource leaks (files/sockets/processes not closed), blocking calls in async code
- Inconsistent formatting, missing docstrings/type hints, unused imports/variables
- Security footguns (eval/exec, pickle on untrusted data, shell=True, SQL string concatenation)

3) Best practice violations
- PEP 8/257 style, import ordering, wildcard imports
- Type hints and dataclasses where appropriate; clear module boundaries
- Logging instead of print; appropriate log levels and configuration
- Using pathlib, subprocess.run, requests.Session, contextlib for context managers
- Configuration via environment/params instead of hard-coding; avoid secrets in code
- Deterministic randomness and secure primitives where needed (secrets, hmac.compare_digest)

If the code is lengthy, point me to the suspected hotspots or the specific functions/modules to prioritize.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”) rather than actual Python code. Please paste the code (or a link/gist) so I can run a specific review.

Helpful context to include:
- Python version and runtime (CPython/PyPy), OS
- Typical input sizes and performance goals (e.g., “process 1M rows under 5s”)
- Memory limits and third-party library constraints (e.g., can use numpy/pandas?)
- Whether public APIs/IO formats are fixed or can change

What I will examine once you provide the code:

1) Performance bottlenecks
- Algorithmic complexity (nested loops, O(n^2) patterns), unnecessary sorting or repeated work
- Inefficient data structures (list scans vs set/dict membership; using list where deque/heap is better)
- Hot-loop inefficiencies (per-iteration allocations, repeated attribute lookups, regex compilation, string concatenation with +)
- I/O and serialization overhead (sync file/network calls in tight loops, N+1 DB queries, lack of batching, no HTTP session reuse)
- Missed vectorization/broadcast (numpy/pandas), misuse of pandas apply/iterrows
- Missing caching/memoization for pure or expensive functions; redundant computations

2) Code quality issues
- Large, monolithic functions; duplication; unclear naming; magic numbers
- Mutable default arguments; hidden global state/side effects
- Fragile error handling (bare except, swallowed exceptions), missing context managers for resources
- Resource leaks (files/sockets/processes not closed), blocking calls in async code
- Inconsistent formatting, missing docstrings/type hints, unused imports/variables
- Security footguns (eval/exec, pickle on untrusted data, shell=True, SQL string concatenation)

3) Best practice violations
- PEP 8/257 style, import ordering, wildcard imports
- Type hints and dataclasses where appropriate; clear module boundaries
- Logging instead of print; appropriate log levels and configuration
- Using pathlib, subprocess.run, requests.Session, contextlib for context managers
- Configuration via environment/params instead of hard-coding; avoid secrets in code
- Deterministic randomness and secure primitives where needed (secrets, hmac.compare_digest)

If the code is lengthy, point me to the suspected hotspots or the specific functions/modules to prioritize.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”) rather than actual Python code. Please paste the code (or a link/gist) so I can run a specific review.

Helpful context to include:
- Python version and runtime (CPython/PyPy), OS
- Typical input sizes and performance goals (e.g., “process 1M rows under 5s”)
- Memory limits and third-party library constraints (e.g., can use numpy/pandas?)
- Whether public APIs/IO formats are fixed or can change

What I will examine once you provide the code:

1) Performance bottlenecks
- Algorithmic complexity (nested loops, O(n^2) patterns), unnecessary sorting or repeated work
- Inefficient data structures (list scans vs set/dict membership; using list where deque/heap is better)
- Hot-loop inefficiencies (per-iteration allocations, repeated attribute lookups, regex compilation, string concatenation with +)
- I/O and serialization overhead (sync file/network calls in tight loops, N+1 DB queries, lack of batching, no HTTP session reuse)
- Missed vectorization/broadcast (numpy/pandas), misuse of pandas apply/iterrows
- Missing caching/memoization for pure or expensive functions; redundant computations

2) Code quality issues
- Large, monolithic functions; duplication; unclear naming; magic numbers
- Mutable default arguments; hidden global state/side effects
- Fragile error handling (bare except, swallowed exceptions), missing context managers for resources
- Resource leaks (files/sockets/processes not closed), blocking calls in async code
- Inconsistent formatting, missing docstrings/type hints, unused imports/variables
- Security footguns (eval/exec, pickle on untrusted data, shell=True, SQL string concatenation)

3) Best practice violations
- PEP 8/257 style, import ordering, wildcard imports
- Type hints and dataclasses where appropriate; clear module boundaries
- Logging instead of print; appropriate log levels and configuration
- Using pathlib, subprocess.run, requests.Session, contextlib for context managers
- Configuration via environment/params instead of hard-coding; avoid secrets in code
- Deterministic randomness and secure primitives where needed (secrets, hmac.compare_digest)

If the code is lengthy, point me to the suspected hotspots or the specific functions/modules to prioritize.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”) rather than actual Python code. Please paste the code (or a link/gist) so I can run a specific review.

Helpful context to include:
- Python version and runtime (CPython/PyPy), OS
- Typical input sizes and performance goals (e.g., “process 1M rows under 5s”)
- Memory limits and third-party library constraints (e.g., can use numpy/pandas?)
- Whether public APIs/IO formats are fixed or can change

What I will examine once you provide the code:

1) Performance bottlenecks
- Algorithmic complexity (nested loops, O(n^2) patterns), unnecessary sorting or repeated work
- Inefficient data structures (list scans vs set/dict membership; using list where deque/heap is better)
- Hot-loop inefficiencies (per-iteration allocations, repeated attribute lookups, regex compilation, string concatenation with +)
- I/O and serialization overhead (sync file/network calls in tight loops, N+1 DB queries, lack of batching, no HTTP session reuse)
- Missed vectorization/broadcast (numpy/pandas), misuse of pandas apply/iterrows
- Missing caching/memoization for pure or expensive functions; redundant computations

2) Code quality issues
- Large, monolithic functions; duplication; unclear naming; magic numbers
- Mutable default arguments; hidden global state/side effects
- Fragile error handling (bare except, swallowed exceptions), missing context managers for resources
- Resource leaks (files/sockets/processes not closed), blocking calls in async code
- Inconsistent formatting, missing docstrings/type hints, unused imports/variables
- Security footguns (eval/exec, pickle on untrusted data, shell=True, SQL string concatenation)

3) Best practice violations
- PEP 8/257 style, import ordering, wildcard imports
- Type hints and dataclasses where appropriate; clear module boundaries
- Logging instead of print; appropriate log levels and configuration
- Using pathlib, subprocess.run, requests.Session, contextlib for context managers
- Configuration via environment/params instead of hard-coding; avoid secrets in code
- Deterministic randomness and secure primitives where needed (secrets, hmac.compare_digest)

If the code is lengthy, point me to the suspected hotspots or the specific functions/modules to prioritize.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”) rather than actual Python code. Please paste the code (or a link/gist) so I can run a specific review.

Helpful context to include:
- Python version and runtime (CPython/PyPy), OS
- Typical input sizes and performance goals (e.g., “process 1M rows under 5s”)
- Memory limits and third-party library constraints (e.g., can use numpy/pandas?)
- Whether public APIs/IO formats are fixed or can change

What I will examine once you provide the code:

1) Performance bottlenecks
- Algorithmic complexity (nested loops, O(n^2) patterns), unnecessary sorting or repeated work
- Inefficient data structures (list scans vs set/dict membership; using list where deque/heap is better)
- Hot-loop inefficiencies (per-iteration allocations, repeated attribute lookups, regex compilation, string concatenation with +)
- I/O and serialization overhead (sync file/network calls in tight loops, N+1 DB queries, lack of batching, no HTTP session reuse)
- Missed vectorization/broadcast (numpy/pandas), misuse of pandas apply/iterrows
- Missing caching/memoization for pure or expensive functions; redundant computations

2) Code quality issues
- Large, monolithic functions; duplication; unclear naming; magic numbers
- Mutable default arguments; hidden global state/side effects
- Fragile error handling (bare except, swallowed exceptions), missing context managers for resources
- Resource leaks (files/sockets/processes not closed), blocking calls in async code
- Inconsistent formatting, missing docstrings/type hints, unused imports/variables
- Security footguns (eval/exec, pickle on untrusted data, shell=True, SQL string concatenation)

3) Best practice violations
- PEP 8/257 style, import ordering, wildcard imports
- Type hints and dataclasses where appropriate; clear module boundaries
- Logging instead of print; appropriate log levels and configuration
- Using pathlib, subprocess.run, requests.Session, contextlib for context managers
- Configuration via environment/params instead of hard-coding; avoid secrets in code
- Deterministic randomness and secure primitives where needed (secrets, hmac.compare_digest)

If the code is lengthy, point me to the suspected hotspots or the specific functions/modules to prioritize.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”) rather than actual Python code. Please paste the code (or a link/gist) so I can run a specific review.

Helpful context to include:
- Python version and runtime (CPython/PyPy), OS
- Typical input sizes and performance goals (e.g., “process 1M rows under 5s”)
- Memory limits and third-party library constraints (e.g., can use numpy/pandas?)
- Whether public APIs/IO formats are fixed or can change

What I will examine once you provide the code:

1) Performance bottlenecks
- Algorithmic complexity (nested loops, O(n^2) patterns), unnecessary sorting or repeated work
- Inefficient data structures (list scans vs set/dict membership; using list where deque/heap is better)
- Hot-loop inefficiencies (per-iteration allocations, repeated attribute lookups, regex compilation, string concatenation with +)
- I/O and serialization overhead (sync file/network calls in tight loops, N+1 DB queries, lack of batching, no HTTP session reuse)
- Missed vectorization/broadcast (numpy/pandas), misuse of pandas apply/iterrows
- Missing caching/memoization for pure or expensive functions; redundant computations

2) Code quality issues
- Large, monolithic functions; duplication; unclear naming; magic numbers
- Mutable default arguments; hidden global state/side effects
- Fragile error handling (bare except, swallowed exceptions), missing context managers for resources
- Resource leaks (files/sockets/processes not closed), blocking calls in async code
- Inconsistent formatting, missing docstrings/type hints, unused imports/variables
- Security footguns (eval/exec, pickle on untrusted data, shell=True, SQL string concatenation)

3) Best practice violations
- PEP 8/257 style, import ordering, wildcard imports
- Type hints and dataclasses where appropriate; clear module boundaries
- Logging instead of print; appropriate log levels and configuration
- Using pathlib, subprocess.run, requests.Session, contextlib for context managers
- Configuration via environment/params instead of hard-coding; avoid secrets in code
- Deterministic randomness and secure primitives where needed (secrets, hmac.compare_digest)

If the code is lengthy, point me to the suspected hotspots or the specific functions/modules to prioritize.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”) rather than actual Python code. Please paste the code (or a link/gist) so I can run a specific review.

Helpful context to include:
- Python version and runtime (CPython/PyPy), OS
- Typical input sizes and performance goals (e.g., “process 1M rows under 5s”)
- Memory limits and third-party library constraints (e.g., can use numpy/pandas?)
- Whether public APIs/IO formats are fixed or can change

What I will examine once you provide the code:

1) Performance bottlenecks
- Algorithmic complexity (nested loops, O(n^2) patterns), unnecessary sorting or repeated work
- Inefficient data structures (list scans vs set/dict membership; using list where deque/heap is better)
- Hot-loop inefficiencies (per-iteration allocations, repeated attribute lookups, regex compilation, string concatenation with +)
- I/O and serialization overhead (sync file/network calls in tight loops, N+1 DB queries, lack of batching, no HTTP session reuse)
- Missed vectorization/broadcast (numpy/pandas), misuse of pandas apply/iterrows
- Missing caching/memoization for pure or expensive functions; redundant computations

2) Code quality issues
- Large, monolithic functions; duplication; unclear naming; magic numbers
- Mutable default arguments; hidden global state/side effects
- Fragile error handling (bare except, swallowed exceptions), missing context managers for resources
- Resource leaks (files/sockets/processes not closed), blocking calls in async code
- Inconsistent formatting, missing docstrings/type hints, unused imports/variables
- Security footguns (eval/exec, pickle on untrusted data, shell=True, SQL string concatenation)

3) Best practice violations
- PEP 8/257 style, import ordering, wildcard imports
- Type hints and dataclasses where appropriate; clear module boundaries
- Logging instead of print; appropriate log levels and configuration
- Using pathlib, subprocess.run, requests.Session, contextlib for context managers
- Configuration via environment/params instead of hard-coding; avoid secrets in code
- Deterministic randomness and secure primitives where needed (secrets, hmac.compare_digest)

If the code is lengthy, point me to the suspected hotspots or the specific functions/modules to prioritize.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”) rather than actual Python code. Please paste the code (or a link/gist) so I can run a specific review.

Helpful context to include:
- Python version and runtime (CPython/PyPy), OS
- Typical input sizes and performance goals (e.g., “process 1M rows under 5s”)
- Memory limits and third-party library constraints (e.g., can use numpy/pandas?)
- Whether public APIs/IO formats are fixed or can change

What I will examine once you provide the code:

1) Performance bottlenecks
- Algorithmic complexity (nested loops, O(n^2) patterns), unnecessary sorting or repeated work
- Inefficient data structures (list scans vs set/dict membership; using list where deque/heap is better)
- Hot-loop inefficiencies (per-iteration allocations, repeated attribute lookups, regex compilation, string concatenation with +)
- I/O and serialization overhead (sync file/network calls in tight loops, N+1 DB queries, lack of batching, no HTTP session reuse)
- Missed vectorization/broadcast (numpy/pandas), misuse of pandas apply/iterrows
- Missing caching/memoization for pure or expensive functions; redundant computations

2) Code quality issues
- Large, monolithic functions; duplication; unclear naming; magic numbers
- Mutable default arguments; hidden global state/side effects
- Fragile error handling (bare except, swallowed exceptions), missing context managers for resources
- Resource leaks (files/sockets/processes not closed), blocking calls in async code
- Inconsistent formatting, missing docstrings/type hints, unused imports/variables
- Security footguns (eval/exec, pickle on untrusted data, shell=True, SQL string concatenation)

3) Best practice violations
- PEP 8/257 style, import ordering, wildcard imports
- Type hints and dataclasses where appropriate; clear module boundaries
- Logging instead of print; appropriate log levels and configuration
- Using pathlib, subprocess.run, requests.Session, contextlib for context managers
- Configuration via environment/params instead of hard-coding; avoid secrets in code
- Deterministic randomness and secure primitives where needed (secrets, hmac.compare_digest)

If the code is lengthy, point me to the suspected hotspots or the specific functions/modules to prioritize.

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