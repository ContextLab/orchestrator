# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:09:18

## Analysis Results

I don’t see any Python code in your message (it shows # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:09:18

## Analysis Results

I don’t see any Python code in your message (it shows # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:09:18

## Analysis Results

I don’t see any Python code in your message (it shows # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:09:18

## Analysis Results

I don’t see any Python code in your message (it shows # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:09:18

## Analysis Results

I don’t see any Python code in your message (it shows # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:09:18

## Analysis Results

I don’t see any Python code in your message (it shows # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:09:18

## Analysis Results

I don’t see any Python code in your message (it shows # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:09:18

## Analysis Results

I don’t see any Python code in your message (it shows # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:09:18

## Analysis Results

I don’t see any Python code in your message (it shows # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:09:18

## Analysis Results

I don’t see any Python code in your message (it shows # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:09:18

## Analysis Results

I don’t see any Python code in your message (it shows # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:09:18

## Analysis Results

I don’t see any Python code in your message (it shows {{content}}). Please paste the code you want reviewed. If it’s long, you can share the critical parts plus a brief description of what it does and where it feels slow.

While you gather that, here’s a concise checklist I’ll use to analyze it:

Performance bottlenecks to look for
- Algorithmic complexity: unnecessary O(n^2) loops; repeated scans/sorts; nested loops over large data.
- Inefficient data structures: list membership checks instead of set; list for counting instead of dict/Counter; linear search vs bisect/heapq.
- Hot-path work inside loops: repeated regex compilation, datetime parsing, JSON loads, DB queries, disk/network I/O.
- Unnecessary materialization: building large intermediate lists instead of generators; copying large objects; deep chaining in pandas that creates many temporaries.
- String handling: repeated concatenation in loops instead of ''.join; excessive formatting.
- Pandas/NumPy: row-wise apply/itertuples/iterrows instead of vectorized ops; Python loops over arrays; not using categorical dtypes; avoid inplace pitfalls that copy anyway.
- Caching: recomputing pure functions; missing memoization/LRU cache.
- Concurrency/parallelism: doing I/O-bound work sequentially (consider asyncio/threads); CPU-bound work in Python without vectorization or multiprocessing; GIL contention.
- I/O patterns: many small file/DB calls; not using batching/bulk operations; N+1 queries; missing indices.
- Repeated setup: opening/closing sessions/clients in loops; not reusing compiled regex, DB connections, requests.Session.

Code quality issues to look for
- Readability: long functions, deep nesting, duplication (DRY), unclear names, magic numbers.
- Structure: lack of separation of concerns; side effects at import time; missing if __name__ == "__main__".
- Error handling: bare except, broad catches, swallowed exceptions, no retries/backoff for transient I/O.
- Types and docs: missing type hints, docstrings, unclear interfaces; high cyclomatic complexity.
- Resource management: not using context managers for files/sockets/locks; leaking processes/threads.
- Testing: no unit tests; hard-to-test code due to tight coupling or hidden globals.
- Configuration: hardcoded paths/secrets; configuration scattered instead of centralized.

Best practice violations to check
- Mutable default arguments; shadowing builtins; reliance on global state.
- Security: constructing SQL/commands with string formatting instead of parameters; storing secrets in code; not validating/escaping inputs; disabling TLS verification.
- Packaging/versions: unpinned dependencies; mixing runtime and dev dependencies; no lockfile.
- Logging: using prints instead of logging with levels; missing context; excessive logging in hot paths.
- Performance hygiene: not profiling (cProfile, py-spy, scalene); no benchmarking; premature micro-optimizations.
- Pandas/NumPy specifics: chained assignments, inplace misuse, hidden copies; mixed dtypes; relying on object dtype for numbers.

Helpful context to include with your code
- Python version and key libraries; data sizes and typical workload.
- What’s slow (functions, inputs), current timings/memory, and target goals.
- Environment (local, serverless, container) and constraints (CPU, RAM).

Once you share the code, I’ll provide concrete, line-level suggestions for:
- Specific bottlenecks with proposed fixes and expected complexity improvements.
- Refactors to improve readability and maintainability.
- Best-practice corrections and safer patterns.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions). Please paste the code you want reviewed. If it’s long, you can share the critical parts plus a brief description of what it does and where it feels slow.

While you gather that, here’s a concise checklist I’ll use to analyze it:

Performance bottlenecks to look for
- Algorithmic complexity: unnecessary O(n^2) loops; repeated scans/sorts; nested loops over large data.
- Inefficient data structures: list membership checks instead of set; list for counting instead of dict/Counter; linear search vs bisect/heapq.
- Hot-path work inside loops: repeated regex compilation, datetime parsing, JSON loads, DB queries, disk/network I/O.
- Unnecessary materialization: building large intermediate lists instead of generators; copying large objects; deep chaining in pandas that creates many temporaries.
- String handling: repeated concatenation in loops instead of ''.join; excessive formatting.
- Pandas/NumPy: row-wise apply/itertuples/iterrows instead of vectorized ops; Python loops over arrays; not using categorical dtypes; avoid inplace pitfalls that copy anyway.
- Caching: recomputing pure functions; missing memoization/LRU cache.
- Concurrency/parallelism: doing I/O-bound work sequentially (consider asyncio/threads); CPU-bound work in Python without vectorization or multiprocessing; GIL contention.
- I/O patterns: many small file/DB calls; not using batching/bulk operations; N+1 queries; missing indices.
- Repeated setup: opening/closing sessions/clients in loops; not reusing compiled regex, DB connections, requests.Session.

Code quality issues to look for
- Readability: long functions, deep nesting, duplication (DRY), unclear names, magic numbers.
- Structure: lack of separation of concerns; side effects at import time; missing if __name__ == "__main__".
- Error handling: bare except, broad catches, swallowed exceptions, no retries/backoff for transient I/O.
- Types and docs: missing type hints, docstrings, unclear interfaces; high cyclomatic complexity.
- Resource management: not using context managers for files/sockets/locks; leaking processes/threads.
- Testing: no unit tests; hard-to-test code due to tight coupling or hidden globals.
- Configuration: hardcoded paths/secrets; configuration scattered instead of centralized.

Best practice violations to check
- Mutable default arguments; shadowing builtins; reliance on global state.
- Security: constructing SQL/commands with string formatting instead of parameters; storing secrets in code; not validating/escaping inputs; disabling TLS verification.
- Packaging/versions: unpinned dependencies; mixing runtime and dev dependencies; no lockfile.
- Logging: using prints instead of logging with levels; missing context; excessive logging in hot paths.
- Performance hygiene: not profiling (cProfile, py-spy, scalene); no benchmarking; premature micro-optimizations.
- Pandas/NumPy specifics: chained assignments, inplace misuse, hidden copies; mixed dtypes; relying on object dtype for numbers.

Helpful context to include with your code
- Python version and key libraries; data sizes and typical workload.
- What’s slow (functions, inputs), current timings/memory, and target goals.
- Environment (local, serverless, container) and constraints (CPU, RAM).

Once you share the code, I’ll provide concrete, line-level suggestions for:
- Specific bottlenecks with proposed fixes and expected complexity improvements.
- Refactors to improve readability and maintainability.
- Best-practice corrections and safer patterns.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions). Please paste the code you want reviewed. If it’s long, you can share the critical parts plus a brief description of what it does and where it feels slow.

While you gather that, here’s a concise checklist I’ll use to analyze it:

Performance bottlenecks to look for
- Algorithmic complexity: unnecessary O(n^2) loops; repeated scans/sorts; nested loops over large data.
- Inefficient data structures: list membership checks instead of set; list for counting instead of dict/Counter; linear search vs bisect/heapq.
- Hot-path work inside loops: repeated regex compilation, datetime parsing, JSON loads, DB queries, disk/network I/O.
- Unnecessary materialization: building large intermediate lists instead of generators; copying large objects; deep chaining in pandas that creates many temporaries.
- String handling: repeated concatenation in loops instead of ''.join; excessive formatting.
- Pandas/NumPy: row-wise apply/itertuples/iterrows instead of vectorized ops; Python loops over arrays; not using categorical dtypes; avoid inplace pitfalls that copy anyway.
- Caching: recomputing pure functions; missing memoization/LRU cache.
- Concurrency/parallelism: doing I/O-bound work sequentially (consider asyncio/threads); CPU-bound work in Python without vectorization or multiprocessing; GIL contention.
- I/O patterns: many small file/DB calls; not using batching/bulk operations; N+1 queries; missing indices.
- Repeated setup: opening/closing sessions/clients in loops; not reusing compiled regex, DB connections, requests.Session.

Code quality issues to look for
- Readability: long functions, deep nesting, duplication (DRY), unclear names, magic numbers.
- Structure: lack of separation of concerns; side effects at import time; missing if __name__ == "__main__".
- Error handling: bare except, broad catches, swallowed exceptions, no retries/backoff for transient I/O.
- Types and docs: missing type hints, docstrings, unclear interfaces; high cyclomatic complexity.
- Resource management: not using context managers for files/sockets/locks; leaking processes/threads.
- Testing: no unit tests; hard-to-test code due to tight coupling or hidden globals.
- Configuration: hardcoded paths/secrets; configuration scattered instead of centralized.

Best practice violations to check
- Mutable default arguments; shadowing builtins; reliance on global state.
- Security: constructing SQL/commands with string formatting instead of parameters; storing secrets in code; not validating/escaping inputs; disabling TLS verification.
- Packaging/versions: unpinned dependencies; mixing runtime and dev dependencies; no lockfile.
- Logging: using prints instead of logging with levels; missing context; excessive logging in hot paths.
- Performance hygiene: not profiling (cProfile, py-spy, scalene); no benchmarking; premature micro-optimizations.
- Pandas/NumPy specifics: chained assignments, inplace misuse, hidden copies; mixed dtypes; relying on object dtype for numbers.

Helpful context to include with your code
- Python version and key libraries; data sizes and typical workload.
- What’s slow (functions, inputs), current timings/memory, and target goals.
- Environment (local, serverless, container) and constraints (CPU, RAM).

Once you share the code, I’ll provide concrete, line-level suggestions for:
- Specific bottlenecks with proposed fixes and expected complexity improvements.
- Refactors to improve readability and maintainability.
- Best-practice corrections and safer patterns.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions). Please paste the code you want reviewed. If it’s long, you can share the critical parts plus a brief description of what it does and where it feels slow.

While you gather that, here’s a concise checklist I’ll use to analyze it:

Performance bottlenecks to look for
- Algorithmic complexity: unnecessary O(n^2) loops; repeated scans/sorts; nested loops over large data.
- Inefficient data structures: list membership checks instead of set; list for counting instead of dict/Counter; linear search vs bisect/heapq.
- Hot-path work inside loops: repeated regex compilation, datetime parsing, JSON loads, DB queries, disk/network I/O.
- Unnecessary materialization: building large intermediate lists instead of generators; copying large objects; deep chaining in pandas that creates many temporaries.
- String handling: repeated concatenation in loops instead of ''.join; excessive formatting.
- Pandas/NumPy: row-wise apply/itertuples/iterrows instead of vectorized ops; Python loops over arrays; not using categorical dtypes; avoid inplace pitfalls that copy anyway.
- Caching: recomputing pure functions; missing memoization/LRU cache.
- Concurrency/parallelism: doing I/O-bound work sequentially (consider asyncio/threads); CPU-bound work in Python without vectorization or multiprocessing; GIL contention.
- I/O patterns: many small file/DB calls; not using batching/bulk operations; N+1 queries; missing indices.
- Repeated setup: opening/closing sessions/clients in loops; not reusing compiled regex, DB connections, requests.Session.

Code quality issues to look for
- Readability: long functions, deep nesting, duplication (DRY), unclear names, magic numbers.
- Structure: lack of separation of concerns; side effects at import time; missing if __name__ == "__main__".
- Error handling: bare except, broad catches, swallowed exceptions, no retries/backoff for transient I/O.
- Types and docs: missing type hints, docstrings, unclear interfaces; high cyclomatic complexity.
- Resource management: not using context managers for files/sockets/locks; leaking processes/threads.
- Testing: no unit tests; hard-to-test code due to tight coupling or hidden globals.
- Configuration: hardcoded paths/secrets; configuration scattered instead of centralized.

Best practice violations to check
- Mutable default arguments; shadowing builtins; reliance on global state.
- Security: constructing SQL/commands with string formatting instead of parameters; storing secrets in code; not validating/escaping inputs; disabling TLS verification.
- Packaging/versions: unpinned dependencies; mixing runtime and dev dependencies; no lockfile.
- Logging: using prints instead of logging with levels; missing context; excessive logging in hot paths.
- Performance hygiene: not profiling (cProfile, py-spy, scalene); no benchmarking; premature micro-optimizations.
- Pandas/NumPy specifics: chained assignments, inplace misuse, hidden copies; mixed dtypes; relying on object dtype for numbers.

Helpful context to include with your code
- Python version and key libraries; data sizes and typical workload.
- What’s slow (functions, inputs), current timings/memory, and target goals.
- Environment (local, serverless, container) and constraints (CPU, RAM).

Once you share the code, I’ll provide concrete, line-level suggestions for:
- Specific bottlenecks with proposed fixes and expected complexity improvements.
- Refactors to improve readability and maintainability.
- Best-practice corrections and safer patterns.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions). Please paste the code you want reviewed. If it’s long, you can share the critical parts plus a brief description of what it does and where it feels slow.

While you gather that, here’s a concise checklist I’ll use to analyze it:

Performance bottlenecks to look for
- Algorithmic complexity: unnecessary O(n^2) loops; repeated scans/sorts; nested loops over large data.
- Inefficient data structures: list membership checks instead of set; list for counting instead of dict/Counter; linear search vs bisect/heapq.
- Hot-path work inside loops: repeated regex compilation, datetime parsing, JSON loads, DB queries, disk/network I/O.
- Unnecessary materialization: building large intermediate lists instead of generators; copying large objects; deep chaining in pandas that creates many temporaries.
- String handling: repeated concatenation in loops instead of ''.join; excessive formatting.
- Pandas/NumPy: row-wise apply/itertuples/iterrows instead of vectorized ops; Python loops over arrays; not using categorical dtypes; avoid inplace pitfalls that copy anyway.
- Caching: recomputing pure functions; missing memoization/LRU cache.
- Concurrency/parallelism: doing I/O-bound work sequentially (consider asyncio/threads); CPU-bound work in Python without vectorization or multiprocessing; GIL contention.
- I/O patterns: many small file/DB calls; not using batching/bulk operations; N+1 queries; missing indices.
- Repeated setup: opening/closing sessions/clients in loops; not reusing compiled regex, DB connections, requests.Session.

Code quality issues to look for
- Readability: long functions, deep nesting, duplication (DRY), unclear names, magic numbers.
- Structure: lack of separation of concerns; side effects at import time; missing if __name__ == "__main__".
- Error handling: bare except, broad catches, swallowed exceptions, no retries/backoff for transient I/O.
- Types and docs: missing type hints, docstrings, unclear interfaces; high cyclomatic complexity.
- Resource management: not using context managers for files/sockets/locks; leaking processes/threads.
- Testing: no unit tests; hard-to-test code due to tight coupling or hidden globals.
- Configuration: hardcoded paths/secrets; configuration scattered instead of centralized.

Best practice violations to check
- Mutable default arguments; shadowing builtins; reliance on global state.
- Security: constructing SQL/commands with string formatting instead of parameters; storing secrets in code; not validating/escaping inputs; disabling TLS verification.
- Packaging/versions: unpinned dependencies; mixing runtime and dev dependencies; no lockfile.
- Logging: using prints instead of logging with levels; missing context; excessive logging in hot paths.
- Performance hygiene: not profiling (cProfile, py-spy, scalene); no benchmarking; premature micro-optimizations.
- Pandas/NumPy specifics: chained assignments, inplace misuse, hidden copies; mixed dtypes; relying on object dtype for numbers.

Helpful context to include with your code
- Python version and key libraries; data sizes and typical workload.
- What’s slow (functions, inputs), current timings/memory, and target goals.
- Environment (local, serverless, container) and constraints (CPU, RAM).

Once you share the code, I’ll provide concrete, line-level suggestions for:
- Specific bottlenecks with proposed fixes and expected complexity improvements.
- Refactors to improve readability and maintainability.
- Best-practice corrections and safer patterns.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions). Please paste the code you want reviewed. If it’s long, you can share the critical parts plus a brief description of what it does and where it feels slow.

While you gather that, here’s a concise checklist I’ll use to analyze it:

Performance bottlenecks to look for
- Algorithmic complexity: unnecessary O(n^2) loops; repeated scans/sorts; nested loops over large data.
- Inefficient data structures: list membership checks instead of set; list for counting instead of dict/Counter; linear search vs bisect/heapq.
- Hot-path work inside loops: repeated regex compilation, datetime parsing, JSON loads, DB queries, disk/network I/O.
- Unnecessary materialization: building large intermediate lists instead of generators; copying large objects; deep chaining in pandas that creates many temporaries.
- String handling: repeated concatenation in loops instead of ''.join; excessive formatting.
- Pandas/NumPy: row-wise apply/itertuples/iterrows instead of vectorized ops; Python loops over arrays; not using categorical dtypes; avoid inplace pitfalls that copy anyway.
- Caching: recomputing pure functions; missing memoization/LRU cache.
- Concurrency/parallelism: doing I/O-bound work sequentially (consider asyncio/threads); CPU-bound work in Python without vectorization or multiprocessing; GIL contention.
- I/O patterns: many small file/DB calls; not using batching/bulk operations; N+1 queries; missing indices.
- Repeated setup: opening/closing sessions/clients in loops; not reusing compiled regex, DB connections, requests.Session.

Code quality issues to look for
- Readability: long functions, deep nesting, duplication (DRY), unclear names, magic numbers.
- Structure: lack of separation of concerns; side effects at import time; missing if __name__ == "__main__".
- Error handling: bare except, broad catches, swallowed exceptions, no retries/backoff for transient I/O.
- Types and docs: missing type hints, docstrings, unclear interfaces; high cyclomatic complexity.
- Resource management: not using context managers for files/sockets/locks; leaking processes/threads.
- Testing: no unit tests; hard-to-test code due to tight coupling or hidden globals.
- Configuration: hardcoded paths/secrets; configuration scattered instead of centralized.

Best practice violations to check
- Mutable default arguments; shadowing builtins; reliance on global state.
- Security: constructing SQL/commands with string formatting instead of parameters; storing secrets in code; not validating/escaping inputs; disabling TLS verification.
- Packaging/versions: unpinned dependencies; mixing runtime and dev dependencies; no lockfile.
- Logging: using prints instead of logging with levels; missing context; excessive logging in hot paths.
- Performance hygiene: not profiling (cProfile, py-spy, scalene); no benchmarking; premature micro-optimizations.
- Pandas/NumPy specifics: chained assignments, inplace misuse, hidden copies; mixed dtypes; relying on object dtype for numbers.

Helpful context to include with your code
- Python version and key libraries; data sizes and typical workload.
- What’s slow (functions, inputs), current timings/memory, and target goals.
- Environment (local, serverless, container) and constraints (CPU, RAM).

Once you share the code, I’ll provide concrete, line-level suggestions for:
- Specific bottlenecks with proposed fixes and expected complexity improvements.
- Refactors to improve readability and maintainability.
- Best-practice corrections and safer patterns.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions). Please paste the code you want reviewed. If it’s long, you can share the critical parts plus a brief description of what it does and where it feels slow.

While you gather that, here’s a concise checklist I’ll use to analyze it:

Performance bottlenecks to look for
- Algorithmic complexity: unnecessary O(n^2) loops; repeated scans/sorts; nested loops over large data.
- Inefficient data structures: list membership checks instead of set; list for counting instead of dict/Counter; linear search vs bisect/heapq.
- Hot-path work inside loops: repeated regex compilation, datetime parsing, JSON loads, DB queries, disk/network I/O.
- Unnecessary materialization: building large intermediate lists instead of generators; copying large objects; deep chaining in pandas that creates many temporaries.
- String handling: repeated concatenation in loops instead of ''.join; excessive formatting.
- Pandas/NumPy: row-wise apply/itertuples/iterrows instead of vectorized ops; Python loops over arrays; not using categorical dtypes; avoid inplace pitfalls that copy anyway.
- Caching: recomputing pure functions; missing memoization/LRU cache.
- Concurrency/parallelism: doing I/O-bound work sequentially (consider asyncio/threads); CPU-bound work in Python without vectorization or multiprocessing; GIL contention.
- I/O patterns: many small file/DB calls; not using batching/bulk operations; N+1 queries; missing indices.
- Repeated setup: opening/closing sessions/clients in loops; not reusing compiled regex, DB connections, requests.Session.

Code quality issues to look for
- Readability: long functions, deep nesting, duplication (DRY), unclear names, magic numbers.
- Structure: lack of separation of concerns; side effects at import time; missing if __name__ == "__main__".
- Error handling: bare except, broad catches, swallowed exceptions, no retries/backoff for transient I/O.
- Types and docs: missing type hints, docstrings, unclear interfaces; high cyclomatic complexity.
- Resource management: not using context managers for files/sockets/locks; leaking processes/threads.
- Testing: no unit tests; hard-to-test code due to tight coupling or hidden globals.
- Configuration: hardcoded paths/secrets; configuration scattered instead of centralized.

Best practice violations to check
- Mutable default arguments; shadowing builtins; reliance on global state.
- Security: constructing SQL/commands with string formatting instead of parameters; storing secrets in code; not validating/escaping inputs; disabling TLS verification.
- Packaging/versions: unpinned dependencies; mixing runtime and dev dependencies; no lockfile.
- Logging: using prints instead of logging with levels; missing context; excessive logging in hot paths.
- Performance hygiene: not profiling (cProfile, py-spy, scalene); no benchmarking; premature micro-optimizations.
- Pandas/NumPy specifics: chained assignments, inplace misuse, hidden copies; mixed dtypes; relying on object dtype for numbers.

Helpful context to include with your code
- Python version and key libraries; data sizes and typical workload.
- What’s slow (functions, inputs), current timings/memory, and target goals.
- Environment (local, serverless, container) and constraints (CPU, RAM).

Once you share the code, I’ll provide concrete, line-level suggestions for:
- Specific bottlenecks with proposed fixes and expected complexity improvements.
- Refactors to improve readability and maintainability.
- Best-practice corrections and safer patterns.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions). Please paste the code you want reviewed. If it’s long, you can share the critical parts plus a brief description of what it does and where it feels slow.

While you gather that, here’s a concise checklist I’ll use to analyze it:

Performance bottlenecks to look for
- Algorithmic complexity: unnecessary O(n^2) loops; repeated scans/sorts; nested loops over large data.
- Inefficient data structures: list membership checks instead of set; list for counting instead of dict/Counter; linear search vs bisect/heapq.
- Hot-path work inside loops: repeated regex compilation, datetime parsing, JSON loads, DB queries, disk/network I/O.
- Unnecessary materialization: building large intermediate lists instead of generators; copying large objects; deep chaining in pandas that creates many temporaries.
- String handling: repeated concatenation in loops instead of ''.join; excessive formatting.
- Pandas/NumPy: row-wise apply/itertuples/iterrows instead of vectorized ops; Python loops over arrays; not using categorical dtypes; avoid inplace pitfalls that copy anyway.
- Caching: recomputing pure functions; missing memoization/LRU cache.
- Concurrency/parallelism: doing I/O-bound work sequentially (consider asyncio/threads); CPU-bound work in Python without vectorization or multiprocessing; GIL contention.
- I/O patterns: many small file/DB calls; not using batching/bulk operations; N+1 queries; missing indices.
- Repeated setup: opening/closing sessions/clients in loops; not reusing compiled regex, DB connections, requests.Session.

Code quality issues to look for
- Readability: long functions, deep nesting, duplication (DRY), unclear names, magic numbers.
- Structure: lack of separation of concerns; side effects at import time; missing if __name__ == "__main__".
- Error handling: bare except, broad catches, swallowed exceptions, no retries/backoff for transient I/O.
- Types and docs: missing type hints, docstrings, unclear interfaces; high cyclomatic complexity.
- Resource management: not using context managers for files/sockets/locks; leaking processes/threads.
- Testing: no unit tests; hard-to-test code due to tight coupling or hidden globals.
- Configuration: hardcoded paths/secrets; configuration scattered instead of centralized.

Best practice violations to check
- Mutable default arguments; shadowing builtins; reliance on global state.
- Security: constructing SQL/commands with string formatting instead of parameters; storing secrets in code; not validating/escaping inputs; disabling TLS verification.
- Packaging/versions: unpinned dependencies; mixing runtime and dev dependencies; no lockfile.
- Logging: using prints instead of logging with levels; missing context; excessive logging in hot paths.
- Performance hygiene: not profiling (cProfile, py-spy, scalene); no benchmarking; premature micro-optimizations.
- Pandas/NumPy specifics: chained assignments, inplace misuse, hidden copies; mixed dtypes; relying on object dtype for numbers.

Helpful context to include with your code
- Python version and key libraries; data sizes and typical workload.
- What’s slow (functions, inputs), current timings/memory, and target goals.
- Environment (local, serverless, container) and constraints (CPU, RAM).

Once you share the code, I’ll provide concrete, line-level suggestions for:
- Specific bottlenecks with proposed fixes and expected complexity improvements.
- Refactors to improve readability and maintainability.
- Best-practice corrections and safer patterns.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions). Please paste the code you want reviewed. If it’s long, you can share the critical parts plus a brief description of what it does and where it feels slow.

While you gather that, here’s a concise checklist I’ll use to analyze it:

Performance bottlenecks to look for
- Algorithmic complexity: unnecessary O(n^2) loops; repeated scans/sorts; nested loops over large data.
- Inefficient data structures: list membership checks instead of set; list for counting instead of dict/Counter; linear search vs bisect/heapq.
- Hot-path work inside loops: repeated regex compilation, datetime parsing, JSON loads, DB queries, disk/network I/O.
- Unnecessary materialization: building large intermediate lists instead of generators; copying large objects; deep chaining in pandas that creates many temporaries.
- String handling: repeated concatenation in loops instead of ''.join; excessive formatting.
- Pandas/NumPy: row-wise apply/itertuples/iterrows instead of vectorized ops; Python loops over arrays; not using categorical dtypes; avoid inplace pitfalls that copy anyway.
- Caching: recomputing pure functions; missing memoization/LRU cache.
- Concurrency/parallelism: doing I/O-bound work sequentially (consider asyncio/threads); CPU-bound work in Python without vectorization or multiprocessing; GIL contention.
- I/O patterns: many small file/DB calls; not using batching/bulk operations; N+1 queries; missing indices.
- Repeated setup: opening/closing sessions/clients in loops; not reusing compiled regex, DB connections, requests.Session.

Code quality issues to look for
- Readability: long functions, deep nesting, duplication (DRY), unclear names, magic numbers.
- Structure: lack of separation of concerns; side effects at import time; missing if __name__ == "__main__".
- Error handling: bare except, broad catches, swallowed exceptions, no retries/backoff for transient I/O.
- Types and docs: missing type hints, docstrings, unclear interfaces; high cyclomatic complexity.
- Resource management: not using context managers for files/sockets/locks; leaking processes/threads.
- Testing: no unit tests; hard-to-test code due to tight coupling or hidden globals.
- Configuration: hardcoded paths/secrets; configuration scattered instead of centralized.

Best practice violations to check
- Mutable default arguments; shadowing builtins; reliance on global state.
- Security: constructing SQL/commands with string formatting instead of parameters; storing secrets in code; not validating/escaping inputs; disabling TLS verification.
- Packaging/versions: unpinned dependencies; mixing runtime and dev dependencies; no lockfile.
- Logging: using prints instead of logging with levels; missing context; excessive logging in hot paths.
- Performance hygiene: not profiling (cProfile, py-spy, scalene); no benchmarking; premature micro-optimizations.
- Pandas/NumPy specifics: chained assignments, inplace misuse, hidden copies; mixed dtypes; relying on object dtype for numbers.

Helpful context to include with your code
- Python version and key libraries; data sizes and typical workload.
- What’s slow (functions, inputs), current timings/memory, and target goals.
- Environment (local, serverless, container) and constraints (CPU, RAM).

Once you share the code, I’ll provide concrete, line-level suggestions for:
- Specific bottlenecks with proposed fixes and expected complexity improvements.
- Refactors to improve readability and maintainability.
- Best-practice corrections and safer patterns.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions). Please paste the code you want reviewed. If it’s long, you can share the critical parts plus a brief description of what it does and where it feels slow.

While you gather that, here’s a concise checklist I’ll use to analyze it:

Performance bottlenecks to look for
- Algorithmic complexity: unnecessary O(n^2) loops; repeated scans/sorts; nested loops over large data.
- Inefficient data structures: list membership checks instead of set; list for counting instead of dict/Counter; linear search vs bisect/heapq.
- Hot-path work inside loops: repeated regex compilation, datetime parsing, JSON loads, DB queries, disk/network I/O.
- Unnecessary materialization: building large intermediate lists instead of generators; copying large objects; deep chaining in pandas that creates many temporaries.
- String handling: repeated concatenation in loops instead of ''.join; excessive formatting.
- Pandas/NumPy: row-wise apply/itertuples/iterrows instead of vectorized ops; Python loops over arrays; not using categorical dtypes; avoid inplace pitfalls that copy anyway.
- Caching: recomputing pure functions; missing memoization/LRU cache.
- Concurrency/parallelism: doing I/O-bound work sequentially (consider asyncio/threads); CPU-bound work in Python without vectorization or multiprocessing; GIL contention.
- I/O patterns: many small file/DB calls; not using batching/bulk operations; N+1 queries; missing indices.
- Repeated setup: opening/closing sessions/clients in loops; not reusing compiled regex, DB connections, requests.Session.

Code quality issues to look for
- Readability: long functions, deep nesting, duplication (DRY), unclear names, magic numbers.
- Structure: lack of separation of concerns; side effects at import time; missing if __name__ == "__main__".
- Error handling: bare except, broad catches, swallowed exceptions, no retries/backoff for transient I/O.
- Types and docs: missing type hints, docstrings, unclear interfaces; high cyclomatic complexity.
- Resource management: not using context managers for files/sockets/locks; leaking processes/threads.
- Testing: no unit tests; hard-to-test code due to tight coupling or hidden globals.
- Configuration: hardcoded paths/secrets; configuration scattered instead of centralized.

Best practice violations to check
- Mutable default arguments; shadowing builtins; reliance on global state.
- Security: constructing SQL/commands with string formatting instead of parameters; storing secrets in code; not validating/escaping inputs; disabling TLS verification.
- Packaging/versions: unpinned dependencies; mixing runtime and dev dependencies; no lockfile.
- Logging: using prints instead of logging with levels; missing context; excessive logging in hot paths.
- Performance hygiene: not profiling (cProfile, py-spy, scalene); no benchmarking; premature micro-optimizations.
- Pandas/NumPy specifics: chained assignments, inplace misuse, hidden copies; mixed dtypes; relying on object dtype for numbers.

Helpful context to include with your code
- Python version and key libraries; data sizes and typical workload.
- What’s slow (functions, inputs), current timings/memory, and target goals.
- Environment (local, serverless, container) and constraints (CPU, RAM).

Once you share the code, I’ll provide concrete, line-level suggestions for:
- Specific bottlenecks with proposed fixes and expected complexity improvements.
- Refactors to improve readability and maintainability.
- Best-practice corrections and safer patterns.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions). Please paste the code you want reviewed. If it’s long, you can share the critical parts plus a brief description of what it does and where it feels slow.

While you gather that, here’s a concise checklist I’ll use to analyze it:

Performance bottlenecks to look for
- Algorithmic complexity: unnecessary O(n^2) loops; repeated scans/sorts; nested loops over large data.
- Inefficient data structures: list membership checks instead of set; list for counting instead of dict/Counter; linear search vs bisect/heapq.
- Hot-path work inside loops: repeated regex compilation, datetime parsing, JSON loads, DB queries, disk/network I/O.
- Unnecessary materialization: building large intermediate lists instead of generators; copying large objects; deep chaining in pandas that creates many temporaries.
- String handling: repeated concatenation in loops instead of ''.join; excessive formatting.
- Pandas/NumPy: row-wise apply/itertuples/iterrows instead of vectorized ops; Python loops over arrays; not using categorical dtypes; avoid inplace pitfalls that copy anyway.
- Caching: recomputing pure functions; missing memoization/LRU cache.
- Concurrency/parallelism: doing I/O-bound work sequentially (consider asyncio/threads); CPU-bound work in Python without vectorization or multiprocessing; GIL contention.
- I/O patterns: many small file/DB calls; not using batching/bulk operations; N+1 queries; missing indices.
- Repeated setup: opening/closing sessions/clients in loops; not reusing compiled regex, DB connections, requests.Session.

Code quality issues to look for
- Readability: long functions, deep nesting, duplication (DRY), unclear names, magic numbers.
- Structure: lack of separation of concerns; side effects at import time; missing if __name__ == "__main__".
- Error handling: bare except, broad catches, swallowed exceptions, no retries/backoff for transient I/O.
- Types and docs: missing type hints, docstrings, unclear interfaces; high cyclomatic complexity.
- Resource management: not using context managers for files/sockets/locks; leaking processes/threads.
- Testing: no unit tests; hard-to-test code due to tight coupling or hidden globals.
- Configuration: hardcoded paths/secrets; configuration scattered instead of centralized.

Best practice violations to check
- Mutable default arguments; shadowing builtins; reliance on global state.
- Security: constructing SQL/commands with string formatting instead of parameters; storing secrets in code; not validating/escaping inputs; disabling TLS verification.
- Packaging/versions: unpinned dependencies; mixing runtime and dev dependencies; no lockfile.
- Logging: using prints instead of logging with levels; missing context; excessive logging in hot paths.
- Performance hygiene: not profiling (cProfile, py-spy, scalene); no benchmarking; premature micro-optimizations.
- Pandas/NumPy specifics: chained assignments, inplace misuse, hidden copies; mixed dtypes; relying on object dtype for numbers.

Helpful context to include with your code
- Python version and key libraries; data sizes and typical workload.
- What’s slow (functions, inputs), current timings/memory, and target goals.
- Environment (local, serverless, container) and constraints (CPU, RAM).

Once you share the code, I’ll provide concrete, line-level suggestions for:
- Specific bottlenecks with proposed fixes and expected complexity improvements.
- Refactors to improve readability and maintainability.
- Best-practice corrections and safer patterns.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions). Please paste the code you want reviewed. If it’s long, you can share the critical parts plus a brief description of what it does and where it feels slow.

While you gather that, here’s a concise checklist I’ll use to analyze it:

Performance bottlenecks to look for
- Algorithmic complexity: unnecessary O(n^2) loops; repeated scans/sorts; nested loops over large data.
- Inefficient data structures: list membership checks instead of set; list for counting instead of dict/Counter; linear search vs bisect/heapq.
- Hot-path work inside loops: repeated regex compilation, datetime parsing, JSON loads, DB queries, disk/network I/O.
- Unnecessary materialization: building large intermediate lists instead of generators; copying large objects; deep chaining in pandas that creates many temporaries.
- String handling: repeated concatenation in loops instead of ''.join; excessive formatting.
- Pandas/NumPy: row-wise apply/itertuples/iterrows instead of vectorized ops; Python loops over arrays; not using categorical dtypes; avoid inplace pitfalls that copy anyway.
- Caching: recomputing pure functions; missing memoization/LRU cache.
- Concurrency/parallelism: doing I/O-bound work sequentially (consider asyncio/threads); CPU-bound work in Python without vectorization or multiprocessing; GIL contention.
- I/O patterns: many small file/DB calls; not using batching/bulk operations; N+1 queries; missing indices.
- Repeated setup: opening/closing sessions/clients in loops; not reusing compiled regex, DB connections, requests.Session.

Code quality issues to look for
- Readability: long functions, deep nesting, duplication (DRY), unclear names, magic numbers.
- Structure: lack of separation of concerns; side effects at import time; missing if __name__ == "__main__".
- Error handling: bare except, broad catches, swallowed exceptions, no retries/backoff for transient I/O.
- Types and docs: missing type hints, docstrings, unclear interfaces; high cyclomatic complexity.
- Resource management: not using context managers for files/sockets/locks; leaking processes/threads.
- Testing: no unit tests; hard-to-test code due to tight coupling or hidden globals.
- Configuration: hardcoded paths/secrets; configuration scattered instead of centralized.

Best practice violations to check
- Mutable default arguments; shadowing builtins; reliance on global state.
- Security: constructing SQL/commands with string formatting instead of parameters; storing secrets in code; not validating/escaping inputs; disabling TLS verification.
- Packaging/versions: unpinned dependencies; mixing runtime and dev dependencies; no lockfile.
- Logging: using prints instead of logging with levels; missing context; excessive logging in hot paths.
- Performance hygiene: not profiling (cProfile, py-spy, scalene); no benchmarking; premature micro-optimizations.
- Pandas/NumPy specifics: chained assignments, inplace misuse, hidden copies; mixed dtypes; relying on object dtype for numbers.

Helpful context to include with your code
- Python version and key libraries; data sizes and typical workload.
- What’s slow (functions, inputs), current timings/memory, and target goals.
- Environment (local, serverless, container) and constraints (CPU, RAM).

Once you share the code, I’ll provide concrete, line-level suggestions for:
- Specific bottlenecks with proposed fixes and expected complexity improvements.
- Refactors to improve readability and maintainability.
- Best-practice corrections and safer patterns.

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