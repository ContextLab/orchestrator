# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:16:34

## Analysis Results

I can’t see any Python code to analyze—the snippet contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:16:34

## Analysis Results

I can’t see any Python code to analyze—the snippet contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:16:34

## Analysis Results

I can’t see any Python code to analyze—the snippet contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:16:34

## Analysis Results

I can’t see any Python code to analyze—the snippet contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:16:34

## Analysis Results

I can’t see any Python code to analyze—the snippet contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:16:34

## Analysis Results

I can’t see any Python code to analyze—the snippet contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:16:34

## Analysis Results

I can’t see any Python code to analyze—the snippet contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:16:34

## Analysis Results

I can’t see any Python code to analyze—the snippet contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:16:34

## Analysis Results

I can’t see any Python code to analyze—the snippet contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:16:34

## Analysis Results

I can’t see any Python code to analyze—the snippet contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:16:34

## Analysis Results

I can’t see any Python code to analyze—the snippet contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:16:34

## Analysis Results

I can’t see any Python code to analyze—the snippet contains a placeholder (“{{content}}”). Please paste the code (or link to a gist/repo), plus any context such as Python version, input sizes, and which parts feel slow.

While you share that, here’s exactly what I’ll review and report back on:

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested scans, repeated work)
- Inefficient data structures (e.g., list lookups vs set/dict, unnecessary copies)
- Hot loops doing I/O or heavy allocations; repeated attribute lookups in tight loops
- String building via += instead of join; excessive JSON/regex work
- N+1 queries, chatty network calls, unbatched I/O; missing caching (functools.lru_cache)
- Missed vectorization opportunities (NumPy/Pandas), using apply/iterrows instead of vectorized ops
- Concurrency/parallelism (CPU-bound without multiprocessing; I/O-bound without asyncio/threads)
- Memory churn/leaks (large intermediate lists; not streaming; unnecessary materialization)

2) Code quality issues
- Long, monolithic functions; deep nesting; duplication; unclear names
- Missing/insufficient docstrings and type hints; weak cohesion between modules
- Inconsistent style; magic numbers; dead code; commented-out code
- Poor error handling (bare except, swallowing exceptions, control-flow via exceptions)
- Resource handling (no context managers for files/sockets); global state; side effects
- Tight coupling; violation of single-responsibility; lack of tests/logging

3) Best practice violations
- PEP 8/style issues; no formatter (black) or linter (ruff/flake8); no import sorter (isort)
- Mutable default arguments; implicit relative imports; missing if __name__ == "__main__"
- Using print instead of logging; no timeouts/retries on network calls
- Security pitfalls (pickle/untrusted YAML load, subprocess without shlex, SQL injection risk)
- Packaging/dependency hygiene (unpinned deps, mixed tool configs, no pyproject.toml)
- Path handling via os.path instead of pathlib; f-strings preferred over format
- Missing tests/type checks (pytest, pytest-benchmark, mypy/pyright)
- Not using context managers, dataclasses, Enum, or stdlib tools (itertools, heapq, bisect, collections)

If you prefer, I can also run a quick tool-based pass and include their outputs:
- Performance: cProfile + snakeviz, py-spy, scalene, line_profiler; memory_profiler
- Quality/security: ruff or flake8, black, isort, mypy/pyright, pylint, bandit, vulture

Send the code and any constraints, and I’ll deliver a concrete, itemized report with targeted fixes and estimated impact.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”). Please paste the code (or link to a gist/repo), plus any context such as Python version, input sizes, and which parts feel slow.

While you share that, here’s exactly what I’ll review and report back on:

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested scans, repeated work)
- Inefficient data structures (e.g., list lookups vs set/dict, unnecessary copies)
- Hot loops doing I/O or heavy allocations; repeated attribute lookups in tight loops
- String building via += instead of join; excessive JSON/regex work
- N+1 queries, chatty network calls, unbatched I/O; missing caching (functools.lru_cache)
- Missed vectorization opportunities (NumPy/Pandas), using apply/iterrows instead of vectorized ops
- Concurrency/parallelism (CPU-bound without multiprocessing; I/O-bound without asyncio/threads)
- Memory churn/leaks (large intermediate lists; not streaming; unnecessary materialization)

2) Code quality issues
- Long, monolithic functions; deep nesting; duplication; unclear names
- Missing/insufficient docstrings and type hints; weak cohesion between modules
- Inconsistent style; magic numbers; dead code; commented-out code
- Poor error handling (bare except, swallowing exceptions, control-flow via exceptions)
- Resource handling (no context managers for files/sockets); global state; side effects
- Tight coupling; violation of single-responsibility; lack of tests/logging

3) Best practice violations
- PEP 8/style issues; no formatter (black) or linter (ruff/flake8); no import sorter (isort)
- Mutable default arguments; implicit relative imports; missing if __name__ == "__main__"
- Using print instead of logging; no timeouts/retries on network calls
- Security pitfalls (pickle/untrusted YAML load, subprocess without shlex, SQL injection risk)
- Packaging/dependency hygiene (unpinned deps, mixed tool configs, no pyproject.toml)
- Path handling via os.path instead of pathlib; f-strings preferred over format
- Missing tests/type checks (pytest, pytest-benchmark, mypy/pyright)
- Not using context managers, dataclasses, Enum, or stdlib tools (itertools, heapq, bisect, collections)

If you prefer, I can also run a quick tool-based pass and include their outputs:
- Performance: cProfile + snakeviz, py-spy, scalene, line_profiler; memory_profiler
- Quality/security: ruff or flake8, black, isort, mypy/pyright, pylint, bandit, vulture

Send the code and any constraints, and I’ll deliver a concrete, itemized report with targeted fixes and estimated impact.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”). Please paste the code (or link to a gist/repo), plus any context such as Python version, input sizes, and which parts feel slow.

While you share that, here’s exactly what I’ll review and report back on:

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested scans, repeated work)
- Inefficient data structures (e.g., list lookups vs set/dict, unnecessary copies)
- Hot loops doing I/O or heavy allocations; repeated attribute lookups in tight loops
- String building via += instead of join; excessive JSON/regex work
- N+1 queries, chatty network calls, unbatched I/O; missing caching (functools.lru_cache)
- Missed vectorization opportunities (NumPy/Pandas), using apply/iterrows instead of vectorized ops
- Concurrency/parallelism (CPU-bound without multiprocessing; I/O-bound without asyncio/threads)
- Memory churn/leaks (large intermediate lists; not streaming; unnecessary materialization)

2) Code quality issues
- Long, monolithic functions; deep nesting; duplication; unclear names
- Missing/insufficient docstrings and type hints; weak cohesion between modules
- Inconsistent style; magic numbers; dead code; commented-out code
- Poor error handling (bare except, swallowing exceptions, control-flow via exceptions)
- Resource handling (no context managers for files/sockets); global state; side effects
- Tight coupling; violation of single-responsibility; lack of tests/logging

3) Best practice violations
- PEP 8/style issues; no formatter (black) or linter (ruff/flake8); no import sorter (isort)
- Mutable default arguments; implicit relative imports; missing if __name__ == "__main__"
- Using print instead of logging; no timeouts/retries on network calls
- Security pitfalls (pickle/untrusted YAML load, subprocess without shlex, SQL injection risk)
- Packaging/dependency hygiene (unpinned deps, mixed tool configs, no pyproject.toml)
- Path handling via os.path instead of pathlib; f-strings preferred over format
- Missing tests/type checks (pytest, pytest-benchmark, mypy/pyright)
- Not using context managers, dataclasses, Enum, or stdlib tools (itertools, heapq, bisect, collections)

If you prefer, I can also run a quick tool-based pass and include their outputs:
- Performance: cProfile + snakeviz, py-spy, scalene, line_profiler; memory_profiler
- Quality/security: ruff or flake8, black, isort, mypy/pyright, pylint, bandit, vulture

Send the code and any constraints, and I’ll deliver a concrete, itemized report with targeted fixes and estimated impact.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”). Please paste the code (or link to a gist/repo), plus any context such as Python version, input sizes, and which parts feel slow.

While you share that, here’s exactly what I’ll review and report back on:

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested scans, repeated work)
- Inefficient data structures (e.g., list lookups vs set/dict, unnecessary copies)
- Hot loops doing I/O or heavy allocations; repeated attribute lookups in tight loops
- String building via += instead of join; excessive JSON/regex work
- N+1 queries, chatty network calls, unbatched I/O; missing caching (functools.lru_cache)
- Missed vectorization opportunities (NumPy/Pandas), using apply/iterrows instead of vectorized ops
- Concurrency/parallelism (CPU-bound without multiprocessing; I/O-bound without asyncio/threads)
- Memory churn/leaks (large intermediate lists; not streaming; unnecessary materialization)

2) Code quality issues
- Long, monolithic functions; deep nesting; duplication; unclear names
- Missing/insufficient docstrings and type hints; weak cohesion between modules
- Inconsistent style; magic numbers; dead code; commented-out code
- Poor error handling (bare except, swallowing exceptions, control-flow via exceptions)
- Resource handling (no context managers for files/sockets); global state; side effects
- Tight coupling; violation of single-responsibility; lack of tests/logging

3) Best practice violations
- PEP 8/style issues; no formatter (black) or linter (ruff/flake8); no import sorter (isort)
- Mutable default arguments; implicit relative imports; missing if __name__ == "__main__"
- Using print instead of logging; no timeouts/retries on network calls
- Security pitfalls (pickle/untrusted YAML load, subprocess without shlex, SQL injection risk)
- Packaging/dependency hygiene (unpinned deps, mixed tool configs, no pyproject.toml)
- Path handling via os.path instead of pathlib; f-strings preferred over format
- Missing tests/type checks (pytest, pytest-benchmark, mypy/pyright)
- Not using context managers, dataclasses, Enum, or stdlib tools (itertools, heapq, bisect, collections)

If you prefer, I can also run a quick tool-based pass and include their outputs:
- Performance: cProfile + snakeviz, py-spy, scalene, line_profiler; memory_profiler
- Quality/security: ruff or flake8, black, isort, mypy/pyright, pylint, bandit, vulture

Send the code and any constraints, and I’ll deliver a concrete, itemized report with targeted fixes and estimated impact.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”). Please paste the code (or link to a gist/repo), plus any context such as Python version, input sizes, and which parts feel slow.

While you share that, here’s exactly what I’ll review and report back on:

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested scans, repeated work)
- Inefficient data structures (e.g., list lookups vs set/dict, unnecessary copies)
- Hot loops doing I/O or heavy allocations; repeated attribute lookups in tight loops
- String building via += instead of join; excessive JSON/regex work
- N+1 queries, chatty network calls, unbatched I/O; missing caching (functools.lru_cache)
- Missed vectorization opportunities (NumPy/Pandas), using apply/iterrows instead of vectorized ops
- Concurrency/parallelism (CPU-bound without multiprocessing; I/O-bound without asyncio/threads)
- Memory churn/leaks (large intermediate lists; not streaming; unnecessary materialization)

2) Code quality issues
- Long, monolithic functions; deep nesting; duplication; unclear names
- Missing/insufficient docstrings and type hints; weak cohesion between modules
- Inconsistent style; magic numbers; dead code; commented-out code
- Poor error handling (bare except, swallowing exceptions, control-flow via exceptions)
- Resource handling (no context managers for files/sockets); global state; side effects
- Tight coupling; violation of single-responsibility; lack of tests/logging

3) Best practice violations
- PEP 8/style issues; no formatter (black) or linter (ruff/flake8); no import sorter (isort)
- Mutable default arguments; implicit relative imports; missing if __name__ == "__main__"
- Using print instead of logging; no timeouts/retries on network calls
- Security pitfalls (pickle/untrusted YAML load, subprocess without shlex, SQL injection risk)
- Packaging/dependency hygiene (unpinned deps, mixed tool configs, no pyproject.toml)
- Path handling via os.path instead of pathlib; f-strings preferred over format
- Missing tests/type checks (pytest, pytest-benchmark, mypy/pyright)
- Not using context managers, dataclasses, Enum, or stdlib tools (itertools, heapq, bisect, collections)

If you prefer, I can also run a quick tool-based pass and include their outputs:
- Performance: cProfile + snakeviz, py-spy, scalene, line_profiler; memory_profiler
- Quality/security: ruff or flake8, black, isort, mypy/pyright, pylint, bandit, vulture

Send the code and any constraints, and I’ll deliver a concrete, itemized report with targeted fixes and estimated impact.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”). Please paste the code (or link to a gist/repo), plus any context such as Python version, input sizes, and which parts feel slow.

While you share that, here’s exactly what I’ll review and report back on:

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested scans, repeated work)
- Inefficient data structures (e.g., list lookups vs set/dict, unnecessary copies)
- Hot loops doing I/O or heavy allocations; repeated attribute lookups in tight loops
- String building via += instead of join; excessive JSON/regex work
- N+1 queries, chatty network calls, unbatched I/O; missing caching (functools.lru_cache)
- Missed vectorization opportunities (NumPy/Pandas), using apply/iterrows instead of vectorized ops
- Concurrency/parallelism (CPU-bound without multiprocessing; I/O-bound without asyncio/threads)
- Memory churn/leaks (large intermediate lists; not streaming; unnecessary materialization)

2) Code quality issues
- Long, monolithic functions; deep nesting; duplication; unclear names
- Missing/insufficient docstrings and type hints; weak cohesion between modules
- Inconsistent style; magic numbers; dead code; commented-out code
- Poor error handling (bare except, swallowing exceptions, control-flow via exceptions)
- Resource handling (no context managers for files/sockets); global state; side effects
- Tight coupling; violation of single-responsibility; lack of tests/logging

3) Best practice violations
- PEP 8/style issues; no formatter (black) or linter (ruff/flake8); no import sorter (isort)
- Mutable default arguments; implicit relative imports; missing if __name__ == "__main__"
- Using print instead of logging; no timeouts/retries on network calls
- Security pitfalls (pickle/untrusted YAML load, subprocess without shlex, SQL injection risk)
- Packaging/dependency hygiene (unpinned deps, mixed tool configs, no pyproject.toml)
- Path handling via os.path instead of pathlib; f-strings preferred over format
- Missing tests/type checks (pytest, pytest-benchmark, mypy/pyright)
- Not using context managers, dataclasses, Enum, or stdlib tools (itertools, heapq, bisect, collections)

If you prefer, I can also run a quick tool-based pass and include their outputs:
- Performance: cProfile + snakeviz, py-spy, scalene, line_profiler; memory_profiler
- Quality/security: ruff or flake8, black, isort, mypy/pyright, pylint, bandit, vulture

Send the code and any constraints, and I’ll deliver a concrete, itemized report with targeted fixes and estimated impact.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”). Please paste the code (or link to a gist/repo), plus any context such as Python version, input sizes, and which parts feel slow.

While you share that, here’s exactly what I’ll review and report back on:

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested scans, repeated work)
- Inefficient data structures (e.g., list lookups vs set/dict, unnecessary copies)
- Hot loops doing I/O or heavy allocations; repeated attribute lookups in tight loops
- String building via += instead of join; excessive JSON/regex work
- N+1 queries, chatty network calls, unbatched I/O; missing caching (functools.lru_cache)
- Missed vectorization opportunities (NumPy/Pandas), using apply/iterrows instead of vectorized ops
- Concurrency/parallelism (CPU-bound without multiprocessing; I/O-bound without asyncio/threads)
- Memory churn/leaks (large intermediate lists; not streaming; unnecessary materialization)

2) Code quality issues
- Long, monolithic functions; deep nesting; duplication; unclear names
- Missing/insufficient docstrings and type hints; weak cohesion between modules
- Inconsistent style; magic numbers; dead code; commented-out code
- Poor error handling (bare except, swallowing exceptions, control-flow via exceptions)
- Resource handling (no context managers for files/sockets); global state; side effects
- Tight coupling; violation of single-responsibility; lack of tests/logging

3) Best practice violations
- PEP 8/style issues; no formatter (black) or linter (ruff/flake8); no import sorter (isort)
- Mutable default arguments; implicit relative imports; missing if __name__ == "__main__"
- Using print instead of logging; no timeouts/retries on network calls
- Security pitfalls (pickle/untrusted YAML load, subprocess without shlex, SQL injection risk)
- Packaging/dependency hygiene (unpinned deps, mixed tool configs, no pyproject.toml)
- Path handling via os.path instead of pathlib; f-strings preferred over format
- Missing tests/type checks (pytest, pytest-benchmark, mypy/pyright)
- Not using context managers, dataclasses, Enum, or stdlib tools (itertools, heapq, bisect, collections)

If you prefer, I can also run a quick tool-based pass and include their outputs:
- Performance: cProfile + snakeviz, py-spy, scalene, line_profiler; memory_profiler
- Quality/security: ruff or flake8, black, isort, mypy/pyright, pylint, bandit, vulture

Send the code and any constraints, and I’ll deliver a concrete, itemized report with targeted fixes and estimated impact.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”). Please paste the code (or link to a gist/repo), plus any context such as Python version, input sizes, and which parts feel slow.

While you share that, here’s exactly what I’ll review and report back on:

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested scans, repeated work)
- Inefficient data structures (e.g., list lookups vs set/dict, unnecessary copies)
- Hot loops doing I/O or heavy allocations; repeated attribute lookups in tight loops
- String building via += instead of join; excessive JSON/regex work
- N+1 queries, chatty network calls, unbatched I/O; missing caching (functools.lru_cache)
- Missed vectorization opportunities (NumPy/Pandas), using apply/iterrows instead of vectorized ops
- Concurrency/parallelism (CPU-bound without multiprocessing; I/O-bound without asyncio/threads)
- Memory churn/leaks (large intermediate lists; not streaming; unnecessary materialization)

2) Code quality issues
- Long, monolithic functions; deep nesting; duplication; unclear names
- Missing/insufficient docstrings and type hints; weak cohesion between modules
- Inconsistent style; magic numbers; dead code; commented-out code
- Poor error handling (bare except, swallowing exceptions, control-flow via exceptions)
- Resource handling (no context managers for files/sockets); global state; side effects
- Tight coupling; violation of single-responsibility; lack of tests/logging

3) Best practice violations
- PEP 8/style issues; no formatter (black) or linter (ruff/flake8); no import sorter (isort)
- Mutable default arguments; implicit relative imports; missing if __name__ == "__main__"
- Using print instead of logging; no timeouts/retries on network calls
- Security pitfalls (pickle/untrusted YAML load, subprocess without shlex, SQL injection risk)
- Packaging/dependency hygiene (unpinned deps, mixed tool configs, no pyproject.toml)
- Path handling via os.path instead of pathlib; f-strings preferred over format
- Missing tests/type checks (pytest, pytest-benchmark, mypy/pyright)
- Not using context managers, dataclasses, Enum, or stdlib tools (itertools, heapq, bisect, collections)

If you prefer, I can also run a quick tool-based pass and include their outputs:
- Performance: cProfile + snakeviz, py-spy, scalene, line_profiler; memory_profiler
- Quality/security: ruff or flake8, black, isort, mypy/pyright, pylint, bandit, vulture

Send the code and any constraints, and I’ll deliver a concrete, itemized report with targeted fixes and estimated impact.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”). Please paste the code (or link to a gist/repo), plus any context such as Python version, input sizes, and which parts feel slow.

While you share that, here’s exactly what I’ll review and report back on:

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested scans, repeated work)
- Inefficient data structures (e.g., list lookups vs set/dict, unnecessary copies)
- Hot loops doing I/O or heavy allocations; repeated attribute lookups in tight loops
- String building via += instead of join; excessive JSON/regex work
- N+1 queries, chatty network calls, unbatched I/O; missing caching (functools.lru_cache)
- Missed vectorization opportunities (NumPy/Pandas), using apply/iterrows instead of vectorized ops
- Concurrency/parallelism (CPU-bound without multiprocessing; I/O-bound without asyncio/threads)
- Memory churn/leaks (large intermediate lists; not streaming; unnecessary materialization)

2) Code quality issues
- Long, monolithic functions; deep nesting; duplication; unclear names
- Missing/insufficient docstrings and type hints; weak cohesion between modules
- Inconsistent style; magic numbers; dead code; commented-out code
- Poor error handling (bare except, swallowing exceptions, control-flow via exceptions)
- Resource handling (no context managers for files/sockets); global state; side effects
- Tight coupling; violation of single-responsibility; lack of tests/logging

3) Best practice violations
- PEP 8/style issues; no formatter (black) or linter (ruff/flake8); no import sorter (isort)
- Mutable default arguments; implicit relative imports; missing if __name__ == "__main__"
- Using print instead of logging; no timeouts/retries on network calls
- Security pitfalls (pickle/untrusted YAML load, subprocess without shlex, SQL injection risk)
- Packaging/dependency hygiene (unpinned deps, mixed tool configs, no pyproject.toml)
- Path handling via os.path instead of pathlib; f-strings preferred over format
- Missing tests/type checks (pytest, pytest-benchmark, mypy/pyright)
- Not using context managers, dataclasses, Enum, or stdlib tools (itertools, heapq, bisect, collections)

If you prefer, I can also run a quick tool-based pass and include their outputs:
- Performance: cProfile + snakeviz, py-spy, scalene, line_profiler; memory_profiler
- Quality/security: ruff or flake8, black, isort, mypy/pyright, pylint, bandit, vulture

Send the code and any constraints, and I’ll deliver a concrete, itemized report with targeted fixes and estimated impact.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”). Please paste the code (or link to a gist/repo), plus any context such as Python version, input sizes, and which parts feel slow.

While you share that, here’s exactly what I’ll review and report back on:

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested scans, repeated work)
- Inefficient data structures (e.g., list lookups vs set/dict, unnecessary copies)
- Hot loops doing I/O or heavy allocations; repeated attribute lookups in tight loops
- String building via += instead of join; excessive JSON/regex work
- N+1 queries, chatty network calls, unbatched I/O; missing caching (functools.lru_cache)
- Missed vectorization opportunities (NumPy/Pandas), using apply/iterrows instead of vectorized ops
- Concurrency/parallelism (CPU-bound without multiprocessing; I/O-bound without asyncio/threads)
- Memory churn/leaks (large intermediate lists; not streaming; unnecessary materialization)

2) Code quality issues
- Long, monolithic functions; deep nesting; duplication; unclear names
- Missing/insufficient docstrings and type hints; weak cohesion between modules
- Inconsistent style; magic numbers; dead code; commented-out code
- Poor error handling (bare except, swallowing exceptions, control-flow via exceptions)
- Resource handling (no context managers for files/sockets); global state; side effects
- Tight coupling; violation of single-responsibility; lack of tests/logging

3) Best practice violations
- PEP 8/style issues; no formatter (black) or linter (ruff/flake8); no import sorter (isort)
- Mutable default arguments; implicit relative imports; missing if __name__ == "__main__"
- Using print instead of logging; no timeouts/retries on network calls
- Security pitfalls (pickle/untrusted YAML load, subprocess without shlex, SQL injection risk)
- Packaging/dependency hygiene (unpinned deps, mixed tool configs, no pyproject.toml)
- Path handling via os.path instead of pathlib; f-strings preferred over format
- Missing tests/type checks (pytest, pytest-benchmark, mypy/pyright)
- Not using context managers, dataclasses, Enum, or stdlib tools (itertools, heapq, bisect, collections)

If you prefer, I can also run a quick tool-based pass and include their outputs:
- Performance: cProfile + snakeviz, py-spy, scalene, line_profiler; memory_profiler
- Quality/security: ruff or flake8, black, isort, mypy/pyright, pylint, bandit, vulture

Send the code and any constraints, and I’ll deliver a concrete, itemized report with targeted fixes and estimated impact.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”). Please paste the code (or link to a gist/repo), plus any context such as Python version, input sizes, and which parts feel slow.

While you share that, here’s exactly what I’ll review and report back on:

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested scans, repeated work)
- Inefficient data structures (e.g., list lookups vs set/dict, unnecessary copies)
- Hot loops doing I/O or heavy allocations; repeated attribute lookups in tight loops
- String building via += instead of join; excessive JSON/regex work
- N+1 queries, chatty network calls, unbatched I/O; missing caching (functools.lru_cache)
- Missed vectorization opportunities (NumPy/Pandas), using apply/iterrows instead of vectorized ops
- Concurrency/parallelism (CPU-bound without multiprocessing; I/O-bound without asyncio/threads)
- Memory churn/leaks (large intermediate lists; not streaming; unnecessary materialization)

2) Code quality issues
- Long, monolithic functions; deep nesting; duplication; unclear names
- Missing/insufficient docstrings and type hints; weak cohesion between modules
- Inconsistent style; magic numbers; dead code; commented-out code
- Poor error handling (bare except, swallowing exceptions, control-flow via exceptions)
- Resource handling (no context managers for files/sockets); global state; side effects
- Tight coupling; violation of single-responsibility; lack of tests/logging

3) Best practice violations
- PEP 8/style issues; no formatter (black) or linter (ruff/flake8); no import sorter (isort)
- Mutable default arguments; implicit relative imports; missing if __name__ == "__main__"
- Using print instead of logging; no timeouts/retries on network calls
- Security pitfalls (pickle/untrusted YAML load, subprocess without shlex, SQL injection risk)
- Packaging/dependency hygiene (unpinned deps, mixed tool configs, no pyproject.toml)
- Path handling via os.path instead of pathlib; f-strings preferred over format
- Missing tests/type checks (pytest, pytest-benchmark, mypy/pyright)
- Not using context managers, dataclasses, Enum, or stdlib tools (itertools, heapq, bisect, collections)

If you prefer, I can also run a quick tool-based pass and include their outputs:
- Performance: cProfile + snakeviz, py-spy, scalene, line_profiler; memory_profiler
- Quality/security: ruff or flake8, black, isort, mypy/pyright, pylint, bandit, vulture

Send the code and any constraints, and I’ll deliver a concrete, itemized report with targeted fixes and estimated impact.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”). Please paste the code (or link to a gist/repo), plus any context such as Python version, input sizes, and which parts feel slow.

While you share that, here’s exactly what I’ll review and report back on:

1) Performance bottlenecks
- Algorithmic complexity (quadratic loops, nested scans, repeated work)
- Inefficient data structures (e.g., list lookups vs set/dict, unnecessary copies)
- Hot loops doing I/O or heavy allocations; repeated attribute lookups in tight loops
- String building via += instead of join; excessive JSON/regex work
- N+1 queries, chatty network calls, unbatched I/O; missing caching (functools.lru_cache)
- Missed vectorization opportunities (NumPy/Pandas), using apply/iterrows instead of vectorized ops
- Concurrency/parallelism (CPU-bound without multiprocessing; I/O-bound without asyncio/threads)
- Memory churn/leaks (large intermediate lists; not streaming; unnecessary materialization)

2) Code quality issues
- Long, monolithic functions; deep nesting; duplication; unclear names
- Missing/insufficient docstrings and type hints; weak cohesion between modules
- Inconsistent style; magic numbers; dead code; commented-out code
- Poor error handling (bare except, swallowing exceptions, control-flow via exceptions)
- Resource handling (no context managers for files/sockets); global state; side effects
- Tight coupling; violation of single-responsibility; lack of tests/logging

3) Best practice violations
- PEP 8/style issues; no formatter (black) or linter (ruff/flake8); no import sorter (isort)
- Mutable default arguments; implicit relative imports; missing if __name__ == "__main__"
- Using print instead of logging; no timeouts/retries on network calls
- Security pitfalls (pickle/untrusted YAML load, subprocess without shlex, SQL injection risk)
- Packaging/dependency hygiene (unpinned deps, mixed tool configs, no pyproject.toml)
- Path handling via os.path instead of pathlib; f-strings preferred over format
- Missing tests/type checks (pytest, pytest-benchmark, mypy/pyright)
- Not using context managers, dataclasses, Enum, or stdlib tools (itertools, heapq, bisect, collections)

If you prefer, I can also run a quick tool-based pass and include their outputs:
- Performance: cProfile + snakeviz, py-spy, scalene, line_profiler; memory_profiler
- Quality/security: ruff or flake8, black, isort, mypy/pyright, pylint, bandit, vulture

Send the code and any constraints, and I’ll deliver a concrete, itemized report with targeted fixes and estimated impact.

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