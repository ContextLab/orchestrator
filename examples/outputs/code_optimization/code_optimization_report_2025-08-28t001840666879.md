# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:18:40

## Analysis Results

I don’t see any code in the placeholder # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:18:40

## Analysis Results

I don’t see any code in the placeholder # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:18:40

## Analysis Results

I don’t see any code in the placeholder # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:18:40

## Analysis Results

I don’t see any code in the placeholder # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:18:40

## Analysis Results

I don’t see any code in the placeholder # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:18:40

## Analysis Results

I don’t see any code in the placeholder # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:18:40

## Analysis Results

I don’t see any code in the placeholder # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:18:40

## Analysis Results

I don’t see any code in the placeholder # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:18:40

## Analysis Results

I don’t see any code in the placeholder # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:18:40

## Analysis Results

I don’t see any code in the placeholder # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:18:40

## Analysis Results

I don’t see any code in the placeholder # Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:18:40

## Analysis Results

I don’t see any code in the placeholder {{content}}. Please paste the Python code (or a minimal reproducible snippet), plus:

- Python version and OS
- Key libraries used
- Typical input sizes and where it feels slow
- Desired runtime/memory targets

Once you share it, I’ll deliver a structured review covering:
1) Performance bottlenecks, 2) Code quality issues, 3) Best-practice violations, with concrete fixes.

While you gather that, here’s a concise checklist you can run through:

Performance bottlenecks (common)
- Algorithmic complexity: nested loops over large data; repeated “x in list” lookups (use sets/dicts); unnecessary sorting.
- Recomputing work: add caching (functools.lru_cache), hoist invariants out of loops, precompile regexes.
- Data handling: avoid string concatenation in loops (use join/StringIO); batch I/O and DB operations; minimize per-row work in pandas (prefer vectorization/merge over apply/iterrows).
- Numerics: prefer NumPy vectorization; avoid Python loops in hot paths; use numba/Cython for tight loops if needed.
- Concurrency: CPU-bound → multiprocessing or native extensions; I/O-bound → asyncio/threading; avoid blocking calls in async code.
- External systems: N+1 DB queries; missing indexes; chatty HTTP without keep-alive/batching; excessive logging in hot paths.
- Memory: large intermediate lists; copy-heavy pandas ops; use generators/itertuples/chunked reads; specify dtypes.

Code quality issues
- Long functions, duplication, inconsistent naming, magic numbers.
- Missing type hints/docstrings; unclear interfaces.
- Broad exceptions, swallowed errors, no specific exception types.
- Mutable default args; global state; side effects in import time.
- Resource handling: not using context managers for files/sockets; unclosed sessions.
- Dead/unreachable code; debug prints instead of logging.

Best-practice violations
- No formatter/linter; lack of type checking (mypy/pyright).
- No virtualenv/requirements pinning; missing lock files.
- Logging misuse (root logger, excessive default DEBUG).
- Security: string-formatted SQL (risk of injection), shell=True with user input, secrets in code, weak crypto.
- Path handling: hard-coded paths instead of pathlib.
- Async: time.sleep in coroutines; creating event loops improperly; mixing blocking I/O in async paths.

Tools and quick profiling starters
- CPU: python -m cProfile -o prof.out your_script.py; visualize with snakeviz prof.out
- Line-level: pip install line_profiler; add @profile to hot funcs; kernprof -l -v your_script.py
- Memory: python -X tracemalloc your_script.py; or pip install memory_profiler and use @profile
- Sampling: py-spy top/record for low-overhead production profiling
- Lint/format: ruff, black, isort
- Types: mypy or pyright
- Security: bandit, pip-audit

Paste the code and I’ll give targeted, line-specific recommendations and optimized rewrites where helpful.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions. Please paste the Python code (or a minimal reproducible snippet), plus:

- Python version and OS
- Key libraries used
- Typical input sizes and where it feels slow
- Desired runtime/memory targets

Once you share it, I’ll deliver a structured review covering:
1) Performance bottlenecks, 2) Code quality issues, 3) Best-practice violations, with concrete fixes.

While you gather that, here’s a concise checklist you can run through:

Performance bottlenecks (common)
- Algorithmic complexity: nested loops over large data; repeated “x in list” lookups (use sets/dicts); unnecessary sorting.
- Recomputing work: add caching (functools.lru_cache), hoist invariants out of loops, precompile regexes.
- Data handling: avoid string concatenation in loops (use join/StringIO); batch I/O and DB operations; minimize per-row work in pandas (prefer vectorization/merge over apply/iterrows).
- Numerics: prefer NumPy vectorization; avoid Python loops in hot paths; use numba/Cython for tight loops if needed.
- Concurrency: CPU-bound → multiprocessing or native extensions; I/O-bound → asyncio/threading; avoid blocking calls in async code.
- External systems: N+1 DB queries; missing indexes; chatty HTTP without keep-alive/batching; excessive logging in hot paths.
- Memory: large intermediate lists; copy-heavy pandas ops; use generators/itertuples/chunked reads; specify dtypes.

Code quality issues
- Long functions, duplication, inconsistent naming, magic numbers.
- Missing type hints/docstrings; unclear interfaces.
- Broad exceptions, swallowed errors, no specific exception types.
- Mutable default args; global state; side effects in import time.
- Resource handling: not using context managers for files/sockets; unclosed sessions.
- Dead/unreachable code; debug prints instead of logging.

Best-practice violations
- No formatter/linter; lack of type checking (mypy/pyright).
- No virtualenv/requirements pinning; missing lock files.
- Logging misuse (root logger, excessive default DEBUG).
- Security: string-formatted SQL (risk of injection), shell=True with user input, secrets in code, weak crypto.
- Path handling: hard-coded paths instead of pathlib.
- Async: time.sleep in coroutines; creating event loops improperly; mixing blocking I/O in async paths.

Tools and quick profiling starters
- CPU: python -m cProfile -o prof.out your_script.py; visualize with snakeviz prof.out
- Line-level: pip install line_profiler; add @profile to hot funcs; kernprof -l -v your_script.py
- Memory: python -X tracemalloc your_script.py; or pip install memory_profiler and use @profile
- Sampling: py-spy top/record for low-overhead production profiling
- Lint/format: ruff, black, isort
- Types: mypy or pyright
- Security: bandit, pip-audit

Paste the code and I’ll give targeted, line-specific recommendations and optimized rewrites where helpful.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions. Please paste the Python code (or a minimal reproducible snippet), plus:

- Python version and OS
- Key libraries used
- Typical input sizes and where it feels slow
- Desired runtime/memory targets

Once you share it, I’ll deliver a structured review covering:
1) Performance bottlenecks, 2) Code quality issues, 3) Best-practice violations, with concrete fixes.

While you gather that, here’s a concise checklist you can run through:

Performance bottlenecks (common)
- Algorithmic complexity: nested loops over large data; repeated “x in list” lookups (use sets/dicts); unnecessary sorting.
- Recomputing work: add caching (functools.lru_cache), hoist invariants out of loops, precompile regexes.
- Data handling: avoid string concatenation in loops (use join/StringIO); batch I/O and DB operations; minimize per-row work in pandas (prefer vectorization/merge over apply/iterrows).
- Numerics: prefer NumPy vectorization; avoid Python loops in hot paths; use numba/Cython for tight loops if needed.
- Concurrency: CPU-bound → multiprocessing or native extensions; I/O-bound → asyncio/threading; avoid blocking calls in async code.
- External systems: N+1 DB queries; missing indexes; chatty HTTP without keep-alive/batching; excessive logging in hot paths.
- Memory: large intermediate lists; copy-heavy pandas ops; use generators/itertuples/chunked reads; specify dtypes.

Code quality issues
- Long functions, duplication, inconsistent naming, magic numbers.
- Missing type hints/docstrings; unclear interfaces.
- Broad exceptions, swallowed errors, no specific exception types.
- Mutable default args; global state; side effects in import time.
- Resource handling: not using context managers for files/sockets; unclosed sessions.
- Dead/unreachable code; debug prints instead of logging.

Best-practice violations
- No formatter/linter; lack of type checking (mypy/pyright).
- No virtualenv/requirements pinning; missing lock files.
- Logging misuse (root logger, excessive default DEBUG).
- Security: string-formatted SQL (risk of injection), shell=True with user input, secrets in code, weak crypto.
- Path handling: hard-coded paths instead of pathlib.
- Async: time.sleep in coroutines; creating event loops improperly; mixing blocking I/O in async paths.

Tools and quick profiling starters
- CPU: python -m cProfile -o prof.out your_script.py; visualize with snakeviz prof.out
- Line-level: pip install line_profiler; add @profile to hot funcs; kernprof -l -v your_script.py
- Memory: python -X tracemalloc your_script.py; or pip install memory_profiler and use @profile
- Sampling: py-spy top/record for low-overhead production profiling
- Lint/format: ruff, black, isort
- Types: mypy or pyright
- Security: bandit, pip-audit

Paste the code and I’ll give targeted, line-specific recommendations and optimized rewrites where helpful.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions. Please paste the Python code (or a minimal reproducible snippet), plus:

- Python version and OS
- Key libraries used
- Typical input sizes and where it feels slow
- Desired runtime/memory targets

Once you share it, I’ll deliver a structured review covering:
1) Performance bottlenecks, 2) Code quality issues, 3) Best-practice violations, with concrete fixes.

While you gather that, here’s a concise checklist you can run through:

Performance bottlenecks (common)
- Algorithmic complexity: nested loops over large data; repeated “x in list” lookups (use sets/dicts); unnecessary sorting.
- Recomputing work: add caching (functools.lru_cache), hoist invariants out of loops, precompile regexes.
- Data handling: avoid string concatenation in loops (use join/StringIO); batch I/O and DB operations; minimize per-row work in pandas (prefer vectorization/merge over apply/iterrows).
- Numerics: prefer NumPy vectorization; avoid Python loops in hot paths; use numba/Cython for tight loops if needed.
- Concurrency: CPU-bound → multiprocessing or native extensions; I/O-bound → asyncio/threading; avoid blocking calls in async code.
- External systems: N+1 DB queries; missing indexes; chatty HTTP without keep-alive/batching; excessive logging in hot paths.
- Memory: large intermediate lists; copy-heavy pandas ops; use generators/itertuples/chunked reads; specify dtypes.

Code quality issues
- Long functions, duplication, inconsistent naming, magic numbers.
- Missing type hints/docstrings; unclear interfaces.
- Broad exceptions, swallowed errors, no specific exception types.
- Mutable default args; global state; side effects in import time.
- Resource handling: not using context managers for files/sockets; unclosed sessions.
- Dead/unreachable code; debug prints instead of logging.

Best-practice violations
- No formatter/linter; lack of type checking (mypy/pyright).
- No virtualenv/requirements pinning; missing lock files.
- Logging misuse (root logger, excessive default DEBUG).
- Security: string-formatted SQL (risk of injection), shell=True with user input, secrets in code, weak crypto.
- Path handling: hard-coded paths instead of pathlib.
- Async: time.sleep in coroutines; creating event loops improperly; mixing blocking I/O in async paths.

Tools and quick profiling starters
- CPU: python -m cProfile -o prof.out your_script.py; visualize with snakeviz prof.out
- Line-level: pip install line_profiler; add @profile to hot funcs; kernprof -l -v your_script.py
- Memory: python -X tracemalloc your_script.py; or pip install memory_profiler and use @profile
- Sampling: py-spy top/record for low-overhead production profiling
- Lint/format: ruff, black, isort
- Types: mypy or pyright
- Security: bandit, pip-audit

Paste the code and I’ll give targeted, line-specific recommendations and optimized rewrites where helpful.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions. Please paste the Python code (or a minimal reproducible snippet), plus:

- Python version and OS
- Key libraries used
- Typical input sizes and where it feels slow
- Desired runtime/memory targets

Once you share it, I’ll deliver a structured review covering:
1) Performance bottlenecks, 2) Code quality issues, 3) Best-practice violations, with concrete fixes.

While you gather that, here’s a concise checklist you can run through:

Performance bottlenecks (common)
- Algorithmic complexity: nested loops over large data; repeated “x in list” lookups (use sets/dicts); unnecessary sorting.
- Recomputing work: add caching (functools.lru_cache), hoist invariants out of loops, precompile regexes.
- Data handling: avoid string concatenation in loops (use join/StringIO); batch I/O and DB operations; minimize per-row work in pandas (prefer vectorization/merge over apply/iterrows).
- Numerics: prefer NumPy vectorization; avoid Python loops in hot paths; use numba/Cython for tight loops if needed.
- Concurrency: CPU-bound → multiprocessing or native extensions; I/O-bound → asyncio/threading; avoid blocking calls in async code.
- External systems: N+1 DB queries; missing indexes; chatty HTTP without keep-alive/batching; excessive logging in hot paths.
- Memory: large intermediate lists; copy-heavy pandas ops; use generators/itertuples/chunked reads; specify dtypes.

Code quality issues
- Long functions, duplication, inconsistent naming, magic numbers.
- Missing type hints/docstrings; unclear interfaces.
- Broad exceptions, swallowed errors, no specific exception types.
- Mutable default args; global state; side effects in import time.
- Resource handling: not using context managers for files/sockets; unclosed sessions.
- Dead/unreachable code; debug prints instead of logging.

Best-practice violations
- No formatter/linter; lack of type checking (mypy/pyright).
- No virtualenv/requirements pinning; missing lock files.
- Logging misuse (root logger, excessive default DEBUG).
- Security: string-formatted SQL (risk of injection), shell=True with user input, secrets in code, weak crypto.
- Path handling: hard-coded paths instead of pathlib.
- Async: time.sleep in coroutines; creating event loops improperly; mixing blocking I/O in async paths.

Tools and quick profiling starters
- CPU: python -m cProfile -o prof.out your_script.py; visualize with snakeviz prof.out
- Line-level: pip install line_profiler; add @profile to hot funcs; kernprof -l -v your_script.py
- Memory: python -X tracemalloc your_script.py; or pip install memory_profiler and use @profile
- Sampling: py-spy top/record for low-overhead production profiling
- Lint/format: ruff, black, isort
- Types: mypy or pyright
- Security: bandit, pip-audit

Paste the code and I’ll give targeted, line-specific recommendations and optimized rewrites where helpful.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions. Please paste the Python code (or a minimal reproducible snippet), plus:

- Python version and OS
- Key libraries used
- Typical input sizes and where it feels slow
- Desired runtime/memory targets

Once you share it, I’ll deliver a structured review covering:
1) Performance bottlenecks, 2) Code quality issues, 3) Best-practice violations, with concrete fixes.

While you gather that, here’s a concise checklist you can run through:

Performance bottlenecks (common)
- Algorithmic complexity: nested loops over large data; repeated “x in list” lookups (use sets/dicts); unnecessary sorting.
- Recomputing work: add caching (functools.lru_cache), hoist invariants out of loops, precompile regexes.
- Data handling: avoid string concatenation in loops (use join/StringIO); batch I/O and DB operations; minimize per-row work in pandas (prefer vectorization/merge over apply/iterrows).
- Numerics: prefer NumPy vectorization; avoid Python loops in hot paths; use numba/Cython for tight loops if needed.
- Concurrency: CPU-bound → multiprocessing or native extensions; I/O-bound → asyncio/threading; avoid blocking calls in async code.
- External systems: N+1 DB queries; missing indexes; chatty HTTP without keep-alive/batching; excessive logging in hot paths.
- Memory: large intermediate lists; copy-heavy pandas ops; use generators/itertuples/chunked reads; specify dtypes.

Code quality issues
- Long functions, duplication, inconsistent naming, magic numbers.
- Missing type hints/docstrings; unclear interfaces.
- Broad exceptions, swallowed errors, no specific exception types.
- Mutable default args; global state; side effects in import time.
- Resource handling: not using context managers for files/sockets; unclosed sessions.
- Dead/unreachable code; debug prints instead of logging.

Best-practice violations
- No formatter/linter; lack of type checking (mypy/pyright).
- No virtualenv/requirements pinning; missing lock files.
- Logging misuse (root logger, excessive default DEBUG).
- Security: string-formatted SQL (risk of injection), shell=True with user input, secrets in code, weak crypto.
- Path handling: hard-coded paths instead of pathlib.
- Async: time.sleep in coroutines; creating event loops improperly; mixing blocking I/O in async paths.

Tools and quick profiling starters
- CPU: python -m cProfile -o prof.out your_script.py; visualize with snakeviz prof.out
- Line-level: pip install line_profiler; add @profile to hot funcs; kernprof -l -v your_script.py
- Memory: python -X tracemalloc your_script.py; or pip install memory_profiler and use @profile
- Sampling: py-spy top/record for low-overhead production profiling
- Lint/format: ruff, black, isort
- Types: mypy or pyright
- Security: bandit, pip-audit

Paste the code and I’ll give targeted, line-specific recommendations and optimized rewrites where helpful.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions. Please paste the Python code (or a minimal reproducible snippet), plus:

- Python version and OS
- Key libraries used
- Typical input sizes and where it feels slow
- Desired runtime/memory targets

Once you share it, I’ll deliver a structured review covering:
1) Performance bottlenecks, 2) Code quality issues, 3) Best-practice violations, with concrete fixes.

While you gather that, here’s a concise checklist you can run through:

Performance bottlenecks (common)
- Algorithmic complexity: nested loops over large data; repeated “x in list” lookups (use sets/dicts); unnecessary sorting.
- Recomputing work: add caching (functools.lru_cache), hoist invariants out of loops, precompile regexes.
- Data handling: avoid string concatenation in loops (use join/StringIO); batch I/O and DB operations; minimize per-row work in pandas (prefer vectorization/merge over apply/iterrows).
- Numerics: prefer NumPy vectorization; avoid Python loops in hot paths; use numba/Cython for tight loops if needed.
- Concurrency: CPU-bound → multiprocessing or native extensions; I/O-bound → asyncio/threading; avoid blocking calls in async code.
- External systems: N+1 DB queries; missing indexes; chatty HTTP without keep-alive/batching; excessive logging in hot paths.
- Memory: large intermediate lists; copy-heavy pandas ops; use generators/itertuples/chunked reads; specify dtypes.

Code quality issues
- Long functions, duplication, inconsistent naming, magic numbers.
- Missing type hints/docstrings; unclear interfaces.
- Broad exceptions, swallowed errors, no specific exception types.
- Mutable default args; global state; side effects in import time.
- Resource handling: not using context managers for files/sockets; unclosed sessions.
- Dead/unreachable code; debug prints instead of logging.

Best-practice violations
- No formatter/linter; lack of type checking (mypy/pyright).
- No virtualenv/requirements pinning; missing lock files.
- Logging misuse (root logger, excessive default DEBUG).
- Security: string-formatted SQL (risk of injection), shell=True with user input, secrets in code, weak crypto.
- Path handling: hard-coded paths instead of pathlib.
- Async: time.sleep in coroutines; creating event loops improperly; mixing blocking I/O in async paths.

Tools and quick profiling starters
- CPU: python -m cProfile -o prof.out your_script.py; visualize with snakeviz prof.out
- Line-level: pip install line_profiler; add @profile to hot funcs; kernprof -l -v your_script.py
- Memory: python -X tracemalloc your_script.py; or pip install memory_profiler and use @profile
- Sampling: py-spy top/record for low-overhead production profiling
- Lint/format: ruff, black, isort
- Types: mypy or pyright
- Security: bandit, pip-audit

Paste the code and I’ll give targeted, line-specific recommendations and optimized rewrites where helpful.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions. Please paste the Python code (or a minimal reproducible snippet), plus:

- Python version and OS
- Key libraries used
- Typical input sizes and where it feels slow
- Desired runtime/memory targets

Once you share it, I’ll deliver a structured review covering:
1) Performance bottlenecks, 2) Code quality issues, 3) Best-practice violations, with concrete fixes.

While you gather that, here’s a concise checklist you can run through:

Performance bottlenecks (common)
- Algorithmic complexity: nested loops over large data; repeated “x in list” lookups (use sets/dicts); unnecessary sorting.
- Recomputing work: add caching (functools.lru_cache), hoist invariants out of loops, precompile regexes.
- Data handling: avoid string concatenation in loops (use join/StringIO); batch I/O and DB operations; minimize per-row work in pandas (prefer vectorization/merge over apply/iterrows).
- Numerics: prefer NumPy vectorization; avoid Python loops in hot paths; use numba/Cython for tight loops if needed.
- Concurrency: CPU-bound → multiprocessing or native extensions; I/O-bound → asyncio/threading; avoid blocking calls in async code.
- External systems: N+1 DB queries; missing indexes; chatty HTTP without keep-alive/batching; excessive logging in hot paths.
- Memory: large intermediate lists; copy-heavy pandas ops; use generators/itertuples/chunked reads; specify dtypes.

Code quality issues
- Long functions, duplication, inconsistent naming, magic numbers.
- Missing type hints/docstrings; unclear interfaces.
- Broad exceptions, swallowed errors, no specific exception types.
- Mutable default args; global state; side effects in import time.
- Resource handling: not using context managers for files/sockets; unclosed sessions.
- Dead/unreachable code; debug prints instead of logging.

Best-practice violations
- No formatter/linter; lack of type checking (mypy/pyright).
- No virtualenv/requirements pinning; missing lock files.
- Logging misuse (root logger, excessive default DEBUG).
- Security: string-formatted SQL (risk of injection), shell=True with user input, secrets in code, weak crypto.
- Path handling: hard-coded paths instead of pathlib.
- Async: time.sleep in coroutines; creating event loops improperly; mixing blocking I/O in async paths.

Tools and quick profiling starters
- CPU: python -m cProfile -o prof.out your_script.py; visualize with snakeviz prof.out
- Line-level: pip install line_profiler; add @profile to hot funcs; kernprof -l -v your_script.py
- Memory: python -X tracemalloc your_script.py; or pip install memory_profiler and use @profile
- Sampling: py-spy top/record for low-overhead production profiling
- Lint/format: ruff, black, isort
- Types: mypy or pyright
- Security: bandit, pip-audit

Paste the code and I’ll give targeted, line-specific recommendations and optimized rewrites where helpful.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions. Please paste the Python code (or a minimal reproducible snippet), plus:

- Python version and OS
- Key libraries used
- Typical input sizes and where it feels slow
- Desired runtime/memory targets

Once you share it, I’ll deliver a structured review covering:
1) Performance bottlenecks, 2) Code quality issues, 3) Best-practice violations, with concrete fixes.

While you gather that, here’s a concise checklist you can run through:

Performance bottlenecks (common)
- Algorithmic complexity: nested loops over large data; repeated “x in list” lookups (use sets/dicts); unnecessary sorting.
- Recomputing work: add caching (functools.lru_cache), hoist invariants out of loops, precompile regexes.
- Data handling: avoid string concatenation in loops (use join/StringIO); batch I/O and DB operations; minimize per-row work in pandas (prefer vectorization/merge over apply/iterrows).
- Numerics: prefer NumPy vectorization; avoid Python loops in hot paths; use numba/Cython for tight loops if needed.
- Concurrency: CPU-bound → multiprocessing or native extensions; I/O-bound → asyncio/threading; avoid blocking calls in async code.
- External systems: N+1 DB queries; missing indexes; chatty HTTP without keep-alive/batching; excessive logging in hot paths.
- Memory: large intermediate lists; copy-heavy pandas ops; use generators/itertuples/chunked reads; specify dtypes.

Code quality issues
- Long functions, duplication, inconsistent naming, magic numbers.
- Missing type hints/docstrings; unclear interfaces.
- Broad exceptions, swallowed errors, no specific exception types.
- Mutable default args; global state; side effects in import time.
- Resource handling: not using context managers for files/sockets; unclosed sessions.
- Dead/unreachable code; debug prints instead of logging.

Best-practice violations
- No formatter/linter; lack of type checking (mypy/pyright).
- No virtualenv/requirements pinning; missing lock files.
- Logging misuse (root logger, excessive default DEBUG).
- Security: string-formatted SQL (risk of injection), shell=True with user input, secrets in code, weak crypto.
- Path handling: hard-coded paths instead of pathlib.
- Async: time.sleep in coroutines; creating event loops improperly; mixing blocking I/O in async paths.

Tools and quick profiling starters
- CPU: python -m cProfile -o prof.out your_script.py; visualize with snakeviz prof.out
- Line-level: pip install line_profiler; add @profile to hot funcs; kernprof -l -v your_script.py
- Memory: python -X tracemalloc your_script.py; or pip install memory_profiler and use @profile
- Sampling: py-spy top/record for low-overhead production profiling
- Lint/format: ruff, black, isort
- Types: mypy or pyright
- Security: bandit, pip-audit

Paste the code and I’ll give targeted, line-specific recommendations and optimized rewrites where helpful.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions. Please paste the Python code (or a minimal reproducible snippet), plus:

- Python version and OS
- Key libraries used
- Typical input sizes and where it feels slow
- Desired runtime/memory targets

Once you share it, I’ll deliver a structured review covering:
1) Performance bottlenecks, 2) Code quality issues, 3) Best-practice violations, with concrete fixes.

While you gather that, here’s a concise checklist you can run through:

Performance bottlenecks (common)
- Algorithmic complexity: nested loops over large data; repeated “x in list” lookups (use sets/dicts); unnecessary sorting.
- Recomputing work: add caching (functools.lru_cache), hoist invariants out of loops, precompile regexes.
- Data handling: avoid string concatenation in loops (use join/StringIO); batch I/O and DB operations; minimize per-row work in pandas (prefer vectorization/merge over apply/iterrows).
- Numerics: prefer NumPy vectorization; avoid Python loops in hot paths; use numba/Cython for tight loops if needed.
- Concurrency: CPU-bound → multiprocessing or native extensions; I/O-bound → asyncio/threading; avoid blocking calls in async code.
- External systems: N+1 DB queries; missing indexes; chatty HTTP without keep-alive/batching; excessive logging in hot paths.
- Memory: large intermediate lists; copy-heavy pandas ops; use generators/itertuples/chunked reads; specify dtypes.

Code quality issues
- Long functions, duplication, inconsistent naming, magic numbers.
- Missing type hints/docstrings; unclear interfaces.
- Broad exceptions, swallowed errors, no specific exception types.
- Mutable default args; global state; side effects in import time.
- Resource handling: not using context managers for files/sockets; unclosed sessions.
- Dead/unreachable code; debug prints instead of logging.

Best-practice violations
- No formatter/linter; lack of type checking (mypy/pyright).
- No virtualenv/requirements pinning; missing lock files.
- Logging misuse (root logger, excessive default DEBUG).
- Security: string-formatted SQL (risk of injection), shell=True with user input, secrets in code, weak crypto.
- Path handling: hard-coded paths instead of pathlib.
- Async: time.sleep in coroutines; creating event loops improperly; mixing blocking I/O in async paths.

Tools and quick profiling starters
- CPU: python -m cProfile -o prof.out your_script.py; visualize with snakeviz prof.out
- Line-level: pip install line_profiler; add @profile to hot funcs; kernprof -l -v your_script.py
- Memory: python -X tracemalloc your_script.py; or pip install memory_profiler and use @profile
- Sampling: py-spy top/record for low-overhead production profiling
- Lint/format: ruff, black, isort
- Types: mypy or pyright
- Security: bandit, pip-audit

Paste the code and I’ll give targeted, line-specific recommendations and optimized rewrites where helpful.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions. Please paste the Python code (or a minimal reproducible snippet), plus:

- Python version and OS
- Key libraries used
- Typical input sizes and where it feels slow
- Desired runtime/memory targets

Once you share it, I’ll deliver a structured review covering:
1) Performance bottlenecks, 2) Code quality issues, 3) Best-practice violations, with concrete fixes.

While you gather that, here’s a concise checklist you can run through:

Performance bottlenecks (common)
- Algorithmic complexity: nested loops over large data; repeated “x in list” lookups (use sets/dicts); unnecessary sorting.
- Recomputing work: add caching (functools.lru_cache), hoist invariants out of loops, precompile regexes.
- Data handling: avoid string concatenation in loops (use join/StringIO); batch I/O and DB operations; minimize per-row work in pandas (prefer vectorization/merge over apply/iterrows).
- Numerics: prefer NumPy vectorization; avoid Python loops in hot paths; use numba/Cython for tight loops if needed.
- Concurrency: CPU-bound → multiprocessing or native extensions; I/O-bound → asyncio/threading; avoid blocking calls in async code.
- External systems: N+1 DB queries; missing indexes; chatty HTTP without keep-alive/batching; excessive logging in hot paths.
- Memory: large intermediate lists; copy-heavy pandas ops; use generators/itertuples/chunked reads; specify dtypes.

Code quality issues
- Long functions, duplication, inconsistent naming, magic numbers.
- Missing type hints/docstrings; unclear interfaces.
- Broad exceptions, swallowed errors, no specific exception types.
- Mutable default args; global state; side effects in import time.
- Resource handling: not using context managers for files/sockets; unclosed sessions.
- Dead/unreachable code; debug prints instead of logging.

Best-practice violations
- No formatter/linter; lack of type checking (mypy/pyright).
- No virtualenv/requirements pinning; missing lock files.
- Logging misuse (root logger, excessive default DEBUG).
- Security: string-formatted SQL (risk of injection), shell=True with user input, secrets in code, weak crypto.
- Path handling: hard-coded paths instead of pathlib.
- Async: time.sleep in coroutines; creating event loops improperly; mixing blocking I/O in async paths.

Tools and quick profiling starters
- CPU: python -m cProfile -o prof.out your_script.py; visualize with snakeviz prof.out
- Line-level: pip install line_profiler; add @profile to hot funcs; kernprof -l -v your_script.py
- Memory: python -X tracemalloc your_script.py; or pip install memory_profiler and use @profile
- Sampling: py-spy top/record for low-overhead production profiling
- Lint/format: ruff, black, isort
- Types: mypy or pyright
- Security: bandit, pip-audit

Paste the code and I’ll give targeted, line-specific recommendations and optimized rewrites where helpful.

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions. Please paste the Python code (or a minimal reproducible snippet), plus:

- Python version and OS
- Key libraries used
- Typical input sizes and where it feels slow
- Desired runtime/memory targets

Once you share it, I’ll deliver a structured review covering:
1) Performance bottlenecks, 2) Code quality issues, 3) Best-practice violations, with concrete fixes.

While you gather that, here’s a concise checklist you can run through:

Performance bottlenecks (common)
- Algorithmic complexity: nested loops over large data; repeated “x in list” lookups (use sets/dicts); unnecessary sorting.
- Recomputing work: add caching (functools.lru_cache), hoist invariants out of loops, precompile regexes.
- Data handling: avoid string concatenation in loops (use join/StringIO); batch I/O and DB operations; minimize per-row work in pandas (prefer vectorization/merge over apply/iterrows).
- Numerics: prefer NumPy vectorization; avoid Python loops in hot paths; use numba/Cython for tight loops if needed.
- Concurrency: CPU-bound → multiprocessing or native extensions; I/O-bound → asyncio/threading; avoid blocking calls in async code.
- External systems: N+1 DB queries; missing indexes; chatty HTTP without keep-alive/batching; excessive logging in hot paths.
- Memory: large intermediate lists; copy-heavy pandas ops; use generators/itertuples/chunked reads; specify dtypes.

Code quality issues
- Long functions, duplication, inconsistent naming, magic numbers.
- Missing type hints/docstrings; unclear interfaces.
- Broad exceptions, swallowed errors, no specific exception types.
- Mutable default args; global state; side effects in import time.
- Resource handling: not using context managers for files/sockets; unclosed sessions.
- Dead/unreachable code; debug prints instead of logging.

Best-practice violations
- No formatter/linter; lack of type checking (mypy/pyright).
- No virtualenv/requirements pinning; missing lock files.
- Logging misuse (root logger, excessive default DEBUG).
- Security: string-formatted SQL (risk of injection), shell=True with user input, secrets in code, weak crypto.
- Path handling: hard-coded paths instead of pathlib.
- Async: time.sleep in coroutines; creating event loops improperly; mixing blocking I/O in async paths.

Tools and quick profiling starters
- CPU: python -m cProfile -o prof.out your_script.py; visualize with snakeviz prof.out
- Line-level: pip install line_profiler; add @profile to hot funcs; kernprof -l -v your_script.py
- Memory: python -X tracemalloc your_script.py; or pip install memory_profiler and use @profile
- Sampling: py-spy top/record for low-overhead production profiling
- Lint/format: ruff, black, isort
- Types: mypy or pyright
- Security: bandit, pip-audit

Paste the code and I’ll give targeted, line-specific recommendations and optimized rewrites where helpful.

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