# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:23:03

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:23:03

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:23:03

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:23:03

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:23:03

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:23:03

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:23:03

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:23:03

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:23:03

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:23:03

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:23:03

## Analysis Results

It looks like the code block contains a placeholder (“# Code Optimization Report

**File:** {{code_file}}
**Language:** python
**Date:** 2025-08-28 00:23:03

## Analysis Results

It looks like the code block contains a placeholder (“{{content}}”), so I don’t have the actual Python code to review.

Please paste the code (or share a repo/gist) and, if possible, include:
- Python version and key libraries
- Typical input sizes and performance goals (e.g., latency/throughput/memory targets)
- Where you suspect problems (functions/files) and how to run the code/tests

While you gather that, here’s how I will assess it and what you can pre-run:

What I will identify
1) Performance bottlenecks
- Hot paths via profiling (CPU: cProfile/pyinstrument; sampling: py-spy; memory: memray/memory_profiler)
- Algorithmic complexity and data structure choices
- Inefficient Python loops where vectorization/NumPy/C-extensions help
- Repeated work (recomputations, repeated regex/JSON parsing, poor caching)
- I/O patterns (chatty FS/network, N+1 DB queries, synchronous waits, small-buffer reads/writes)
- Concurrency issues (GIL-bound CPU work in threads, missing multiprocessing/numba, blocking in asyncio)
- Pandas/NumPy pitfalls (row-by-row applies, SettingWithCopy, non-inplace ops that copy large arrays)
- Serialization and IPC overhead; unnecessary conversions

2) Code quality issues
- Readability, naming, duplication, long functions, high cyclomatic complexity
- Error handling (broad except, swallowed exceptions, missing retries/backoff)
- Resource management (missing context managers for files/sockets/processes)
- Logging quality and log levels; excessive string formatting in hot paths
- Dead code, unused imports, magic numbers, poor configuration handling
- Lack of tests or type hints; fragile APIs; circular dependencies

3) Best practice violations
- PEP 8/257 conventions; inconsistent formatting (recommend black + ruff)
- Security hazards: eval/exec, subprocess with shell=True, yaml.load without SafeLoader, pickle on untrusted data, verify=False, hardcoded secrets
- DB/SQL not parameterized; missing timeouts/retries; insecure randomness for crypto
- Timezones (naive datetimes), path handling without pathlib, non-deterministic behavior
- Packaging/env: unpinned deps, missing constraints, no __main__/entry points
- Async misuse (blocking calls in coroutines), thread/process resource cleanup

Quick commands you can run and share results
- CPU profile: python -m cProfile -o prof.out your_script.py; then snakeviz prof.out
- Sampling profile (low overhead): py-spy top -- python your_script.py
- Memory profile: pip install memory_profiler; add @profile and run mprof run python your_script.py; mprof plot
- Static checks: pip install ruff mypy bandit; run:
  - ruff check .
  - mypy .
  - bandit -r .
- Complexity: pip install radon; radon cc -s -a .

If you provide the code or profiling outputs, I’ll produce a focused list of:
- Concrete bottlenecks with evidence and optimized alternatives
- Specific code quality fixes (file:line) with rationale
- Best practice violations and remediation steps
- A prioritized “quick wins” list and longer-term refactors

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”), so I don’t have the actual Python code to review.

Please paste the code (or share a repo/gist) and, if possible, include:
- Python version and key libraries
- Typical input sizes and performance goals (e.g., latency/throughput/memory targets)
- Where you suspect problems (functions/files) and how to run the code/tests

While you gather that, here’s how I will assess it and what you can pre-run:

What I will identify
1) Performance bottlenecks
- Hot paths via profiling (CPU: cProfile/pyinstrument; sampling: py-spy; memory: memray/memory_profiler)
- Algorithmic complexity and data structure choices
- Inefficient Python loops where vectorization/NumPy/C-extensions help
- Repeated work (recomputations, repeated regex/JSON parsing, poor caching)
- I/O patterns (chatty FS/network, N+1 DB queries, synchronous waits, small-buffer reads/writes)
- Concurrency issues (GIL-bound CPU work in threads, missing multiprocessing/numba, blocking in asyncio)
- Pandas/NumPy pitfalls (row-by-row applies, SettingWithCopy, non-inplace ops that copy large arrays)
- Serialization and IPC overhead; unnecessary conversions

2) Code quality issues
- Readability, naming, duplication, long functions, high cyclomatic complexity
- Error handling (broad except, swallowed exceptions, missing retries/backoff)
- Resource management (missing context managers for files/sockets/processes)
- Logging quality and log levels; excessive string formatting in hot paths
- Dead code, unused imports, magic numbers, poor configuration handling
- Lack of tests or type hints; fragile APIs; circular dependencies

3) Best practice violations
- PEP 8/257 conventions; inconsistent formatting (recommend black + ruff)
- Security hazards: eval/exec, subprocess with shell=True, yaml.load without SafeLoader, pickle on untrusted data, verify=False, hardcoded secrets
- DB/SQL not parameterized; missing timeouts/retries; insecure randomness for crypto
- Timezones (naive datetimes), path handling without pathlib, non-deterministic behavior
- Packaging/env: unpinned deps, missing constraints, no __main__/entry points
- Async misuse (blocking calls in coroutines), thread/process resource cleanup

Quick commands you can run and share results
- CPU profile: python -m cProfile -o prof.out your_script.py; then snakeviz prof.out
- Sampling profile (low overhead): py-spy top -- python your_script.py
- Memory profile: pip install memory_profiler; add @profile and run mprof run python your_script.py; mprof plot
- Static checks: pip install ruff mypy bandit; run:
  - ruff check .
  - mypy .
  - bandit -r .
- Complexity: pip install radon; radon cc -s -a .

If you provide the code or profiling outputs, I’ll produce a focused list of:
- Concrete bottlenecks with evidence and optimized alternatives
- Specific code quality fixes (file:line) with rationale
- Best practice violations and remediation steps
- A prioritized “quick wins” list and longer-term refactors

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”), so I don’t have the actual Python code to review.

Please paste the code (or share a repo/gist) and, if possible, include:
- Python version and key libraries
- Typical input sizes and performance goals (e.g., latency/throughput/memory targets)
- Where you suspect problems (functions/files) and how to run the code/tests

While you gather that, here’s how I will assess it and what you can pre-run:

What I will identify
1) Performance bottlenecks
- Hot paths via profiling (CPU: cProfile/pyinstrument; sampling: py-spy; memory: memray/memory_profiler)
- Algorithmic complexity and data structure choices
- Inefficient Python loops where vectorization/NumPy/C-extensions help
- Repeated work (recomputations, repeated regex/JSON parsing, poor caching)
- I/O patterns (chatty FS/network, N+1 DB queries, synchronous waits, small-buffer reads/writes)
- Concurrency issues (GIL-bound CPU work in threads, missing multiprocessing/numba, blocking in asyncio)
- Pandas/NumPy pitfalls (row-by-row applies, SettingWithCopy, non-inplace ops that copy large arrays)
- Serialization and IPC overhead; unnecessary conversions

2) Code quality issues
- Readability, naming, duplication, long functions, high cyclomatic complexity
- Error handling (broad except, swallowed exceptions, missing retries/backoff)
- Resource management (missing context managers for files/sockets/processes)
- Logging quality and log levels; excessive string formatting in hot paths
- Dead code, unused imports, magic numbers, poor configuration handling
- Lack of tests or type hints; fragile APIs; circular dependencies

3) Best practice violations
- PEP 8/257 conventions; inconsistent formatting (recommend black + ruff)
- Security hazards: eval/exec, subprocess with shell=True, yaml.load without SafeLoader, pickle on untrusted data, verify=False, hardcoded secrets
- DB/SQL not parameterized; missing timeouts/retries; insecure randomness for crypto
- Timezones (naive datetimes), path handling without pathlib, non-deterministic behavior
- Packaging/env: unpinned deps, missing constraints, no __main__/entry points
- Async misuse (blocking calls in coroutines), thread/process resource cleanup

Quick commands you can run and share results
- CPU profile: python -m cProfile -o prof.out your_script.py; then snakeviz prof.out
- Sampling profile (low overhead): py-spy top -- python your_script.py
- Memory profile: pip install memory_profiler; add @profile and run mprof run python your_script.py; mprof plot
- Static checks: pip install ruff mypy bandit; run:
  - ruff check .
  - mypy .
  - bandit -r .
- Complexity: pip install radon; radon cc -s -a .

If you provide the code or profiling outputs, I’ll produce a focused list of:
- Concrete bottlenecks with evidence and optimized alternatives
- Specific code quality fixes (file:line) with rationale
- Best practice violations and remediation steps
- A prioritized “quick wins” list and longer-term refactors

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”), so I don’t have the actual Python code to review.

Please paste the code (or share a repo/gist) and, if possible, include:
- Python version and key libraries
- Typical input sizes and performance goals (e.g., latency/throughput/memory targets)
- Where you suspect problems (functions/files) and how to run the code/tests

While you gather that, here’s how I will assess it and what you can pre-run:

What I will identify
1) Performance bottlenecks
- Hot paths via profiling (CPU: cProfile/pyinstrument; sampling: py-spy; memory: memray/memory_profiler)
- Algorithmic complexity and data structure choices
- Inefficient Python loops where vectorization/NumPy/C-extensions help
- Repeated work (recomputations, repeated regex/JSON parsing, poor caching)
- I/O patterns (chatty FS/network, N+1 DB queries, synchronous waits, small-buffer reads/writes)
- Concurrency issues (GIL-bound CPU work in threads, missing multiprocessing/numba, blocking in asyncio)
- Pandas/NumPy pitfalls (row-by-row applies, SettingWithCopy, non-inplace ops that copy large arrays)
- Serialization and IPC overhead; unnecessary conversions

2) Code quality issues
- Readability, naming, duplication, long functions, high cyclomatic complexity
- Error handling (broad except, swallowed exceptions, missing retries/backoff)
- Resource management (missing context managers for files/sockets/processes)
- Logging quality and log levels; excessive string formatting in hot paths
- Dead code, unused imports, magic numbers, poor configuration handling
- Lack of tests or type hints; fragile APIs; circular dependencies

3) Best practice violations
- PEP 8/257 conventions; inconsistent formatting (recommend black + ruff)
- Security hazards: eval/exec, subprocess with shell=True, yaml.load without SafeLoader, pickle on untrusted data, verify=False, hardcoded secrets
- DB/SQL not parameterized; missing timeouts/retries; insecure randomness for crypto
- Timezones (naive datetimes), path handling without pathlib, non-deterministic behavior
- Packaging/env: unpinned deps, missing constraints, no __main__/entry points
- Async misuse (blocking calls in coroutines), thread/process resource cleanup

Quick commands you can run and share results
- CPU profile: python -m cProfile -o prof.out your_script.py; then snakeviz prof.out
- Sampling profile (low overhead): py-spy top -- python your_script.py
- Memory profile: pip install memory_profiler; add @profile and run mprof run python your_script.py; mprof plot
- Static checks: pip install ruff mypy bandit; run:
  - ruff check .
  - mypy .
  - bandit -r .
- Complexity: pip install radon; radon cc -s -a .

If you provide the code or profiling outputs, I’ll produce a focused list of:
- Concrete bottlenecks with evidence and optimized alternatives
- Specific code quality fixes (file:line) with rationale
- Best practice violations and remediation steps
- A prioritized “quick wins” list and longer-term refactors

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”), so I don’t have the actual Python code to review.

Please paste the code (or share a repo/gist) and, if possible, include:
- Python version and key libraries
- Typical input sizes and performance goals (e.g., latency/throughput/memory targets)
- Where you suspect problems (functions/files) and how to run the code/tests

While you gather that, here’s how I will assess it and what you can pre-run:

What I will identify
1) Performance bottlenecks
- Hot paths via profiling (CPU: cProfile/pyinstrument; sampling: py-spy; memory: memray/memory_profiler)
- Algorithmic complexity and data structure choices
- Inefficient Python loops where vectorization/NumPy/C-extensions help
- Repeated work (recomputations, repeated regex/JSON parsing, poor caching)
- I/O patterns (chatty FS/network, N+1 DB queries, synchronous waits, small-buffer reads/writes)
- Concurrency issues (GIL-bound CPU work in threads, missing multiprocessing/numba, blocking in asyncio)
- Pandas/NumPy pitfalls (row-by-row applies, SettingWithCopy, non-inplace ops that copy large arrays)
- Serialization and IPC overhead; unnecessary conversions

2) Code quality issues
- Readability, naming, duplication, long functions, high cyclomatic complexity
- Error handling (broad except, swallowed exceptions, missing retries/backoff)
- Resource management (missing context managers for files/sockets/processes)
- Logging quality and log levels; excessive string formatting in hot paths
- Dead code, unused imports, magic numbers, poor configuration handling
- Lack of tests or type hints; fragile APIs; circular dependencies

3) Best practice violations
- PEP 8/257 conventions; inconsistent formatting (recommend black + ruff)
- Security hazards: eval/exec, subprocess with shell=True, yaml.load without SafeLoader, pickle on untrusted data, verify=False, hardcoded secrets
- DB/SQL not parameterized; missing timeouts/retries; insecure randomness for crypto
- Timezones (naive datetimes), path handling without pathlib, non-deterministic behavior
- Packaging/env: unpinned deps, missing constraints, no __main__/entry points
- Async misuse (blocking calls in coroutines), thread/process resource cleanup

Quick commands you can run and share results
- CPU profile: python -m cProfile -o prof.out your_script.py; then snakeviz prof.out
- Sampling profile (low overhead): py-spy top -- python your_script.py
- Memory profile: pip install memory_profiler; add @profile and run mprof run python your_script.py; mprof plot
- Static checks: pip install ruff mypy bandit; run:
  - ruff check .
  - mypy .
  - bandit -r .
- Complexity: pip install radon; radon cc -s -a .

If you provide the code or profiling outputs, I’ll produce a focused list of:
- Concrete bottlenecks with evidence and optimized alternatives
- Specific code quality fixes (file:line) with rationale
- Best practice violations and remediation steps
- A prioritized “quick wins” list and longer-term refactors

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”), so I don’t have the actual Python code to review.

Please paste the code (or share a repo/gist) and, if possible, include:
- Python version and key libraries
- Typical input sizes and performance goals (e.g., latency/throughput/memory targets)
- Where you suspect problems (functions/files) and how to run the code/tests

While you gather that, here’s how I will assess it and what you can pre-run:

What I will identify
1) Performance bottlenecks
- Hot paths via profiling (CPU: cProfile/pyinstrument; sampling: py-spy; memory: memray/memory_profiler)
- Algorithmic complexity and data structure choices
- Inefficient Python loops where vectorization/NumPy/C-extensions help
- Repeated work (recomputations, repeated regex/JSON parsing, poor caching)
- I/O patterns (chatty FS/network, N+1 DB queries, synchronous waits, small-buffer reads/writes)
- Concurrency issues (GIL-bound CPU work in threads, missing multiprocessing/numba, blocking in asyncio)
- Pandas/NumPy pitfalls (row-by-row applies, SettingWithCopy, non-inplace ops that copy large arrays)
- Serialization and IPC overhead; unnecessary conversions

2) Code quality issues
- Readability, naming, duplication, long functions, high cyclomatic complexity
- Error handling (broad except, swallowed exceptions, missing retries/backoff)
- Resource management (missing context managers for files/sockets/processes)
- Logging quality and log levels; excessive string formatting in hot paths
- Dead code, unused imports, magic numbers, poor configuration handling
- Lack of tests or type hints; fragile APIs; circular dependencies

3) Best practice violations
- PEP 8/257 conventions; inconsistent formatting (recommend black + ruff)
- Security hazards: eval/exec, subprocess with shell=True, yaml.load without SafeLoader, pickle on untrusted data, verify=False, hardcoded secrets
- DB/SQL not parameterized; missing timeouts/retries; insecure randomness for crypto
- Timezones (naive datetimes), path handling without pathlib, non-deterministic behavior
- Packaging/env: unpinned deps, missing constraints, no __main__/entry points
- Async misuse (blocking calls in coroutines), thread/process resource cleanup

Quick commands you can run and share results
- CPU profile: python -m cProfile -o prof.out your_script.py; then snakeviz prof.out
- Sampling profile (low overhead): py-spy top -- python your_script.py
- Memory profile: pip install memory_profiler; add @profile and run mprof run python your_script.py; mprof plot
- Static checks: pip install ruff mypy bandit; run:
  - ruff check .
  - mypy .
  - bandit -r .
- Complexity: pip install radon; radon cc -s -a .

If you provide the code or profiling outputs, I’ll produce a focused list of:
- Concrete bottlenecks with evidence and optimized alternatives
- Specific code quality fixes (file:line) with rationale
- Best practice violations and remediation steps
- A prioritized “quick wins” list and longer-term refactors

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”), so I don’t have the actual Python code to review.

Please paste the code (or share a repo/gist) and, if possible, include:
- Python version and key libraries
- Typical input sizes and performance goals (e.g., latency/throughput/memory targets)
- Where you suspect problems (functions/files) and how to run the code/tests

While you gather that, here’s how I will assess it and what you can pre-run:

What I will identify
1) Performance bottlenecks
- Hot paths via profiling (CPU: cProfile/pyinstrument; sampling: py-spy; memory: memray/memory_profiler)
- Algorithmic complexity and data structure choices
- Inefficient Python loops where vectorization/NumPy/C-extensions help
- Repeated work (recomputations, repeated regex/JSON parsing, poor caching)
- I/O patterns (chatty FS/network, N+1 DB queries, synchronous waits, small-buffer reads/writes)
- Concurrency issues (GIL-bound CPU work in threads, missing multiprocessing/numba, blocking in asyncio)
- Pandas/NumPy pitfalls (row-by-row applies, SettingWithCopy, non-inplace ops that copy large arrays)
- Serialization and IPC overhead; unnecessary conversions

2) Code quality issues
- Readability, naming, duplication, long functions, high cyclomatic complexity
- Error handling (broad except, swallowed exceptions, missing retries/backoff)
- Resource management (missing context managers for files/sockets/processes)
- Logging quality and log levels; excessive string formatting in hot paths
- Dead code, unused imports, magic numbers, poor configuration handling
- Lack of tests or type hints; fragile APIs; circular dependencies

3) Best practice violations
- PEP 8/257 conventions; inconsistent formatting (recommend black + ruff)
- Security hazards: eval/exec, subprocess with shell=True, yaml.load without SafeLoader, pickle on untrusted data, verify=False, hardcoded secrets
- DB/SQL not parameterized; missing timeouts/retries; insecure randomness for crypto
- Timezones (naive datetimes), path handling without pathlib, non-deterministic behavior
- Packaging/env: unpinned deps, missing constraints, no __main__/entry points
- Async misuse (blocking calls in coroutines), thread/process resource cleanup

Quick commands you can run and share results
- CPU profile: python -m cProfile -o prof.out your_script.py; then snakeviz prof.out
- Sampling profile (low overhead): py-spy top -- python your_script.py
- Memory profile: pip install memory_profiler; add @profile and run mprof run python your_script.py; mprof plot
- Static checks: pip install ruff mypy bandit; run:
  - ruff check .
  - mypy .
  - bandit -r .
- Complexity: pip install radon; radon cc -s -a .

If you provide the code or profiling outputs, I’ll produce a focused list of:
- Concrete bottlenecks with evidence and optimized alternatives
- Specific code quality fixes (file:line) with rationale
- Best practice violations and remediation steps
- A prioritized “quick wins” list and longer-term refactors

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”), so I don’t have the actual Python code to review.

Please paste the code (or share a repo/gist) and, if possible, include:
- Python version and key libraries
- Typical input sizes and performance goals (e.g., latency/throughput/memory targets)
- Where you suspect problems (functions/files) and how to run the code/tests

While you gather that, here’s how I will assess it and what you can pre-run:

What I will identify
1) Performance bottlenecks
- Hot paths via profiling (CPU: cProfile/pyinstrument; sampling: py-spy; memory: memray/memory_profiler)
- Algorithmic complexity and data structure choices
- Inefficient Python loops where vectorization/NumPy/C-extensions help
- Repeated work (recomputations, repeated regex/JSON parsing, poor caching)
- I/O patterns (chatty FS/network, N+1 DB queries, synchronous waits, small-buffer reads/writes)
- Concurrency issues (GIL-bound CPU work in threads, missing multiprocessing/numba, blocking in asyncio)
- Pandas/NumPy pitfalls (row-by-row applies, SettingWithCopy, non-inplace ops that copy large arrays)
- Serialization and IPC overhead; unnecessary conversions

2) Code quality issues
- Readability, naming, duplication, long functions, high cyclomatic complexity
- Error handling (broad except, swallowed exceptions, missing retries/backoff)
- Resource management (missing context managers for files/sockets/processes)
- Logging quality and log levels; excessive string formatting in hot paths
- Dead code, unused imports, magic numbers, poor configuration handling
- Lack of tests or type hints; fragile APIs; circular dependencies

3) Best practice violations
- PEP 8/257 conventions; inconsistent formatting (recommend black + ruff)
- Security hazards: eval/exec, subprocess with shell=True, yaml.load without SafeLoader, pickle on untrusted data, verify=False, hardcoded secrets
- DB/SQL not parameterized; missing timeouts/retries; insecure randomness for crypto
- Timezones (naive datetimes), path handling without pathlib, non-deterministic behavior
- Packaging/env: unpinned deps, missing constraints, no __main__/entry points
- Async misuse (blocking calls in coroutines), thread/process resource cleanup

Quick commands you can run and share results
- CPU profile: python -m cProfile -o prof.out your_script.py; then snakeviz prof.out
- Sampling profile (low overhead): py-spy top -- python your_script.py
- Memory profile: pip install memory_profiler; add @profile and run mprof run python your_script.py; mprof plot
- Static checks: pip install ruff mypy bandit; run:
  - ruff check .
  - mypy .
  - bandit -r .
- Complexity: pip install radon; radon cc -s -a .

If you provide the code or profiling outputs, I’ll produce a focused list of:
- Concrete bottlenecks with evidence and optimized alternatives
- Specific code quality fixes (file:line) with rationale
- Best practice violations and remediation steps
- A prioritized “quick wins” list and longer-term refactors

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”), so I don’t have the actual Python code to review.

Please paste the code (or share a repo/gist) and, if possible, include:
- Python version and key libraries
- Typical input sizes and performance goals (e.g., latency/throughput/memory targets)
- Where you suspect problems (functions/files) and how to run the code/tests

While you gather that, here’s how I will assess it and what you can pre-run:

What I will identify
1) Performance bottlenecks
- Hot paths via profiling (CPU: cProfile/pyinstrument; sampling: py-spy; memory: memray/memory_profiler)
- Algorithmic complexity and data structure choices
- Inefficient Python loops where vectorization/NumPy/C-extensions help
- Repeated work (recomputations, repeated regex/JSON parsing, poor caching)
- I/O patterns (chatty FS/network, N+1 DB queries, synchronous waits, small-buffer reads/writes)
- Concurrency issues (GIL-bound CPU work in threads, missing multiprocessing/numba, blocking in asyncio)
- Pandas/NumPy pitfalls (row-by-row applies, SettingWithCopy, non-inplace ops that copy large arrays)
- Serialization and IPC overhead; unnecessary conversions

2) Code quality issues
- Readability, naming, duplication, long functions, high cyclomatic complexity
- Error handling (broad except, swallowed exceptions, missing retries/backoff)
- Resource management (missing context managers for files/sockets/processes)
- Logging quality and log levels; excessive string formatting in hot paths
- Dead code, unused imports, magic numbers, poor configuration handling
- Lack of tests or type hints; fragile APIs; circular dependencies

3) Best practice violations
- PEP 8/257 conventions; inconsistent formatting (recommend black + ruff)
- Security hazards: eval/exec, subprocess with shell=True, yaml.load without SafeLoader, pickle on untrusted data, verify=False, hardcoded secrets
- DB/SQL not parameterized; missing timeouts/retries; insecure randomness for crypto
- Timezones (naive datetimes), path handling without pathlib, non-deterministic behavior
- Packaging/env: unpinned deps, missing constraints, no __main__/entry points
- Async misuse (blocking calls in coroutines), thread/process resource cleanup

Quick commands you can run and share results
- CPU profile: python -m cProfile -o prof.out your_script.py; then snakeviz prof.out
- Sampling profile (low overhead): py-spy top -- python your_script.py
- Memory profile: pip install memory_profiler; add @profile and run mprof run python your_script.py; mprof plot
- Static checks: pip install ruff mypy bandit; run:
  - ruff check .
  - mypy .
  - bandit -r .
- Complexity: pip install radon; radon cc -s -a .

If you provide the code or profiling outputs, I’ll produce a focused list of:
- Concrete bottlenecks with evidence and optimized alternatives
- Specific code quality fixes (file:line) with rationale
- Best practice violations and remediation steps
- A prioritized “quick wins” list and longer-term refactors

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”), so I don’t have the actual Python code to review.

Please paste the code (or share a repo/gist) and, if possible, include:
- Python version and key libraries
- Typical input sizes and performance goals (e.g., latency/throughput/memory targets)
- Where you suspect problems (functions/files) and how to run the code/tests

While you gather that, here’s how I will assess it and what you can pre-run:

What I will identify
1) Performance bottlenecks
- Hot paths via profiling (CPU: cProfile/pyinstrument; sampling: py-spy; memory: memray/memory_profiler)
- Algorithmic complexity and data structure choices
- Inefficient Python loops where vectorization/NumPy/C-extensions help
- Repeated work (recomputations, repeated regex/JSON parsing, poor caching)
- I/O patterns (chatty FS/network, N+1 DB queries, synchronous waits, small-buffer reads/writes)
- Concurrency issues (GIL-bound CPU work in threads, missing multiprocessing/numba, blocking in asyncio)
- Pandas/NumPy pitfalls (row-by-row applies, SettingWithCopy, non-inplace ops that copy large arrays)
- Serialization and IPC overhead; unnecessary conversions

2) Code quality issues
- Readability, naming, duplication, long functions, high cyclomatic complexity
- Error handling (broad except, swallowed exceptions, missing retries/backoff)
- Resource management (missing context managers for files/sockets/processes)
- Logging quality and log levels; excessive string formatting in hot paths
- Dead code, unused imports, magic numbers, poor configuration handling
- Lack of tests or type hints; fragile APIs; circular dependencies

3) Best practice violations
- PEP 8/257 conventions; inconsistent formatting (recommend black + ruff)
- Security hazards: eval/exec, subprocess with shell=True, yaml.load without SafeLoader, pickle on untrusted data, verify=False, hardcoded secrets
- DB/SQL not parameterized; missing timeouts/retries; insecure randomness for crypto
- Timezones (naive datetimes), path handling without pathlib, non-deterministic behavior
- Packaging/env: unpinned deps, missing constraints, no __main__/entry points
- Async misuse (blocking calls in coroutines), thread/process resource cleanup

Quick commands you can run and share results
- CPU profile: python -m cProfile -o prof.out your_script.py; then snakeviz prof.out
- Sampling profile (low overhead): py-spy top -- python your_script.py
- Memory profile: pip install memory_profiler; add @profile and run mprof run python your_script.py; mprof plot
- Static checks: pip install ruff mypy bandit; run:
  - ruff check .
  - mypy .
  - bandit -r .
- Complexity: pip install radon; radon cc -s -a .

If you provide the code or profiling outputs, I’ll produce a focused list of:
- Concrete bottlenecks with evidence and optimized alternatives
- Specific code quality fixes (file:line) with rationale
- Best practice violations and remediation steps
- A prioritized “quick wins” list and longer-term refactors

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”), so I don’t have the actual Python code to review.

Please paste the code (or share a repo/gist) and, if possible, include:
- Python version and key libraries
- Typical input sizes and performance goals (e.g., latency/throughput/memory targets)
- Where you suspect problems (functions/files) and how to run the code/tests

While you gather that, here’s how I will assess it and what you can pre-run:

What I will identify
1) Performance bottlenecks
- Hot paths via profiling (CPU: cProfile/pyinstrument; sampling: py-spy; memory: memray/memory_profiler)
- Algorithmic complexity and data structure choices
- Inefficient Python loops where vectorization/NumPy/C-extensions help
- Repeated work (recomputations, repeated regex/JSON parsing, poor caching)
- I/O patterns (chatty FS/network, N+1 DB queries, synchronous waits, small-buffer reads/writes)
- Concurrency issues (GIL-bound CPU work in threads, missing multiprocessing/numba, blocking in asyncio)
- Pandas/NumPy pitfalls (row-by-row applies, SettingWithCopy, non-inplace ops that copy large arrays)
- Serialization and IPC overhead; unnecessary conversions

2) Code quality issues
- Readability, naming, duplication, long functions, high cyclomatic complexity
- Error handling (broad except, swallowed exceptions, missing retries/backoff)
- Resource management (missing context managers for files/sockets/processes)
- Logging quality and log levels; excessive string formatting in hot paths
- Dead code, unused imports, magic numbers, poor configuration handling
- Lack of tests or type hints; fragile APIs; circular dependencies

3) Best practice violations
- PEP 8/257 conventions; inconsistent formatting (recommend black + ruff)
- Security hazards: eval/exec, subprocess with shell=True, yaml.load without SafeLoader, pickle on untrusted data, verify=False, hardcoded secrets
- DB/SQL not parameterized; missing timeouts/retries; insecure randomness for crypto
- Timezones (naive datetimes), path handling without pathlib, non-deterministic behavior
- Packaging/env: unpinned deps, missing constraints, no __main__/entry points
- Async misuse (blocking calls in coroutines), thread/process resource cleanup

Quick commands you can run and share results
- CPU profile: python -m cProfile -o prof.out your_script.py; then snakeviz prof.out
- Sampling profile (low overhead): py-spy top -- python your_script.py
- Memory profile: pip install memory_profiler; add @profile and run mprof run python your_script.py; mprof plot
- Static checks: pip install ruff mypy bandit; run:
  - ruff check .
  - mypy .
  - bandit -r .
- Complexity: pip install radon; radon cc -s -a .

If you provide the code or profiling outputs, I’ll produce a focused list of:
- Concrete bottlenecks with evidence and optimized alternatives
- Specific code quality fixes (file:line) with rationale
- Best practice violations and remediation steps
- A prioritized “quick wins” list and longer-term refactors

## Optimization Summary

The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file}}

## Original vs Optimized

### Original Code Issues Identified:
See analysis above for detailed breakdown.

### Optimized Code Benefits:
- Improved performance through algorithmic optimizations
- Enhanced code quality and maintainability
- Better error handling and validation
- Adherence to best practices and conventions”), so I don’t have the actual Python code to review.

Please paste the code (or share a repo/gist) and, if possible, include:
- Python version and key libraries
- Typical input sizes and performance goals (e.g., latency/throughput/memory targets)
- Where you suspect problems (functions/files) and how to run the code/tests

While you gather that, here’s how I will assess it and what you can pre-run:

What I will identify
1) Performance bottlenecks
- Hot paths via profiling (CPU: cProfile/pyinstrument; sampling: py-spy; memory: memray/memory_profiler)
- Algorithmic complexity and data structure choices
- Inefficient Python loops where vectorization/NumPy/C-extensions help
- Repeated work (recomputations, repeated regex/JSON parsing, poor caching)
- I/O patterns (chatty FS/network, N+1 DB queries, synchronous waits, small-buffer reads/writes)
- Concurrency issues (GIL-bound CPU work in threads, missing multiprocessing/numba, blocking in asyncio)
- Pandas/NumPy pitfalls (row-by-row applies, SettingWithCopy, non-inplace ops that copy large arrays)
- Serialization and IPC overhead; unnecessary conversions

2) Code quality issues
- Readability, naming, duplication, long functions, high cyclomatic complexity
- Error handling (broad except, swallowed exceptions, missing retries/backoff)
- Resource management (missing context managers for files/sockets/processes)
- Logging quality and log levels; excessive string formatting in hot paths
- Dead code, unused imports, magic numbers, poor configuration handling
- Lack of tests or type hints; fragile APIs; circular dependencies

3) Best practice violations
- PEP 8/257 conventions; inconsistent formatting (recommend black + ruff)
- Security hazards: eval/exec, subprocess with shell=True, yaml.load without SafeLoader, pickle on untrusted data, verify=False, hardcoded secrets
- DB/SQL not parameterized; missing timeouts/retries; insecure randomness for crypto
- Timezones (naive datetimes), path handling without pathlib, non-deterministic behavior
- Packaging/env: unpinned deps, missing constraints, no __main__/entry points
- Async misuse (blocking calls in coroutines), thread/process resource cleanup

Quick commands you can run and share results
- CPU profile: python -m cProfile -o prof.out your_script.py; then snakeviz prof.out
- Sampling profile (low overhead): py-spy top -- python your_script.py
- Memory profile: pip install memory_profiler; add @profile and run mprof run python your_script.py; mprof plot
- Static checks: pip install ruff mypy bandit; run:
  - ruff check .
  - mypy .
  - bandit -r .
- Complexity: pip install radon; radon cc -s -a .

If you provide the code or profiling outputs, I’ll produce a focused list of:
- Concrete bottlenecks with evidence and optimized alternatives
- Specific code quality fixes (file:line) with rationale
- Best practice violations and remediation steps
- A prioritized “quick wins” list and longer-term refactors

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