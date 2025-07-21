# Remove Mock Implementations from test_sandboxed_executor_missing_lines.py

## Description

The test file `tests/test_sandboxed_executor_missing_lines.py` extensively uses `unittest.mock` for mocking Docker operations and other system interactions. According to our testing methodology in CLAUDE.md, we should use real functionality instead of mocks.

## Current Mock Usage

The file contains extensive mocking throughout:

1. **Line 3**: Import of mock utilities
   ```python
   from unittest.mock import AsyncMock, Mock, patch
   ```

2. **Lines 30-43**: Mocking Docker client and container operations
   ```python
   with patch("docker.from_env") as mock_from_env:
       mock_client = Mock()
       mock_from_env.return_value = mock_client
       mock_client.containers.run.side_effect = Exception("Docker execution failed")
   ```

3. **Lines 52-73**: Multiple patches for Docker import, tempfile, and os.unlink
   - Mocking `builtins.__import__` for Docker
   - Mocking `tempfile.NamedTemporaryFile`
   - Mocking `os.unlink` to simulate file cleanup failures

4. **Lines 82-104**: Mocking subprocess execution and file operations
   ```python
   with patch("asyncio.create_subprocess_exec", return_value=mock_process):
   ```

5. **Lines 112-118**: Mocking executor availability checks
   ```python
   executor.is_available = Mock(return_value=False)
   ```

## Required Changes

### 1. Replace Docker Mocks with Real Docker Operations
- Use actual Docker containers for testing
- Create real temporary containers that can fail in controlled ways
- Test actual Docker exception scenarios

### 2. Replace File System Mocks with Real File Operations
- Create actual temporary files
- Test real permission issues and cleanup failures
- Use actual file system operations

### 3. Replace Process Mocks with Real Process Execution
- Use actual subprocess calls
- Create real scripts that can fail in controlled ways
- Test actual process execution scenarios

### 4. Test Real Executor Availability
- Test with actual Docker availability checks
- Test fallback scenarios with real conditions

## Implementation Notes

According to CLAUDE.md testing methodology:
- Use real Docker containers even for failure scenarios
- Create actual files and test real cleanup failures
- Execute real subprocesses for testing
- Test actual availability of executors

## Acceptance Criteria

- [ ] All `unittest.mock` imports removed
- [ ] Docker operations use real Docker API
- [ ] File operations use real temporary files
- [ ] Subprocess operations use real process execution
- [ ] All tests verify actual behavior, not mocked responses
- [ ] Tests handle real Docker availability scenarios
- [ ] Tests create and clean up real resources

## Related Issues

- Similar to other mock removal issues in the test suite
- Part of the effort to use real-world testing patterns