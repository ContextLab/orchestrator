# Phase 1: Infrastructure Fixes

## Objective
Fix fundamental infrastructure issues preventing tests from executing.

## Root Issues
- Missing TestModel/TestProvider pattern across all test files
- Model registry initialization failures
- API key/credential requirements for integration tests
- External service dependencies

## Action Plan

### 1.1: Apply TestModel Pattern Systematically (Priority: Critical)
**Files to Update**: All test files using orchestrator
**Pattern**: Replace real model dependencies with TestModel infrastructure
**Expected Impact**: 60-80 tests can now execute

**Implementation**:
```python
# Add to each test file
from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider

# Replace orchestrator fixtures with:
@pytest.fixture
async def orchestrator():
    return create_test_orchestrator()
```

### 1.2: Fix Model Registry Issues (Priority: Critical)  
**Files to Update**: Control systems, compilers, resolvers
**Pattern**: Ensure select_model() + get_model() pattern applied everywhere
**Expected Impact**: Eliminate remaining coroutine errors

### 1.3: Mock External Services (Priority: High)
**Services to Mock**:
- LangChain integrations
- MCP server connections  
- External API calls
- File system operations (where appropriate)

### 1.4: Environment Setup (Priority: Medium)
**Configuration**:
- Test-specific environment variables
- Mock API keys for integration tests
- Isolated test database/storage

## Success Criteria
- [ ] TestModel pattern applied to 100% of test files
- [ ] All model registry issues resolved
- [ ] No test failures due to missing external services  
- [ ] 80% of tests can execute (may still fail on business logic)

## Estimated Impact
- **Before**: 0-20% of tests can execute
- **After**: 80-90% of tests can execute