# Issue #169: Model Routing Demo Problems

## Critical Issues Found

### 1. GPT-5-nano Returns Empty Responses
- **Problem**: Model fails with "config" parameter error
- **Location**: `src/orchestrator/models/openai_model.py`
- **Error**: `AsyncCompletions.create() got an unexpected keyword argument 'config'`
- **Impact**: Code generation step always fails

### 2. analyze_text Action Ignores Custom Prompts
- **Problem**: When using `action: analyze_text`, the custom `prompt` parameter is ignored
- **Location**: `src/orchestrator/control_systems/model_based_control_system.py:171-189`
- **Impact**: Models don't receive the actual analysis instructions
- **Fixed**: ✅ Added support for custom prompt parameter

### 3. Models Not Receiving Full Context
- **Problem**: Sales data is not being passed to the model properly
- **Location**: Pipeline configuration and prompt building
- **Impact**: Models ask for data that should have been provided

### 4. Cost Estimates Show $0.00
- **Problem**: Even expensive models like Claude Sonnet show $0.00 cost
- **Location**: `src/orchestrator/tools/llm_tools.py`
- **Cause**: TaskDelegationTool returns 0.0 for estimated_cost

### 5. Model Registry Not Available in Tests
- **Problem**: When running tools directly, model registry is empty
- **Location**: Tool initialization and model registry access
- **Impact**: Tests fall back to ollama with $0 cost

### 6. Template Rendering Issues
- **Problem**: Some outputs show unrendered templates like `{{content}}`
- **Location**: Final report generation step
- **Impact**: Poor quality outputs

### 7. Routing Strategy Not Affecting Model Selection
- **Problem**: Same models selected regardless of cost/balanced/quality priority
- **Location**: MultiModelRoutingTool routing logic
- **Impact**: Priority parameter has no effect

## Fixes Applied So Far

1. ✅ Fixed analyze_text to use custom prompt parameter
2. ✅ Fixed Task creation to include name parameter
3. ✅ Made MultiModelRoutingTool parameters optional
4. ✅ Fixed budget constraint type conversion (string to float)

## Still Need to Fix

1. ❌ GPT-5-nano API error with config parameter
2. ❌ Cost estimation returning $0.00
3. ❌ Model registry initialization in tests
4. ❌ Routing strategy not affecting model selection
5. ❌ Better context passing to models

## Test Results
- 3/7 unit tests passing
- Pipeline runs but with quality issues
- Models not receiving proper context
- Cost tracking broken