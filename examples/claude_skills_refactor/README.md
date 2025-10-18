# Claude Skills Refactor - Example Pipelines

This directory contains example pipelines demonstrating the Claude Skills refactor capabilities.

## Overview

These examples showcase:
- **Automatic skill creation** using the ROMA pattern
- **Multi-agent workflows** with review and refinement
- **Advanced control flow** (loops, conditionals, parallel execution)
- **Latest Anthropic models** (Opus 4.1, Sonnet 4.5, Haiku 4.5)
- **Real-world testing** (no mocks or simulations)

## Example Pipelines

### 1. Simple Code Review (`01_simple_code_review.yaml`)

Analyzes code structure and generates a review report.

**Features:**
- Sequential analysis pipeline
- Multi-step reasoning
- Markdown report generation

**Models Used:**
- Claude 3.5 Sonnet (for all analysis steps)

**Usage:**
```bash
orchestrator run examples/claude_skills_refactor/01_simple_code_review.yaml \
  --input code_directory="./my_project"
```

### 2. Research Synthesis (`02_research_synthesis.yaml`)

Researches a topic and synthesizes findings into a comprehensive report.

**Features:**
- Research planning
- Multi-source information gathering
- Intelligent synthesis with Opus 4.1

**Models Used:**
- Claude 3.5 Sonnet (planning and research)
- Claude Opus 4.1 (final synthesis - highest quality)

**Usage:**
```bash
orchestrator run examples/claude_skills_refactor/02_research_synthesis.yaml \
  --input research_topic="Quantum computing applications in ML"
```

### 3. Parallel Data Processing (`03_parallel_data_processing.yaml`)

Processes multiple data sources in parallel and aggregates results.

**Features:**
- Parallel step execution
- Model selection based on processing mode
- Data aggregation and synthesis

**Models Used:**
- Claude Haiku 4.5 (fast mode - cost-effective)
- Claude 3.5 Sonnet (thorough mode - higher quality)

**Usage:**
```bash
# Fast mode (uses Haiku)
orchestrator run examples/claude_skills_refactor/03_parallel_data_processing.yaml \
  --input processing_mode="fast"

# Thorough mode (uses Sonnet)
orchestrator run examples/claude_skills_refactor/03_parallel_data_processing.yaml \
  --input processing_mode="thorough"
```

## Key Features Demonstrated

### Automatic Skill Creation
When pipelines reference skills that don't exist (via `tool` field), the EnhancedSkillsCompiler automatically:
1. Detects missing skills
2. Uses ROMA pattern to create them:
   - **Atomize**: Breaks capability into discrete tasks
   - **Plan**: Designs skill structure
   - **Execute**: Generates implementation
   - **Aggregate**: Reviews and refines
3. Saves skills to `~/.orchestrator/skills/`
4. Registers in skill registry

### Model Selection
Examples demonstrate intelligent model routing:
- **Haiku 4.5**: Fast, cost-effective for simple tasks
- **Sonnet 4.5**: Balanced quality/cost for complex coding
- **Opus 4.1**: Highest quality for critical analysis and synthesis

### Control Flow
Pipelines use advanced control flow:
- **Sequential**: Steps with dependencies
- **Parallel**: Independent steps execute simultaneously
- **Conditional**: Model selection based on parameters

## Running the Examples

### Prerequisites
```bash
# Ensure API key is configured
export ANTHROPIC_API_KEY="your-key-here"

# Or configure in ~/.orchestrator/.env
echo "ANTHROPIC_API_KEY=your-key-here" >> ~/.orchestrator/.env
```

### Execution
```bash
# Run any example pipeline
orchestrator run examples/claude_skills_refactor/<pipeline>.yaml

# With custom inputs
orchestrator run <pipeline>.yaml --input param1="value1" --input param2="value2"

# View compilation stats
orchestrator compile <pipeline>.yaml --show-stats
```

## Expected Outputs

### Code Review
- `./code_review_report.md`: Comprehensive code analysis with recommendations

### Research Synthesis
- `./research_report.md`: Synthesized research findings with references

### Parallel Data Processing
- `./aggregated_results.json`: Combined and deduplicated data

## Pipeline Structure

All pipelines follow best practices:
```yaml
id: unique-pipeline-id
name: "Human-Readable Name"
description: "What this pipeline does"
version: "1.0.0"

inputs:
  parameter_name:
    type: string|integer|array|object
    description: "Parameter description"
    default: "default value"

steps:
  - id: step_id
    name: "Step description"
    action: llm_generate  # or other actions
    dependencies: [previous_step_id]  # optional
    parameters:
      prompt: "Task description with {{ variable }} substitution"
      model: "claude-3-5-sonnet-20241022"
      max_tokens: 2000
    produces: output_artifact  # optional
    location: "./output_file.md"  # optional
```

## Next Steps

These examples provide a foundation for building more complex pipelines:
- Multi-agent problem solving
- Content generation workflows
- Security audit pipelines
- API integration builders
- Documentation generators

See the [full technical design document](../../TECHNICAL_DESIGN_CLAUDE_SKILLS_REFACTOR_V2.md) for additional pipeline ideas and advanced patterns.