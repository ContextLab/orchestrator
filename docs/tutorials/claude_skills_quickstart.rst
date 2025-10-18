==================================
Claude Skills System Quick Start
==================================

.. contents:: Table of Contents
   :depth: 2
   :local:

Overview
========

The Claude Skills Orchestrator provides automatic skill creation, intelligent model selection, and advanced workflow orchestration using Anthropic's Claude models.

Key Features
------------

* **Automatic Skill Creation**: Skills created on-demand using ROMA pattern
* **Latest Claude Models**: Opus 4.1, Sonnet 4.5, Haiku 4.5
* **Real-World Testing**: All skills validated with real APIs (no mocks)
* **Advanced Control Flow**: Loops, conditionals, parallel execution
* **Intelligent Routing**: Right model for each task

Installation
============

Basic Setup
-----------

.. code-block:: bash

   # Clone repository
   git clone https://github.com/ContextLab/orchestrator.git
   cd orchestrator

   # Install dependencies
   pip install -r requirements.txt

   # Install orchestrator
   pip install -e .

API Key Configuration
---------------------

Set up your Anthropic API key:

.. code-block:: bash

   # Create configuration directory
   mkdir -p ~/.orchestrator

   # Add API key
   echo "ANTHROPIC_API_KEY=your-key-here" >> ~/.orchestrator/.env

The API key is:
* Automatically loaded by the framework
* Securely stored (excluded from git)
* Shared across all projects

Your First Pipeline
===================

Create a Simple Pipeline
------------------------

Create ``my_first_pipeline.yaml``:

.. code-block:: yaml

   id: my-first-pipeline
   name: "My First Pipeline"
   description: "A simple introduction to Claude Skills"
   version: "1.0.0"

   inputs:
     topic:
       type: string
       default: "artificial intelligence"

   steps:
     - id: generate_summary
       name: "Generate Summary"
       action: llm_generate
       parameters:
         prompt: "Write a brief summary about {{ topic }}"
         model: claude-3-5-sonnet-20241022
         max_tokens: 500

     - id: expand_summary
       name: "Expand Summary"
       dependencies: [generate_summary]
       action: llm_generate
       parameters:
         prompt: |
           Expand on this summary with more details:
           {{ generate_summary.result }}
         model: claude-3-5-sonnet-20241022
         max_tokens: 1000
       produces: final_content
       location: "./output.md"

Run the Pipeline
----------------

.. code-block:: bash

   # Run with default inputs
   python scripts/execution/run_pipeline.py my_first_pipeline.yaml

   # Run with custom input
   python scripts/execution/run_pipeline.py my_first_pipeline.yaml \
     -i topic="machine learning"

Understanding the Skills System
================================

What are Skills?
----------------

Skills are reusable capabilities that the orchestrator creates automatically when needed.

**How it works:**

1. Pipeline references a skill (via ``tool`` field)
2. Orchestrator checks if skill exists
3. If missing, creates it using the **ROMA pattern**
4. Skill is saved and reusable

ROMA Pattern
------------

Every skill goes through 4 stages:

.. code-block:: text

   1. ATOMIZE
      └─> Break capability into discrete atomic tasks
          (e.g., "web search" → parse query, make request, extract results)

   2. PLAN
      └─> Design skill structure
          - Parameters and types
          - Expected outputs
          - Implementation approach

   3. EXECUTE
      └─> Generate Python implementation
          - Real code (not stubs)
          - Error handling
          - Documentation

   4. AGGREGATE
      └─> Review and refine
          - Opus 4.1 reviews the skill
          - Iterative improvements
          - Real-world testing

Example: Automatic Skill Creation
----------------------------------

When you write:

.. code-block:: yaml

   steps:
     - id: search_web
       tool: web-searcher  # Skill doesn't exist yet
       action: llm_generate
       parameters:
         query: "{{ search_term }}"
         prompt: "Search for {{ search_term }}"

The orchestrator will:

1. Detect ``web-searcher`` is missing
2. Use Claude Sonnet to create the skill
3. Use Claude Opus to review it
4. Test with real web searches
5. Save to ``~/.orchestrator/skills/web-searcher/``

Model Selection Guide
=====================

Available Models
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 35 15 15

   * - Model
     - Context
     - Best For
     - Speed
     - Cost
   * - Opus 4.1
     - 200K
     - Deep analysis, critical reviews
     - Slow
     - $$$
   * - Sonnet 4.5
     - 1M
     - Coding, orchestration, general
     - Medium
     - $$
   * - Haiku 4.5
     - 200K
     - Simple tasks, validation
     - Fast
     - $

When to Use Each Model
----------------------

**Claude Haiku 4.5** (``claude-haiku-4-5``)

* Quick validations
* Formatting tasks
* Simple queries
* High-volume operations

.. code-block:: yaml

   parameters:
     model: claude-haiku-4-5
     prompt: "Format this JSON: {data}"

**Claude Sonnet 4.5** (``claude-sonnet-4-5``)

* Code generation
* Complex analysis
* Orchestration tasks
* General workflows

.. code-block:: yaml

   parameters:
     model: claude-sonnet-4-5
     prompt: "Analyze this codebase and create a report"

**Claude Opus 4.1** (``claude-opus-4-1-20250805``)

* Critical reviews
* Complex reasoning
* Final synthesis
* Quality-critical tasks

.. code-block:: yaml

   parameters:
     model: claude-opus-4-1-20250805
     prompt: "Comprehensive security audit of this system"

Conditional Model Selection
----------------------------

Use templates to select models based on parameters:

.. code-block:: yaml

   steps:
     - id: process_data
       action: llm_generate
       parameters:
         model: >
           {% if priority == 'high' %}
           claude-opus-4.1
           {% elif volume == 'large' %}
           claude-haiku-4-5
           {% else %}
           claude-3-5-sonnet-20241022
           {% endif %}
         prompt: "Process this data"

Control Flow Patterns
======================

Sequential Processing
---------------------

Steps with dependencies run in order:

.. code-block:: yaml

   steps:
     - id: step1
       action: llm_generate
       parameters:
         prompt: "First task"

     - id: step2
       dependencies: [step1]  # Waits for step1
       action: llm_generate
       parameters:
         prompt: "Use {{ step1.result }}"

     - id: step3
       dependencies: [step2]  # Waits for step2
       action: llm_generate
       parameters:
         prompt: "Final task with {{ step2.result }}"

Parallel Execution
------------------

Steps without dependencies run simultaneously:

.. code-block:: yaml

   steps:
     # These three run in parallel
     - id: analyze_code
       action: llm_generate

     - id: analyze_tests
       action: llm_generate

     - id: analyze_docs
       action: llm_generate

     # This waits for all three
     - id: comprehensive_report
       dependencies: [analyze_code, analyze_tests, analyze_docs]
       action: llm_generate

Conditional Execution
---------------------

Use conditionals to control execution:

.. code-block:: yaml

   steps:
     - id: check_conditions
       action: llm_generate
       parameters:
         prompt: "Should we proceed? Return JSON: {should_proceed: true/false}"

     - id: conditional_step
       condition: "{{ check_conditions.result.should_proceed == true }}"
       action: llm_generate
       parameters:
         prompt: "Only runs if condition is true"

Advanced Examples
=================

Example 1: Code Review Pipeline
--------------------------------

.. code-block:: yaml

   id: code-review
   name: "Comprehensive Code Review"
   version: "1.0.0"

   inputs:
     code_directory:
       type: string
       default: "./src"

   steps:
     - id: analyze_structure
       name: "Analyze Code Structure"
       action: llm_generate
       parameters:
         prompt: |
           Analyze the code structure in {{ code_directory }}:
           1. File organization
           2. Module dependencies
           3. Complexity metrics
         model: claude-3-5-sonnet-20241022
         max_tokens: 2000

     - id: identify_issues
       name: "Identify Issues"
       dependencies: [analyze_structure]
       action: llm_generate
       parameters:
         prompt: |
           Based on this analysis:
           {{ analyze_structure.result }}

           Identify potential issues:
           - Code smells
           - Security concerns
           - Performance bottlenecks
         model: claude-3-5-sonnet-20241022
         max_tokens: 2000

     - id: generate_report
       name: "Generate Review Report"
       dependencies: [identify_issues]
       action: llm_generate
       parameters:
         prompt: |
           Create a comprehensive code review report:

           Structure: {{ analyze_structure.result }}
           Issues: {{ identify_issues.result }}

           Format as professional markdown.
         model: claude-opus-4-1-20250805  # Use Opus for quality
         max_tokens: 3000
       produces: review_report
       location: "./code_review.md"

Example 2: Research Synthesis
------------------------------

.. code-block:: yaml

   id: research-synthesis
   name: "Research and Synthesize"
   version: "1.0.0"

   inputs:
     research_topic:
       type: string
       description: "Topic to research"

   steps:
     - id: create_plan
       action: llm_generate
       parameters:
         prompt: "Create research plan for {{ research_topic }}"
         model: claude-3-5-sonnet-20241022
         max_tokens: 1000

     - id: conduct_research
       dependencies: [create_plan]
       action: llm_generate
       parameters:
         prompt: |
           Research {{ research_topic }} following this plan:
           {{ create_plan.result }}
         model: claude-3-5-sonnet-20241022
         max_tokens: 4000

     - id: synthesize
       dependencies: [conduct_research]
       action: llm_generate
       parameters:
         prompt: |
           Synthesize findings into a report:
           {{ conduct_research.result }}
         model: claude-opus-4-1-20250805
         max_tokens: 5000
       produces: research_report
       location: "./research.md"

Python API Usage
================

Basic Pipeline Execution
-------------------------

.. code-block:: python

   import orchestrator as orc

   # Initialize
   orc.init_models()

   # Compile pipeline
   pipeline = orc.compile("my_pipeline.yaml")

   # Run with inputs
   result = pipeline.run(topic="AI agents")

   print(result)

Working with Skills
-------------------

.. code-block:: python

   from orchestrator.skills import SkillCreator, SkillRegistry

   # Create a new skill
   creator = SkillCreator()
   skill = await creator.create_skill(
       capability="Extract data from PDFs",
       max_iterations=3
   )

   # List all skills
   registry = SkillRegistry()
   all_skills = registry.list_skills()

   # Search for skills
   pdf_skills = registry.search("pdf")

   # Get skill details
   skill_data = registry.get("pdf-extractor")

Advanced Compilation
--------------------

.. code-block:: python

   from orchestrator.compiler import EnhancedSkillsCompiler

   # Create compiler with skills support
   compiler = EnhancedSkillsCompiler()

   # Compile with automatic skill creation
   with open("pipeline.yaml") as f:
       pipeline = await compiler.compile(
           f.read(),
           context={"param": "value"},
           auto_create_missing_skills=True
       )

   # Check compilation stats
   stats = compiler.get_compilation_stats()
   print(f"Auto-created {stats['skills_auto_created']} skills")
   print(f"Skills: {stats['created_skill_names']}")

Best Practices
==============

Model Selection
---------------

1. **Start with Haiku** for simple tasks
2. **Use Sonnet** for most workflows
3. **Reserve Opus** for critical decisions

.. code-block:: yaml

   # Good model choices
   - Validation: claude-haiku-4-5
   - Code generation: claude-sonnet-4-5
   - Final review: claude-opus-4.1

Pipeline Design
---------------

1. **Break complex tasks** into sequential steps
2. **Use parallel execution** for independent operations
3. **Add explicit dependencies** for clarity
4. **Include descriptions** for maintainability

.. code-block:: yaml

   # Good pipeline structure
   steps:
     - id: load_data
       name: "Load Data"
       description: "Load input data from source"
       action: llm_generate

     - id: transform
       name: "Transform Data"
       description: "Apply transformations"
       dependencies: [load_data]
       action: llm_generate

     - id: save_results
       name: "Save Results"
       description: "Write to output file"
       dependencies: [transform]
       action: llm_generate
       produces: output_data
       location: "./results.json"

Skills Management
-----------------

1. **Let skills auto-create** for common operations
2. **Review auto-created skills** in ``~/.orchestrator/skills/``
3. **Test with real data** before production
4. **Export skills** for sharing

Error Handling
--------------

.. code-block:: yaml

   steps:
     - id: risky_operation
       action: llm_generate
       parameters:
         prompt: "Complex task"
       on_failure: continue  # Options: continue, fail, retry
       max_retries: 3
       timeout: 300

Troubleshooting
===============

API Key Issues
--------------

Verify API key is set:

.. code-block:: bash

   # Check if key exists
   cat ~/.orchestrator/.env | grep ANTHROPIC_API_KEY

   # Test the key
   python -c "from orchestrator.utils.api_keys_flexible import ensure_api_key; print(ensure_api_key('anthropic')[:20])"

Skill Creation Fails
--------------------

Common issues:

1. **Rate limits**: Wait and retry
2. **Invalid capability**: Be more specific
3. **Network issues**: Check connectivity

.. code-block:: python

   # Manual skill creation with error handling
   try:
       skill = await creator.create_skill(
           capability="Specific, clear capability description",
           max_iterations=5  # More iterations for complex skills
       )
   except Exception as e:
       print(f"Skill creation failed: {e}")
       # Check logs in ~/.orchestrator/logs/

Docker Not Starting
-------------------

Docker starts automatically, but if issues persist:

.. code-block:: bash

   # macOS
   open -a Docker

   # Linux
   sudo systemctl start docker

   # Verify
   docker info

Next Steps
==========

1. **Try the examples**: ``examples/claude_skills_refactor/``
2. **Read the full guide**: ``docs/CLAUDE_SKILLS_USER_GUIDE.md``
3. **Build your pipeline**: Start simple, add complexity
4. **Explore skills**: Check ``~/.orchestrator/skills/``

Additional Resources
====================

* :doc:`../CLAUDE_SKILLS_USER_GUIDE` - Complete documentation
* :doc:`../QUICK_START` - 5-minute tutorial
* :doc:`examples` - Working pipeline examples
* `GitHub Issues <https://github.com/ContextLab/orchestrator/issues>`_ - Get help

---

**Version**: 1.0.0 (Claude Skills Refactor)

**Last Updated**: January 2025