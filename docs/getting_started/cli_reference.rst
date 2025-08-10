CLI Reference
==============

The Orchestrator framework provides command-line tools for running pipelines and managing workflows. This section covers the available CLI tools and their usage.

Pipeline Runner
---------------

The ``run_pipeline.py`` script is the primary CLI tool for executing YAML pipeline files.

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   python scripts/run_pipeline.py <pipeline_file> [options]

Arguments
^^^^^^^^^

**Positional Arguments:**

``pipeline``
   Path to the YAML pipeline file to execute.

**Optional Arguments:**

``-h, --help``
   Show help message and exit.

``-i INPUT, --input INPUT``
   Specify input parameters for the pipeline in key=value format. Can be used multiple times.

``-f INPUT_FILE, --input-file INPUT_FILE``
   Path to a JSON or YAML file containing input parameters.

``-o OUTPUT_DIR, --output-dir OUTPUT_DIR``
   Directory where pipeline outputs should be saved. Creates the directory if it doesn't exist.

Examples
^^^^^^^^

**Run a pipeline with default parameters:**

.. code-block:: bash

   python scripts/run_pipeline.py examples/research_minimal.yaml

**Run a pipeline with custom inputs:**

.. code-block:: bash

   python scripts/run_pipeline.py examples/research_minimal.yaml -i topic="artificial intelligence"

**Run a pipeline with multiple inputs:**

.. code-block:: bash

   python scripts/run_pipeline.py examples/research_basic.yaml \\
       -i topic="machine learning" \\
       -i depth="comprehensive"

**Run a pipeline with custom output directory:**

.. code-block:: bash

   python scripts/run_pipeline.py examples/research_minimal.yaml \\
       -i topic="quantum computing" \\
       -o /path/to/custom/output

**Run a pipeline with input file:**

.. code-block:: bash

   python scripts/run_pipeline.py examples/web_research_pipeline.yaml \\
       -f pipeline_inputs.json \\
       -o ./research_results

Output Directory Behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``-o/--output-dir`` flag controls where pipeline outputs are saved, but its effectiveness depends on how the pipeline is configured.

**Compatible Pipelines:**

Pipelines that use the ``{{ output_path }}`` template parameter will respect the ``-o`` flag:

.. code-block:: yaml

   parameters:
     output_path:
       type: string
       default: "examples/outputs/my_pipeline"
       description: Directory where output files will be saved

   steps:
     - id: save_report
       tool: filesystem
       action: write
       parameters:
         path: "{{ output_path }}/report.md"
         content: "{{ analysis_result }}"

**Incompatible Pipelines:**

Pipelines with hardcoded output paths will ignore the ``-o`` flag:

.. code-block:: yaml

   # This will NOT respect the -o flag
   steps:
     - id: save_report
       tool: filesystem
       action: write
       parameters:
         path: "examples/outputs/hardcoded_path/report.md"  # Hardcoded path
         content: "{{ analysis_result }}"

**Warning System:**

The CLI will automatically detect incompatible pipelines and display a warning:

.. code-block:: bash

   $ python scripts/run_pipeline.py examples/old_pipeline.yaml -o /tmp/output
   
   ⚠️  Warning: This pipeline may not respect the -o flag.
      Pipeline uses hardcoded output paths instead of {{ output_path }} parameter.
      Files may be saved to default locations instead of: /tmp/output

Making Pipelines Compatible
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To make a pipeline compatible with the ``-o`` flag:

1. **Add output_path parameter:**

   .. code-block:: yaml

      parameters:
        output_path:
          type: string
          default: "examples/outputs/my_pipeline"
          description: Directory where output files will be saved

2. **Use the parameter in file paths:**

   .. code-block:: yaml

      steps:
        - id: save_results
          tool: filesystem
          action: write
          parameters:
            path: "{{ output_path }}/results.json"  # Use template parameter
            content: "{{ processed_data }}"

3. **Test the pipeline:**

   .. code-block:: bash

      python scripts/run_pipeline.py examples/my_pipeline.yaml -o /tmp/test_output

Input File Formats
^^^^^^^^^^^^^^^^^^^

The ``-f/--input-file`` flag accepts JSON or YAML input files.

**JSON Format:**

.. code-block:: json

   {
     "topic": "artificial intelligence",
     "depth": "comprehensive",
     "max_sources": 15
   }

**YAML Format:**

.. code-block:: yaml

   topic: "artificial intelligence"
   depth: "comprehensive"
   max_sources: 15

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

The pipeline runner respects the following environment variables:

``PYTHONPATH``
   Should include the path to the Orchestrator source directory for development usage.

``OPENAI_API_KEY``
   OpenAI API key for GPT models.

``ANTHROPIC_API_KEY``
   Anthropic API key for Claude models.

``GOOGLE_API_KEY``
   Google API key for Gemini models.

Error Handling
^^^^^^^^^^^^^^

The CLI provides detailed error messages for common issues:

**Pipeline File Not Found:**

.. code-block:: bash

   $ python scripts/run_pipeline.py nonexistent.yaml
   Error: Pipeline file 'nonexistent.yaml' not found.

**Invalid Input Format:**

.. code-block:: bash

   $ python scripts/run_pipeline.py examples/pipeline.yaml -i "invalid_format"
   Error: Input must be in key=value format. Got: invalid_format

**Missing Required Parameters:**

.. code-block:: bash

   $ python scripts/run_pipeline.py examples/pipeline.yaml
   Error: Missing required parameter 'topic'. Use -i topic="your_topic"

**API Key Not Set:**

.. code-block:: bash

   $ python scripts/run_pipeline.py examples/pipeline.yaml
   Error: No API keys found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY.

Exit Codes
^^^^^^^^^^

The pipeline runner returns standard exit codes:

- ``0``: Success
- ``1``: General error (invalid arguments, pipeline execution failure)
- ``2``: File not found
- ``3``: Configuration error (missing API keys, invalid input format)

Best Practices
^^^^^^^^^^^^^^

1. **Always specify output directory** for production workflows:

   .. code-block:: bash

      python scripts/run_pipeline.py pipeline.yaml -o ./results/$(date +%Y%m%d_%H%M%S)

2. **Use input files** for complex configurations:

   .. code-block:: bash

      python scripts/run_pipeline.py pipeline.yaml -f production_config.yaml

3. **Check pipeline compatibility** before running in production:

   .. code-block:: bash

      # Test with a temporary output directory
      python scripts/run_pipeline.py pipeline.yaml -o /tmp/test_run

4. **Set up proper API keys** in your environment or ``.env`` file.

5. **Use absolute paths** for output directories when running from different working directories.

Troubleshooting
^^^^^^^^^^^^^^^

**Pipeline ignores -o flag:**
   Check if the pipeline uses ``{{ output_path }}`` parameters. See the warning system output.

**Permission denied errors:**
   Ensure the output directory is writable:

   .. code-block:: bash

      mkdir -p /path/to/output && chmod 755 /path/to/output

**Module not found errors:**
   Set the PYTHONPATH correctly:

   .. code-block:: bash

      export PYTHONPATH=/path/to/orchestrator/src:$PYTHONPATH

**API rate limits:**
   Some pipelines may hit API rate limits. The framework includes automatic retry logic, but you may need to wait or use different API keys.