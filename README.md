# Orchestrator 🧟

Orchestrator is a convenient wrapper for LangGraph, MCP, model spec, and other AI (LLM) agent control systems. It provides a way to:
 - Specify task-based control flows, dependencies (including conditional dependencies), and completion criteria
 - Use external tools (through MCP and custom scripts or command line calls)
 - Run sandboxed code (using Docker, conda, uv, and/or pyenv)
 - Enforce requirements of model outputs, including simple checks (e.g., data formatting) and complex checks (e.g., high-level quality control checks).
 - Specify and enforce operating constraints (e.g., monetary cost, computing costs like time or memory, maximum total runtime, minimum time to first response, permissions, privacy, etc.)

## How it works

Orchestrator is built around two core components:
  - YAML files for defining task control flows
  - A Python library for compiling and executing control flows

A task control flow looks something like this (ambiguous definitions or components are be sent to LLMs for parsing if enclosed by `<AUTO>...</AUTO>` tags, which can be nested):
```yaml
# pipelines/write-research-report.yaml

inputs:
 - topic: {report-topic}
 - instructions: {report-instructions}

outputs:
 - pdf: <AUTO>come up with an appropriate filename for the final report</AUTO>
 - tex: {pdf}[:-3] + '.tex'  # text enclosed in curly braces can refer to keyword arguments, methods, variables, or outputs of pipeline steps

pipeline:
 - web-search:
  - action: <AUTO>search the web for <AUTO>construct an appropriate web query about {topic}, using these additional instructions: {instructions}</AUTO></AUTO>  # here "topic" and "instructions" are keyword arguments
  - tool: headless-browser
  - produces: <AUTO>markdown file with detailed notes on each relevant result with annotated links to original sources; other relevant files like images, code, data, etc., that can be saved locally</AUTO>
  - location: ./searches/{pdf}/
  - requires-model:
   - min-size: 7B
   - expertise: medium
 - compile-search-results:
  - depends-on: web-search
  - action: <AUTO>create a markdown file collating the content from {web-search} into a single cohesive document. maintain annotated links back to original sources.</AUTO>
  - produces: markdown-file
  - location: ./searches/{pdf}/compiled_results.md
  - requires-model:
    - min-size: 10B
    - expertise: medium-high
 - quality-check-compilation:
  - depends-on: compile-search-results
  - requires-model:
    - min-size: 7B
    - expertise: medium
  - action:
    - create-parallel-queue:  # use the "on" attribute to create a list of tasks and execute each in parallel (omit "parallel" to execute sequentially)
      - on: <AUTO>create a list of every source in this document: {compile-search-results}</AUTO>
      - tool: headless-browser
      - action-loop: # list of actions to loop through sequentially; if unspecified, inherit properties and requirements from parent
        - action: <AUTO>verify the authenticity of this source (\#item) by following the web link and ensuring it is accurately described in {compile-search-results}</AUTO>  # use #item to refer to the current list/queue item; use square brackets to refer to specific *other* items
          - name: verify
          - produces: <AUTO>"true" if reference was verified, "<AUTO>corrected reference</AUTO>" if reference could be fixed with minor edits, or "false" if reference seems to be hallucinated</AUTO>
        - action: <AUTO>if {verify} is "false", update {compilation-search-results} to remove the reference. if {verify} has a corrected reference, update {compilation-search-results} to use the corrected reference.</AUTO>
        - until: <AUTO>all sources have been verified (or removed, if incorrect)</AUTO>
  - produces: markdown-file
  - location: ./searches/{pdf}/compiled_results_corrected.md
 - draft-report:
  - depends-on: quality-check-compilation
  - action <AUTO><<report_draft_prompt.md>></AUTO> # text between double angled brackets refers to filenames in the prompts/ folder
  - produces: markdown-file
  - requires-model:
   - min-size: 20B
   - expertise: high
  - produces: markdown-file
  - location: ./searches/{pdf}/draft_report.md
 - quality-check-assumptions:
  - depends-on: draft-report
  - requires-model:
    - min-size: 20B
    - expertise: high
  - action:
    - create-parallel-queue:
      - on: <AUTO>create a comprehensive list of every non-trivial claim made in this document (include, for each claim, any sources or supporting evidence provided in the document): {draft-report}</AUTO>
      - tool: headless-browser
      - action-loop:
        - action: <AUTO>verify the accuracy of this claim (\#item) by (a) doing a web search and (b) using logical reasoning and deductive inference *based only on the provided claim, sources, and supporting evidence. be sure to manually follow every source link to verify accuracy.</AUTO>
          - name: claim-check
          - produces: <AUTO>"true" if claim was verified, "<AUTO>corrected claim</AUTO>" if claim could be fixed with minor edits, or "false" if claim seems to be hallucinated</AUTO>
        - action: <AUTO>if {claim-check} is "false", update {draft-report} to remove the claim. if {claim-check} has a corrected claim, update {draft-report} to use the corrected claim.</AUTO>
        - until: <AUTO>all claims have been verified (or removed, if innacurate)</AUTO>
  - produces: markdown-file
  - location: ./searches/{pdf}/draft_report_corrected.md
 - quality-check-full-report:
  - depends-on: draft-report
  - requires-model:
    - min-size: 40B
    - expertise: very-high
  - action: <AUTO>do a thorough pass through the this document. without adding *any* new claims or references, revise the document to improve (a) clarity, (b) logical flow, (c) grammar, and (d) writing quality: {quality-check-assumptions}</AUTO>
  - produces: markdown-file
  - location: ./searches/{pdf}/draft_report_final.md
 - compile-pdf
  - depends-on: quality-check-full-report
  - tool: terminal
  - location: ./searches/{pdf}/report.pdf
  - action: "!pandoc -o {location} {quality-check-full-report}"  # appending "!" runs a terminal command in a sandboxed environment
  - requires-model: none
  - produces: pdf
  - on-error: debug-compilation
 - debug-compilation:
  - requires-model:
   - min-size: 40B
   - expertise: very-high
  - action-loop:
   - action: <AUTO>carefully check the output logs in the current dirctory to see why compiling the pdf failed; use bash commands to update the document and/or run other bash commands as needed until {compile-pdf} is a valid pdf.</AUTO>
   - tool: terminal
   - until: <AUTO>{compile-pdf} is a valid pdf</AUTO>
  - produces: pdf
  - location: ./searches/{pdf}/report.pdf
```

Given the above YAML file, the user calls this pipeline as follows:
```python
import orchestrator as orc

orc.init_models()  # initializes the pool of available models by reading models.yaml, along with examining environment variables or secrets (if running in Colab or a GitHub action) for relevant API keys
report_writer = orc.pipeline('pipelines/write-research-report.yaml')  # parse the pipeline into a callable OrchestratorPipeline object

# generate the report by running the pipeline
report = report_writer.run(topic='agents', instructions='Teach me everything about how AI agents work, how to create them, and how to use them. Be sure to include example use cases and cite specific studies and resources-- especially Python toolboxes and open source tools.")
```
