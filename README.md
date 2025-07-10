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
write-research-report

inputs:
 - topic: {report-topic}
 - instructions: {report-instructions}

outputs:
 - pdf: <AUTO>come up with an appropriate filename, ending in .pdf</AUTO>
 - tex: {pdf}[:-3] + '.tex'  # we can refer to the values of other attributes using curly braces

pipeline:
 - web-search:
  - action: <AUTO>search the web for <AUTO>construct an appropriate web query about {topic}, using these additional instructions: {instructions}</AUTO></AUTO>
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
        - until: <AUTO>all sources have been verified</AUTO>
  - produces: markdown-file
  - location: ./searches/{pdf}/compiled_results_corrected.md
 - TODO: construct report w/ appropriate sections, then make a list of all claims + check each one, then do a final check for consistency and accuracy, then create a latex file, then compile into a pdf
      

```
