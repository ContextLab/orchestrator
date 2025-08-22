# 🔧 Orchestrator: Core Development & Refactoring Plan

This issue contains a comprehensive action plan derived from a design/debugging monologue. It outlines technical debt, missing features, and critical improvements needed to get the `Orchestrator` toolbox production-ready. Each major item should become its own GitHub issue or project task.

---

## ✅ 1. Repository Hygiene & Issue Cleanup

- [ ] Review and triage **all open GitHub issues**:
  - Close obsolete or redundant issues
  - Merge overlapping ones
- [ ] Ensure no stale "scratchpad" files are left in the repo
- [ ] Convert local notes/scripts into GitHub issues if they contain unresolved problems
- [ ] Move toward tracking all dev work through GitHub (not ephemeral scratch files)

---

## 🧪 2. Replace Mockups with Real Tests

- [ ] Remove or minimize use of **mock objects** and placeholder simulations in tests
- [ ] Implement **end-to-end functional tests** that:
  - Start/stop actual services (e.g., Docker, subprocesses)
  - Make real API calls where applicable
- [ ] Identify and replace any remaining placeholder/stubbed functionality

---

## 🧼 3. Codebase Cleanup & Consolidation

- [ ] Remove outdated or broken example scripts
- [ ] Ensure no lingering “simulated” or placeholder logic is in production code
- [ ] Unify model registry references:
  - Currently split between local project folder and `$HOME/.orchestrator`
  - Decide on single source of truth and update usage throughout codebase
- [ ] Clean up the **pipeline status tracker** system:
  - Remove orphaned or corrupted logs
  - Make the resume/restart mechanism robust and sustainable

---

## 🛠️ 4. Tool Registry & Tooling Infrastructure

Design and implement a **registry-based tool abstraction** where each tool has:

- Clearly defined `input`, `output`, and `process` logic
- Optional support for:
  - Pipelines as tools (nested pipelines)
  - GUI/CLI interactivity
  - Docker-based execution
  - Remote or local context

Tool categories to support:

- [ ] 🧠 LLM interaction (task delegation, multi-model routing)
- [ ] 🕸️ Browsers (headless & visual)
  - With screenshot → vision model support
  - Safe handling of JavaScript/CSS-rendered pages
- [ ] 🧪 Code execution
  - Controlled via Docker or subprocess
  - Specify packages, timeouts, and resource limits
- [ ] 🔁 Pipeline recursion & nesting
  - Prevent infinite loops
  - Ensure inputs/outputs are composable
- [ ] 👤 User interaction tools
  - Blocking vs non-blocking prompts
  - Supports human or LLM "users"
- [ ] 📁 File I/O tools
  - Read/write/delete files and directories
- [ ] 📚 PDF/Markdown compilation & parsing
  - `pandoc`, `latex`, markdown → PDF workflows
  - Verify PDF output visually or by rules
- [ ] 🖼️ Multimodal content handling
  - Allow model access to image/text combos (e.g. browser screenshots)
- [ ] 🔍 Search / Retrieval
  - Web search and format-correction tools
- [ ] 💬 Model Context Protocol tools
  - Interact with MxP-native tools and memory

---

## 🧭 5. Pipeline Control Flow

Reinstate advanced control flow features:

- [ ] Conditional execution (if/else branches)
- [ ] Looping:
  - `for` loops with iterator sourced from earlier steps
  - `while` loops with stopping conditions
- [ ] `goto`-like functionality for jumping to other steps based on results
- [ ] Dependency chaining between steps

---

## 🧩 6. Model Routing / Assignment

- [ ] Enable **explicit model-to-task assignments** in pipeline YAML
- [ ] Use local/small/cheap models for simple syntax resolution, and heavy models for generation
- [ ] Dynamically switch models depending on domain or pipeline step

---

## 📚 7. Documentation Overhaul

Once the refactor is complete:

- [ ] Purge outdated/incorrect docs
- [ ] Rewrite documentation to reflect current pipeline & tool APIs
- [ ] Add clear usage examples that:
  - Run successfully
  - Produce high-quality outputs
- [ ] Every code snippet in docs must be **covered by a test**

---

## 🔬 8. Tests for Every Snippet

- [ ] Every piece of logic or example must have a test
- [ ] Tests must verify:
  - Functional correctness
  - Output quality
  - Error handling and fallbacks

---

## 🎯 9. Final Goal

- A modular, tested, reliable orchestration system that supports:
  - Multi-model pipelines
  - Dynamic tool invocation
  - Recursion, conditionals, looping
  - GUI/CLI integration
  - Real-world code execution & content generation

---

Let’s break this plan into discrete, trackable GitHub issues. Once done, LLMs can work through them one by one. ✅
