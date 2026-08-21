# Documentation writing style

Orchestrator documentation should be exact before it is expansive. A page may describe a feature as supported only when the product contract and its tests do. Anything else should be labeled experimental, unverified, or historical.

This project adapts the parts of the ContextLab writing guide that transfer to technical documentation. The source guide explicitly says that it has no validated profile for project documentation, so this is a selective adaptation, not a claim that the scientific-paper profile applies unchanged.

## Write for a reader with a task

- Start with what the reader will accomplish.
- Introduce a term where the reader first needs it, then use one spelling for it.
- Put a working example immediately after an abstract explanation.
- Restate formal behavior in plain English when the consequence is not obvious.
- Prefer concrete claims over promotional adjectives.

## Make evidence visible

- Link claims about supported behavior to a contract-tested example or reference.
- Distinguish “present in the repository” from “supported.”
- State prerequisites, side effects, expected output, and intentional failures.
- Never invent output. Run the example or say why it was not run.
- Use sentence-case headings and avoid decorative emoji.

## Keep the prose direct

- Address the reader as “you” in procedures.
- Use active voice when the actor matters: “The compiler rejects the pipeline.”
- Use passive voice when the actor does not matter: “The result is serialized.”
- Prefer ordinary words. Avoid “powerful,” “seamless,” “easy,” “advanced,” and similar claims unless the page defines and demonstrates them.
- Use em dashes sparingly. Parentheses should hold short clarifications, not the main instruction.

## Structure tutorials consistently

Every tutorial answers these questions, in this order:

1. What will I build or learn?
2. What do I need before I start?
3. Which file am I using?
4. What should I run?
5. What should happen?
6. Why did it happen?
7. What should I try next?

Tutorials use files under `examples/supported/`. Those files are tested through both the CLI and Python API. A legacy example is not a tutorial source until it joins that supported set.

## Review checklist

Before merging documentation, check that:

- every command works from the repository root;
- code and YAML parse without omitted “obvious” lines;
- links resolve in the built documentation;
- terminology agrees with the product contract;
- unsupported providers and features are labeled accurately;
- examples do not require credentials unless the tutorial says so; and
- failures are identified as intentional or filed under the documentation overhaul tracking issue.
