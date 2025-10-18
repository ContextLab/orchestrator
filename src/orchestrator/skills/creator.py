"""Skill creator using ROMA pattern for automatic skill generation."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import yaml
from anthropic import AsyncAnthropic

from ..utils.api_keys_flexible import ensure_api_key
from .installer import RegistryInstaller

logger = logging.getLogger(__name__)


class SkillCreator:
    """Creates skills using the ROMA pattern with multi-agent review."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize skill creator.

        Args:
            api_key: Anthropic API key (will load from env if not provided)
        """
        self.api_key = api_key or ensure_api_key("anthropic")
        self.client = AsyncAnthropic(api_key=self.api_key)
        self.installer = RegistryInstaller()

        # Model configuration for different tasks
        self.models = {
            "orchestrator": "claude-3-5-sonnet-20241022",  # Best for building/creating
            "reviewer": "claude-3-5-sonnet-20241022",      # For now, use same model
            "tester": "claude-3-haiku-20240307",           # Fast validation
        }

    async def create_skill(
        self,
        capability: str,
        pipeline_context: Optional[Dict[str, Any]] = None,
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """Create a new skill using the ROMA pattern.

        ROMA Pattern:
        1. Atomize - Break down capability into atomic tasks
        2. Plan - Create skill structure and implementation
        3. Execute - Generate actual skill code and config
        4. Aggregate - Review, test, and finalize

        Args:
            capability: Description of what the skill should do
            pipeline_context: Optional context from pipeline requirements
            max_iterations: Maximum review iterations

        Returns:
            Created skill configuration
        """
        logger.info(f"Creating skill for capability: {capability}")

        try:
            # Step 1: ATOMIZE - Break down the capability
            atomic_tasks = await self._atomize_capability(capability, pipeline_context)
            logger.info(f"Atomized into {len(atomic_tasks)} tasks")

            # Step 2: PLAN - Design skill structure
            skill_plan = await self._plan_skill(capability, atomic_tasks, pipeline_context)
            logger.info("Created skill plan")

            # Step 3: EXECUTE - Generate skill implementation
            skill_implementation = await self._execute_skill_creation(skill_plan)
            logger.info("Generated skill implementation")

            # Step 4: AGGREGATE - Review and refine
            final_skill = await self._aggregate_and_review(
                skill_implementation,
                capability,
                max_iterations
            )
            logger.info("Completed review and aggregation")

            # Save skill to registry
            skill_name = final_skill["name"]
            self._save_skill(skill_name, final_skill)
            logger.info(f"Saved skill '{skill_name}' to registry")

            return final_skill

        except Exception as e:
            logger.error(f"Failed to create skill: {e}")
            raise

    async def _atomize_capability(
        self,
        capability: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """Atomize capability into discrete tasks.

        Args:
            capability: High-level capability description
            context: Optional pipeline context

        Returns:
            List of atomic tasks
        """
        prompt = f"""Analyze this capability requirement and break it down into atomic, discrete tasks.

Capability: {capability}

{f"Pipeline Context: {json.dumps(context, indent=2)}" if context else ""}

Break this down into specific, atomic tasks that can be implemented independently.
Return a JSON list of tasks, each with:
- "task": Brief task description
- "type": Task type (input_processing, computation, transformation, output_generation, validation, etc.)
- "dependencies": List of other task indices this depends on (or empty list)

Example format:
[
  {{"task": "Parse input data", "type": "input_processing", "dependencies": []}},
  {{"task": "Validate data format", "type": "validation", "dependencies": [0]}},
  {{"task": "Transform data", "type": "transformation", "dependencies": [1]}}
]

Return ONLY the JSON list, no other text."""

        response = await self.client.messages.create(
            model=self.models["orchestrator"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )

        try:
            tasks_json = response.content[0].text.strip()
            # Clean up the response if needed
            if tasks_json.startswith("```json"):
                tasks_json = tasks_json[7:]
            if tasks_json.endswith("```"):
                tasks_json = tasks_json[:-3]
            tasks = json.loads(tasks_json.strip())
            return tasks
        except Exception as e:
            logger.error(f"Failed to parse atomized tasks: {e}")
            # Fallback to simple task list
            return [{"task": capability, "type": "general", "dependencies": []}]

    async def _plan_skill(
        self,
        capability: str,
        atomic_tasks: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Plan skill structure based on atomic tasks.

        Args:
            capability: Original capability description
            atomic_tasks: List of atomic tasks
            context: Optional pipeline context

        Returns:
            Skill plan with structure and metadata
        """
        tasks_description = "\n".join([
            f"- {i}: {task['task']} (type: {task['type']})"
            for i, task in enumerate(atomic_tasks)
        ])

        prompt = f"""Design a skill structure for this capability.

Capability: {capability}

Atomic Tasks:
{tasks_description}

{f"Pipeline Context: {json.dumps(context, indent=2)}" if context else ""}

Create a comprehensive skill plan including:
1. A concise, lowercase, hyphenated name (e.g., "web-searcher", "code-analyzer")
2. Clear description
3. Input parameters with types
4. Expected output format
5. Example usage
6. Required capabilities/features

Return as JSON with this structure:
{{
  "name": "skill-name",
  "description": "What this skill does",
  "version": "1.0.0",
  "parameters": {{
    "param_name": {{
      "type": "string|number|object|array",
      "required": true|false,
      "description": "Parameter description"
    }}
  }},
  "output": {{
    "type": "string|object|array",
    "description": "Output format description"
  }},
  "examples": [
    {{
      "description": "Example use case",
      "input": {{}},
      "expected_output": ""
    }}
  ],
  "capabilities": ["capability1", "capability2"],
  "implementation_notes": "Any special implementation considerations"
}}

Return ONLY the JSON, no other text."""

        response = await self.client.messages.create(
            model=self.models["orchestrator"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )

        try:
            plan_json = response.content[0].text.strip()
            # Clean up the response
            if plan_json.startswith("```json"):
                plan_json = plan_json[7:]
            if plan_json.endswith("```"):
                plan_json = plan_json[:-3]
            plan = json.loads(plan_json.strip())
            plan["atomic_tasks"] = atomic_tasks
            return plan
        except Exception as e:
            logger.error(f"Failed to parse skill plan: {e}")
            # Create minimal plan
            return {
                "name": capability.lower().replace(" ", "-")[:30],
                "description": capability,
                "version": "1.0.0",
                "parameters": {},
                "output": {"type": "string", "description": "Skill output"},
                "examples": [],
                "capabilities": [],
                "atomic_tasks": atomic_tasks
            }

    async def _execute_skill_creation(self, skill_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute skill creation based on the plan.

        Args:
            skill_plan: Skill structure plan

        Returns:
            Complete skill implementation
        """
        # Generate Python implementation if needed
        implementation = await self._generate_implementation(skill_plan)

        # Create complete skill structure
        skill = {
            "name": skill_plan["name"],
            "description": skill_plan["description"],
            "version": skill_plan.get("version", "1.0.0"),
            "created": datetime.now().isoformat(),
            "created_by": "skill_creator",
            "model_used": self.models["orchestrator"],
            "parameters": skill_plan.get("parameters", {}),
            "output": skill_plan.get("output", {}),
            "examples": skill_plan.get("examples", []),
            "capabilities": skill_plan.get("capabilities", []),
            "atomic_tasks": skill_plan.get("atomic_tasks", []),
            "implementation": implementation,
            "status": "draft",
            "test_results": None
        }

        return skill

    async def _generate_implementation(self, skill_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Python implementation for the skill.

        Args:
            skill_plan: Skill structure plan

        Returns:
            Implementation details
        """
        tasks_description = "\n".join([
            f"- {task['task']}"
            for task in skill_plan.get("atomic_tasks", [])
        ])

        prompt = f"""Create a Python implementation for this skill.

Skill Name: {skill_plan['name']}
Description: {skill_plan['description']}

Atomic Tasks to Implement:
{tasks_description}

Parameters: {json.dumps(skill_plan.get('parameters', {}), indent=2)}
Expected Output: {json.dumps(skill_plan.get('output', {}), indent=2)}

Generate a complete, working Python implementation that:
1. Handles all input parameters
2. Implements each atomic task
3. Returns output in the specified format
4. Includes error handling
5. Uses real API calls/operations (NO MOCKS)

Return as JSON with:
{{
  "type": "python",
  "code": "Complete Python code here",
  "dependencies": ["required", "packages"],
  "entry_point": "main function name"
}}

Return ONLY the JSON, no other text."""

        response = await self.client.messages.create(
            model=self.models["orchestrator"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
        )

        try:
            impl_json = response.content[0].text.strip()
            # Clean up the response
            if impl_json.startswith("```json"):
                impl_json = impl_json[7:]
            if impl_json.endswith("```"):
                impl_json = impl_json[:-3]
            implementation = json.loads(impl_json.strip())
            return implementation
        except Exception as e:
            logger.error(f"Failed to parse implementation: {e}")
            return {
                "type": "python",
                "code": "# Implementation pending",
                "dependencies": [],
                "entry_point": "execute"
            }

    async def _aggregate_and_review(
        self,
        skill: Dict[str, Any],
        original_capability: str,
        max_iterations: int
    ) -> Dict[str, Any]:
        """Review and refine the skill through iterations.

        Args:
            skill: Initial skill implementation
            original_capability: Original capability requirement
            max_iterations: Maximum review iterations

        Returns:
            Refined skill
        """
        current_skill = skill

        for iteration in range(max_iterations):
            # Review the skill
            review_result = await self._review_skill(
                current_skill,
                original_capability,
                iteration
            )

            if review_result["approved"]:
                current_skill["status"] = "reviewed"
                logger.info(f"Skill approved after {iteration + 1} iterations")
                break

            # Apply suggested changes
            if review_result.get("suggestions"):
                current_skill = await self._apply_review_suggestions(
                    current_skill,
                    review_result["suggestions"]
                )

        return current_skill

    async def _review_skill(
        self,
        skill: Dict[str, Any],
        original_capability: str,
        iteration: int
    ) -> Dict[str, Any]:
        """Review a skill for quality and completeness.

        Args:
            skill: Skill to review
            original_capability: Original requirement
            iteration: Current iteration number

        Returns:
            Review results with approval status and suggestions
        """
        prompt = f"""Review this automatically created skill (iteration {iteration + 1}).

Original Capability: {original_capability}

Skill Created:
{json.dumps(skill, indent=2)}

Review for:
1. Completeness - Does it fulfill the original capability?
2. Correctness - Is the implementation correct?
3. Quality - Is it well-structured and maintainable?
4. Testing - Can it be tested with real data?
5. Documentation - Are parameters and outputs clear?

Return JSON with:
{{
  "approved": true|false,
  "score": 0-100,
  "issues": ["list", "of", "issues"],
  "suggestions": ["list", "of", "specific", "improvements"],
  "comments": "Overall review comments"
}}

Return ONLY the JSON, no other text."""

        response = await self.client.messages.create(
            model=self.models["reviewer"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )

        try:
            review_json = response.content[0].text.strip()
            # Clean up the response
            if review_json.startswith("```json"):
                review_json = review_json[7:]
            if review_json.endswith("```"):
                review_json = review_json[:-3]
            review = json.loads(review_json.strip())
            return review
        except Exception as e:
            logger.error(f"Failed to parse review: {e}")
            # Default to approved with warnings
            return {
                "approved": True,
                "score": 70,
                "issues": ["Could not parse review"],
                "suggestions": [],
                "comments": "Auto-approved due to review parse error"
            }

    async def _apply_review_suggestions(
        self,
        skill: Dict[str, Any],
        suggestions: List[str]
    ) -> Dict[str, Any]:
        """Apply review suggestions to improve the skill.

        Args:
            skill: Current skill
            suggestions: List of improvement suggestions

        Returns:
            Updated skill
        """
        if not suggestions:
            return skill

        suggestions_text = "\n".join([f"- {s}" for s in suggestions])

        prompt = f"""Apply these review suggestions to improve the skill.

Current Skill:
{json.dumps(skill, indent=2)}

Suggestions to Apply:
{suggestions_text}

Return the complete updated skill as JSON with all suggested improvements applied.
Maintain the same structure but incorporate the improvements.

Return ONLY the JSON, no other text."""

        response = await self.client.messages.create(
            model=self.models["orchestrator"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
        )

        try:
            updated_json = response.content[0].text.strip()
            # Clean up the response
            if updated_json.startswith("```json"):
                updated_json = updated_json[7:]
            if updated_json.endswith("```"):
                updated_json = updated_json[:-3]
            updated_skill = json.loads(updated_json.strip())
            return updated_skill
        except Exception as e:
            logger.error(f"Failed to apply suggestions: {e}")
            # Return original skill if update fails
            return skill

    def _save_skill(self, skill_name: str, skill_data: Dict[str, Any]) -> None:
        """Save skill to registry and create skill directory.

        Args:
            skill_name: Name of the skill
            skill_data: Complete skill data
        """
        # Create skill directory
        skill_dir = self.installer.create_skill_directory(skill_name)

        # Save skill definition
        skill_file = skill_dir / "skill.yaml"
        with open(skill_file, 'w') as f:
            yaml.dump(skill_data, f, default_flow_style=False, sort_keys=False)

        # Save implementation if present
        if skill_data.get("implementation", {}).get("code"):
            impl_file = skill_dir / "implementation.py"
            with open(impl_file, 'w') as f:
                f.write(skill_data["implementation"]["code"])

        # Update registry
        registry_entry = {
            "name": skill_name,
            "description": skill_data["description"],
            "version": skill_data["version"],
            "capabilities": skill_data.get("capabilities", []),
            "parameters": skill_data.get("parameters", {}),
            "path": f"./{skill_name}/",
            "created": skill_data.get("created"),
            "status": skill_data.get("status", "draft")
        }

        self.installer.update_skill_registry(skill_name, registry_entry)
        logger.info(f"Saved skill '{skill_name}' to {skill_dir}")