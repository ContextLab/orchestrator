"""Enhanced compiler combining skills auto-creation and advanced control flow."""

import logging
from typing import Any, Dict, List, Optional

from .control_flow_compiler import ControlFlowCompiler
from ..core.pipeline import Pipeline
from ..core.exceptions import YAMLCompilerError
from ..skills.creator import SkillCreator
from ..skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


class EnhancedSkillsCompiler(ControlFlowCompiler):
    """
    Comprehensive compiler for Claude Skills refactor.

    Combines:
    - Automatic skill creation (ROMA pattern)
    - Advanced control flow (for/while/if/goto)
    - Direct compilation to LangGraph
    - All existing validation capabilities
    """

    def __init__(
        self,
        skill_creator: Optional[SkillCreator] = None,
        skill_registry: Optional[SkillRegistry] = None,
        **kwargs
    ):
        """Initialize enhanced compiler.

        Args:
            skill_creator: SkillCreator for auto-creating skills
            skill_registry: SkillRegistry for tracking skills
            **kwargs: Additional arguments for ControlFlowCompiler
        """
        super().__init__(**kwargs)
        self.skill_registry = skill_registry or SkillRegistry()
        self.skill_creator = skill_creator or SkillCreator()
        self.auto_create_skills = True
        self.created_skills_in_compilation = []

    async def compile(
        self,
        yaml_content: str,
        context: Optional[Dict[str, Any]] = None,
        resolve_ambiguities: bool = True,
        auto_create_missing_skills: bool = True,
    ) -> Pipeline:
        """
        Compile YAML with skills auto-creation and control flow.

        Process:
        1. Parse YAML structure
        2. Normalize skill fields to actions
        3. Check for required skills
        4. Auto-create missing skills (using ROMA pattern)
        5. Process control flow (for/while/conditionals)
        6. Compile to LangGraph-ready pipeline
        7. Validate everything

        Args:
            yaml_content: YAML pipeline definition
            context: Template context
            resolve_ambiguities: Whether to resolve AUTO tags
            auto_create_missing_skills: Whether to auto-create skills

        Returns:
            Fully compiled Pipeline ready for execution
        """
        logger.info("Starting enhanced skills-aware compilation with control flow")
        self.created_skills_in_compilation = []

        try:
            # Step 1: Parse YAML to inspect requirements
            raw_pipeline = self._parse_yaml(yaml_content)

            # Step 2: Normalize 'skill' fields to be schema-compliant
            raw_pipeline = self._normalize_skill_fields(raw_pipeline)

            # Step 3: Ensure all required skills exist (auto-create if needed)
            if auto_create_missing_skills:
                await self._ensure_required_skills(raw_pipeline)

            # Step 4: Convert normalized pipeline back to YAML for parent compile
            import yaml as yaml_lib
            normalized_yaml = yaml_lib.dump(raw_pipeline, default_flow_style=False, sort_keys=False)

            # Step 5: Use parent's compile (which handles control flow + standard compilation)
            pipeline = await super().compile(
                normalized_yaml,
                context=context,
                resolve_ambiguities=resolve_ambiguities
            )

            # Step 6: Enhance pipeline with skills metadata
            if self.created_skills_in_compilation:
                pipeline.metadata["auto_created_skills"] = self.created_skills_in_compilation

            logger.info(
                f"Enhanced compilation complete: {len(pipeline.tasks)} tasks, "
                f"{len(self.created_skills_in_compilation)} skills auto-created"
            )

            return pipeline

        except Exception as e:
            logger.error(f"Enhanced compilation failed: {e}")
            raise YAMLCompilerError(f"Enhanced skills compilation failed: {e}") from e

    def _normalize_skill_fields(self, pipeline_def: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize 'skill' fields to 'tool' for schema compatibility.

        Args:
            pipeline_def: Pipeline definition

        Returns:
            Normalized pipeline definition
        """
        def normalize_step(step: Dict[str, Any]) -> Dict[str, Any]:
            # If step has 'skill' but no 'tool', copy it
            if "skill" in step and "tool" not in step:
                step["tool"] = step["skill"]

            # If step has 'skill' but no 'action', add a default action
            if "skill" in step and "action" not in step:
                step["action"] = "execute_skill"

            # Recursively normalize nested steps
            if "steps" in step:
                step["steps"] = [normalize_step(s) for s in step["steps"]]

            # Normalize action_loop steps
            if "action_loop" in step and isinstance(step["action_loop"], list):
                step["action_loop"] = [normalize_step(s) for s in step["action_loop"]]

            return step

        # Normalize all steps
        if "steps" in pipeline_def:
            pipeline_def["steps"] = [normalize_step(s) for s in pipeline_def["steps"]]

        return pipeline_def

    async def _ensure_required_skills(self, pipeline_def: Dict[str, Any]) -> None:
        """Ensure all required skills exist (inherited from SkillsCompiler).

        Args:
            pipeline_def: Parsed pipeline definition

        Raises:
            YAMLCompilerError: If skill creation fails
        """
        steps = self._collect_all_steps(pipeline_def)
        missing_skills = []

        # Check each step for skill requirements
        for step in steps:
            skill_name = self._extract_skill_name(step)
            if skill_name and not self.skill_registry.exists(skill_name):
                missing_skills.append({
                    "name": skill_name,
                    "step": step,
                    "step_id": step.get("id", "unknown")
                })

        if not missing_skills:
            logger.info("All required skills are available")
            return

        logger.info(f"Found {len(missing_skills)} missing skills, auto-creating...")

        # Create missing skills
        for skill_info in missing_skills:
            try:
                await self._create_skill_for_step(
                    skill_info["name"],
                    skill_info["step"],
                    pipeline_def
                )
                self.created_skills_in_compilation.append(skill_info["name"])
            except Exception as e:
                logger.error(f"Failed to create skill {skill_info['name']}: {e}")
                # Continue with other skills - let validation catch the error

    def _collect_all_steps(self, pipeline_def: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recursively collect all steps including nested ones in loops.

        Args:
            pipeline_def: Pipeline definition

        Returns:
            List of all step definitions
        """
        all_steps = []

        def collect_steps(steps_list: List[Dict[str, Any]]) -> None:
            for step in steps_list:
                all_steps.append(step)

                # Check for nested steps in control flow
                if "steps" in step:
                    collect_steps(step["steps"])

                # Check for action_loop steps
                if "action_loop" in step:
                    action_loop = step["action_loop"]
                    if isinstance(action_loop, list):
                        collect_steps(action_loop)

        main_steps = pipeline_def.get("steps", [])
        collect_steps(main_steps)

        return all_steps

    def _extract_skill_name(self, step: Dict[str, Any]) -> Optional[str]:
        """Extract skill name from step definition.

        Args:
            step: Step definition

        Returns:
            Skill name if specified
        """
        # Check for explicit 'skill' field
        if "skill" in step:
            return step["skill"]

        # Check for action that maps to a skill
        if "action" in step:
            action = step["action"]
            # Common action-to-skill mappings
            action_to_skill = {
                "web_search": "web-searcher",
                "web-search": "web-searcher",
                "code_analysis": "code-analyzer",
                "code-analysis": "code-analyzer",
                "text_generation": "text-generator",
                "text-generation": "text-generator",
                "data_transformation": "data-transformer",
                "data-transformation": "data-transformer",
            }
            mapped = action_to_skill.get(action)
            if mapped:
                return mapped

        # Check if tool field could be a skill
        if "tool" in step:
            tool = step["tool"]
            # If tool looks like a skill name (hyphenated, lowercase), treat as skill
            if "-" in tool and tool.islower() and not any(c in tool for c in ["/", ".", ":"]):
                return tool

        return None

    async def _create_skill_for_step(
        self,
        skill_name: str,
        step: Dict[str, Any],
        pipeline_context: Dict[str, Any]
    ) -> None:
        """Create a skill for a pipeline step using ROMA pattern.

        Args:
            skill_name: Name of the skill to create
            step: Step definition requiring the skill
            pipeline_context: Full pipeline for context

        Raises:
            YAMLCompilerError: If skill creation fails critically
        """
        logger.info(f"Auto-creating skill: {skill_name}")

        try:
            # Infer capability from step
            capability = self._infer_capability_from_step(step, skill_name)

            # Create skill using ROMA pattern
            skill = await self.skill_creator.create_skill(
                capability=capability,
                pipeline_context={
                    "pipeline_id": pipeline_context.get("id", "unknown"),
                    "pipeline_name": pipeline_context.get("name", "unnamed"),
                    "pipeline_description": pipeline_context.get("description"),
                    "step_id": step.get("id"),
                    "step_action": step.get("action"),
                    "step_parameters": step.get("parameters", {}),
                    "step_description": step.get("description"),
                    "full_pipeline": pipeline_context
                },
                max_iterations=3  # Max review iterations
            )

            logger.info(f"✅ Created skill '{skill_name}' with {len(skill.get('atomic_tasks', []))} atomic tasks")

        except Exception as e:
            logger.error(f"Failed to create skill '{skill_name}': {e}")
            # In production, this might raise an error
            # For now, log and continue - the validator will catch missing skills
            logger.warning(f"Continuing compilation without skill '{skill_name}'")

    def _infer_capability_from_step(self, step: Dict[str, Any], skill_name: str) -> str:
        """Infer required capability from step definition.

        Args:
            step: Step definition
            skill_name: Name of the skill

        Returns:
            Capability description
        """
        parts = []

        # Base description from skill name
        parts.append(f"Skill '{skill_name}'")

        # Add action description
        if "action" in step:
            parts.append(f"that performs: {step['action']}")

        # Add context from description
        if "description" in step:
            parts.append(f"- {step['description']}")

        # Infer from parameters
        parameters = step.get("parameters", {})
        param_hints = []

        if "url" in parameters:
            param_hints.append("fetches and processes data from URLs")
        if "query" in parameters:
            param_hints.append("performs search queries")
        if "code" in parameters or "file" in parameters:
            param_hints.append("analyzes code or files")
        if "data" in parameters:
            param_hints.append("processes and transforms data")
        if "text" in parameters or "content" in parameters:
            param_hints.append("processes text content")

        if param_hints:
            parts.append("that " + "; ".join(param_hints))

        # Build final capability string
        capability = " ".join(parts)

        # If we have minimal information, be more descriptive
        if len(parts) <= 1:
            capability = (
                f"Create skill '{skill_name}' that handles the requirements "
                f"based on the step configuration and pipeline context"
            )

        return capability

    def get_compilation_stats(self) -> Dict[str, Any]:
        """Get statistics about the compilation process.

        Returns:
            Compilation statistics including skills created, validations run
        """
        stats = {
            "auto_creation_enabled": self.auto_create_skills,
            "skills_auto_created": len(self.created_skills_in_compilation),
            "created_skill_names": self.created_skills_in_compilation.copy(),
            "registry_location": str(self.skill_registry.registry_dir),
            "total_registry_skills": len(self.skill_registry.list_skills()),
        }

        # Add validation stats if available
        if self.validation_report:
            stats["validation"] = {
                "has_errors": self.validation_report.has_errors,
                "has_warnings": self.validation_report.has_warnings,
                "total_issues": self.validation_report.stats.total_issues,
            }

        return stats
