"""Enhanced compiler with automatic skill creation for Claude Skills refactor."""

import logging
from typing import Dict, Any, Optional

from .yaml_compiler import YAMLCompiler
from ..core.pipeline import Pipeline
from ..skills.creator import SkillCreator
from ..skills.registry import SkillRegistry
from ..core.exceptions import YAMLCompilerError

logger = logging.getLogger(__name__)


class SkillsCompiler(YAMLCompiler):
    """
    Enhanced YAML compiler with automatic skill creation.

    This compiler extends the standard YAMLCompiler to automatically:
    1. Check if required skills exist in the registry
    2. Create missing skills using the SkillCreator (with ROMA pattern)
    3. Integrate skill execution into pipeline steps
    4. Compile directly to LangGraph without LLM prompting in compilation path
    """

    def __init__(self, skill_creator: Optional[SkillCreator] = None, **kwargs):
        """Initialize enhanced compiler with skills support.

        Args:
            skill_creator: SkillCreator instance for auto-creation
            **kwargs: Additional arguments passed to YAMLCompiler
        """
        super().__init__(**kwargs)
        self.skill_registry = SkillRegistry()
        self.skill_creator = skill_creator or SkillCreator()
        self.auto_create_skills = True  # Enable automatic skill creation

    async def compile(
        self,
        yaml_content: str,
        context: Optional[Dict[str, Any]] = None,
        resolve_ambiguities: bool = True,
        auto_create_missing_skills: bool = True,
    ) -> Pipeline:
        """
        Compile YAML content to Pipeline with automatic skill creation.

        Args:
            yaml_content: YAML pipeline definition
            context: Template context variables
            resolve_ambiguities: Whether to resolve AUTO tags
            auto_create_missing_skills: Whether to auto-create missing skills

        Returns:
            Compiled Pipeline object with all skills available

        Raises:
            YAMLCompilerError: If compilation fails
        """
        logger.info("Starting skills-aware compilation")

        try:
            # Parse YAML to get pipeline definition
            raw_pipeline = self._parse_yaml(yaml_content)

            # Check for required skills
            if auto_create_missing_skills:
                await self._ensure_required_skills(raw_pipeline)

            # Perform standard compilation with all validations
            pipeline = await super().compile(
                yaml_content,
                context=context,
                resolve_ambiguities=resolve_ambiguities
            )

            logger.info(f"Compiled pipeline '{pipeline.id}' with {len(pipeline.tasks)} tasks")
            return pipeline

        except Exception as e:
            logger.error(f"Skills-aware compilation failed: {e}")
            raise YAMLCompilerError(f"Failed to compile with skills support: {e}") from e

    async def _ensure_required_skills(self, pipeline_def: Dict[str, Any]) -> None:
        """Ensure all required skills exist, creating them if necessary.

        Args:
            pipeline_def: Parsed pipeline definition

        Raises:
            YAMLCompilerError: If skill creation fails
        """
        steps = pipeline_def.get("steps", [])
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

        logger.info(f"Found {len(missing_skills)} missing skills, will auto-create")

        # Create missing skills
        for skill_info in missing_skills:
            await self._create_skill_for_step(
                skill_info["name"],
                skill_info["step"],
                pipeline_def
            )

    def _extract_skill_name(self, step: Dict[str, Any]) -> Optional[str]:
        """Extract skill name from step definition.

        Args:
            step: Step definition

        Returns:
            Skill name if specified, None otherwise
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
                "code_analysis": "code-analyzer",
                "text_generation": "text-generator",
                "data_transformation": "data-transformer",
            }
            return action_to_skill.get(action)

        # Check if tool field could be a skill
        if "tool" in step:
            tool = step["tool"]
            # If tool looks like a skill name (hyphenated), treat as skill
            if "-" in tool and not any(c in tool for c in ["/", ".", ":"]):
                return tool

        return None

    async def _create_skill_for_step(
        self,
        skill_name: str,
        step: Dict[str, Any],
        pipeline_context: Dict[str, Any]
    ) -> None:
        """Create a skill for a pipeline step.

        Args:
            skill_name: Name of the skill to create
            step: Step definition requiring the skill
            pipeline_context: Full pipeline definition for context

        Raises:
            YAMLCompilerError: If skill creation fails
        """
        logger.info(f"Auto-creating skill: {skill_name}")

        try:
            # Extract capability requirement from step
            capability = self._infer_capability_from_step(step, skill_name)

            # Create the skill using ROMA pattern
            await self.skill_creator.create_skill(
                capability=capability,
                pipeline_context={
                    "pipeline_id": pipeline_context.get("id", "unknown"),
                    "pipeline_description": pipeline_context.get("description"),
                    "step_context": step,
                    "full_pipeline": pipeline_context
                },
                max_iterations=3
            )

            logger.info(f"Successfully created skill '{skill_name}' for step '{step.get('id')}'")

        except Exception as e:
            error_msg = f"Failed to create skill '{skill_name}' for step '{step.get('id')}': {e}"
            logger.error(error_msg)
            raise YAMLCompilerError(error_msg) from e

    def _infer_capability_from_step(self, step: Dict[str, Any], skill_name: str) -> str:
        """Infer what capability is required from step definition.

        Args:
            step: Step definition
            skill_name: Name of the skill

        Returns:
            Capability description string
        """
        # Build capability description from available information
        parts = []

        # Use step name or id
        step_id = step.get("name") or step.get("id", "unknown")
        parts.append(f"for step '{step_id}'")

        # Add action if specified
        if "action" in step:
            parts.append(f"to perform action: {step['action']}")

        # Add description if available
        if "description" in step:
            parts.append(f"({step['description']})")

        # Check parameters for clues
        parameters = step.get("parameters", {})
        if parameters:
            param_hints = []
            if "url" in parameters:
                param_hints.append("fetches from URL")
            if "query" in parameters:
                param_hints.append("performs search or query")
            if "code" in parameters or "file" in parameters:
                param_hints.append("analyzes code or files")
            if "data" in parameters:
                param_hints.append("processes data")

            if param_hints:
                parts.append("that " + ", ".join(param_hints))

        capability = f"Create skill '{skill_name}' " + " ".join(parts)

        # If we couldn't infer much, use a generic capability
        if len(parts) <= 1:
            capability = f"Create skill '{skill_name}' that handles the step's requirements based on its parameters and context"

        return capability

    def compile_to_langgraph_spec(self, yaml_content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compile pipeline YAML directly to LangGraph specification.

        This method compiles the pipeline without LLM prompting in the compilation path,
        creating a LangGraph specification that can be directly executed.

        Args:
            yaml_content: YAML pipeline definition
            context: Template context variables

        Returns:
            LangGraph specification as dictionary
        """
        logger.info("Compiling to LangGraph specification (no LLM prompting)")

        # Parse YAML
        pipeline_def = self._parse_yaml(yaml_content)

        # Build LangGraph spec
        spec = {
            "graph_type": "StateGraph",
            "state_schema": {
                "pipeline_id": "str",
                "execution_id": "str",
                "current_step": "Optional[str]",
                "variables": "Dict[str, Any]",
                "step_outputs": "Dict[str, Dict[str, Any]]",
                "completed_steps": "List[str]",
                "failed_steps": "List[str]",
                "errors": "List[Dict[str, Any]]",
            },
            "nodes": [],
            "edges": [],
            "entry_point": "START",
            "finish_points": []
        }

        # Add nodes for each step
        for step in pipeline_def.get("steps", []):
            node = {
                "id": step["id"],
                "name": step.get("name", step["id"]),
                "type": "skill" if self._extract_skill_name(step) else "tool",
                "skill": self._extract_skill_name(step),
                "action": step.get("action", step.get("tool")),
                "parameters": step.get("parameters", {}),
                "metadata": step.get("metadata", {})
            }
            spec["nodes"].append(node)

        # Add edges based on dependencies
        for step in pipeline_def.get("steps", []):
            step_id = step["id"]
            dependencies = step.get("dependencies", step.get("depends_on", []))

            if isinstance(dependencies, str):
                dependencies = [dependencies]

            if dependencies:
                for dep in dependencies:
                    spec["edges"].append({
                        "from": dep,
                        "to": step_id
                    })
            else:
                # No dependencies - connect to START
                spec["edges"].append({
                    "from": "START",
                    "to": step_id
                })

        # Find terminal nodes (no dependents)
        all_step_ids = {step["id"] for step in pipeline_def.get("steps", [])}
        steps_with_dependents = {edge["from"] for edge in spec["edges"] if edge["from"] != "START"}
        terminal_nodes = all_step_ids - steps_with_dependents

        for node_id in terminal_nodes:
            spec["edges"].append({
                "from": node_id,
                "to": "END"
            })
            spec["finish_points"].append(node_id)

        logger.info(f"Created LangGraph spec with {len(spec['nodes'])} nodes and {len(spec['edges'])} edges")
        return spec

    def get_compilation_summary(self) -> Dict[str, Any]:
        """Get summary of the compilation process.

        Returns:
            Summary with skills created, validations performed, etc.
        """
        return {
            "skills_checked": True,
            "auto_creation_enabled": self.auto_create_skills,
            "registry_location": str(self.skill_registry.registry_dir),
            "validation_enabled": self.enable_validation_report,
            "validation_level": str(self.validation_level) if hasattr(self, 'validation_level') else "unknown"
        }
