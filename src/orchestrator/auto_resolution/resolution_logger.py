"""Logging integration for AUTO tag resolution."""

import logging
import time
from typing import Any, Dict, List, Optional
from datetime import datetime

from .models import AutoTagContext, AutoTagResolution

logger = logging.getLogger(__name__)


class ResolutionLogger:
    """Logs AUTO tag resolution process with checkpoint integration."""
    
    def __init__(self, checkpoint_resolutions: bool = True):
        self.checkpoint_resolutions = checkpoint_resolutions
        self.resolution_history: List[AutoTagResolution] = []
    
    def log_resolution_start(self, tag: str, context: AutoTagContext):
        """Log the start of AUTO tag resolution."""
        logger.debug(
            f"AUTO Tag Resolution - Starting resolution",
            extra={
                "tag": tag[:100] + "..." if len(tag) > 100 else tag,
                "location": context.tag_location,
                "task_id": context.current_task_id,
                "pipeline_id": context.pipeline.id,
                "resolution_depth": context.resolution_depth,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def log_model_attempt(self, model: str, retry_count: int):
        """Log attempt with specific model."""
        logger.debug(
            f"AUTO Tag Resolution - Attempting with model: {model}",
            extra={
                "model": model,
                "retry_count": retry_count,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def log_pass_start(self, pass_name: str, model: str, context: Dict[str, Any]):
        """Log the start of a resolution pass."""
        logger.debug(
            f"AUTO Tag Resolution - Starting {pass_name}",
            extra={
                "pass": pass_name,
                "model": model,
                "tag_preview": context.get("tag", "")[:50],
                "location": context.get("location", ""),
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def log_pass_result(
        self, 
        pass_name: str, 
        result: Any, 
        duration_ms: int,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Log the result of a resolution pass."""
        log_data = {
            "pass": pass_name,
            "duration_ms": duration_ms,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        
        if success:
            # Safely preview result
            try:
                result_preview = str(result)[:200]
            except Exception:
                result_preview = f"<{type(result).__name__} object>"
            
            log_data["result_preview"] = result_preview
            logger.debug(f"AUTO Tag Resolution - Completed {pass_name}", extra=log_data)
        else:
            log_data["error"] = error
            logger.warning(f"AUTO Tag Resolution - Failed {pass_name}", extra=log_data)
    
    def log_requirements_analysis(self, requirements: Dict[str, Any], duration_ms: int):
        """Log requirements analysis results."""
        logger.debug(
            "AUTO Tag Resolution - Requirements analyzed",
            extra={
                "tools_needed": requirements.get("tools_needed", []),
                "output_format": requirements.get("output_format"),
                "expected_type": requirements.get("expected_output_type"),
                "dependencies": requirements.get("data_dependencies", []),
                "duration_ms": duration_ms,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def log_prompt_construction(self, prompt_data: Dict[str, Any], duration_ms: int):
        """Log prompt construction results."""
        logger.debug(
            "AUTO Tag Resolution - Prompt constructed",
            extra={
                "target_model": prompt_data.get("target_model"),
                "prompt_length": len(prompt_data.get("prompt", "")),
                "has_schema": "output_schema" in prompt_data,
                "temperature": prompt_data.get("temperature"),
                "duration_ms": duration_ms,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def log_resolution_execution(self, resolved_value: Any, duration_ms: int):
        """Log resolution execution results."""
        try:
            value_type = type(resolved_value).__name__
            if isinstance(resolved_value, (str, int, float, bool)):
                value_preview = str(resolved_value)[:100]
            elif isinstance(resolved_value, (list, dict)):
                value_preview = f"{value_type} with {len(resolved_value)} items"
            else:
                value_preview = f"<{value_type} object>"
        except Exception:
            value_preview = "<unprintable>"
            
        logger.debug(
            "AUTO Tag Resolution - Value resolved",
            extra={
                "value_type": value_type,
                "value_preview": value_preview,
                "duration_ms": duration_ms,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def log_action_determination(self, action_plan: Dict[str, Any], duration_ms: int):
        """Log action determination results."""
        logger.debug(
            "AUTO Tag Resolution - Action determined",
            extra={
                "action_type": action_plan.get("action_type"),
                "tool_name": action_plan.get("tool_name"),
                "has_file_path": "file_path" in action_plan,
                "has_context_updates": "context_updates" in action_plan,
                "duration_ms": duration_ms,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def log_resolution_complete(self, resolution: AutoTagResolution):
        """Log completion of AUTO tag resolution."""
        logger.info(
            f"AUTO Tag Resolution - Completed successfully",
            extra={
                "tag_location": resolution.tag_location,
                "total_time_ms": resolution.total_time_ms,
                "retry_count": resolution.retry_count,
                "final_model": resolution.final_model_used,
                "action_type": resolution.action_plan.action_type,
                "models_attempted": resolution.models_attempted,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Add to history for checkpointing
        if self.checkpoint_resolutions:
            self.resolution_history.append(resolution)
    
    def log_resolution_error(
        self, 
        tag: str, 
        location: str, 
        error: Exception,
        models_attempted: List[str]
    ):
        """Log AUTO tag resolution failure."""
        logger.error(
            f"AUTO Tag Resolution - Failed after all attempts",
            extra={
                "tag": tag[:100] + "..." if len(tag) > 100 else tag,
                "location": location,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "models_attempted": models_attempted,
                "timestamp": datetime.now().isoformat()
            },
            exc_info=True
        )
    
    def log_nested_resolution(self, depth: int, parent_tag: str, nested_tag: str):
        """Log nested AUTO tag resolution."""
        logger.debug(
            f"AUTO Tag Resolution - Resolving nested tag",
            extra={
                "depth": depth,
                "parent_tag": parent_tag[:50] + "..." if len(parent_tag) > 50 else parent_tag,
                "nested_tag": nested_tag[:50] + "..." if len(nested_tag) > 50 else nested_tag,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def get_checkpoint_data(self) -> List[Dict[str, Any]]:
        """Get resolution history for checkpointing."""
        return [res.to_checkpoint_data() for res in self.resolution_history]
    
    def clear_history(self):
        """Clear resolution history (e.g., after checkpoint)."""
        self.resolution_history.clear()