"""Extended loop context system for parallel queue support."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import time
import logging

from .loop_context import LoopContextVariables, GlobalLoopContextManager, ItemListAccessor

logger = logging.getLogger(__name__)


@dataclass
class ParallelQueueContext(LoopContextVariables):
    """Extended loop context for parallel queue execution.
    
    This extends the base LoopContextVariables to support parallel execution patterns
    with cross-task synchronization and shared state management.
    """
    
    # Parallel execution specific
    queue_id: str = ""
    max_parallel: int = 1
    current_parallel_count: int = 0
    
    # Execution tracking
    completed_items: Set[int] = field(default_factory=set)
    failed_items: Set[int] = field(default_factory=set)
    active_items: Set[int] = field(default_factory=set)
    
    # Results aggregation
    item_results: Dict[int, Any] = field(default_factory=dict)
    shared_state: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tracking
    item_start_times: Dict[int, float] = field(default_factory=dict)
    item_end_times: Dict[int, float] = field(default_factory=dict)
    
    # Condition evaluation state
    until_condition: Optional[str] = None
    while_condition: Optional[str] = None
    condition_last_evaluated: Optional[float] = None
    condition_last_result: Optional[bool] = None
    
    def mark_item_started(self, item_index: int) -> None:
        """Mark an item as started."""
        self.active_items.add(item_index)
        self.item_start_times[item_index] = time.time()
        self.current_parallel_count = len(self.active_items)
    
    def mark_item_completed(self, item_index: int, result: Any = None) -> None:
        """Mark an item as completed."""
        self.active_items.discard(item_index)
        self.completed_items.add(item_index)
        self.item_end_times[item_index] = time.time()
        self.current_parallel_count = len(self.active_items)
        
        if result is not None:
            self.item_results[item_index] = result
    
    def mark_item_failed(self, item_index: int, error: Exception = None) -> None:
        """Mark an item as failed."""
        self.active_items.discard(item_index)
        self.failed_items.add(item_index)
        self.item_end_times[item_index] = time.time()
        self.current_parallel_count = len(self.active_items)
        
        if error is not None:
            self.item_results[item_index] = {"error": str(error)}
    
    def get_item_execution_time(self, item_index: int) -> Optional[float]:
        """Get execution time for a specific item."""
        if item_index not in self.item_start_times:
            return None
        
        start_time = self.item_start_times[item_index]
        end_time = self.item_end_times.get(item_index, time.time())
        return end_time - start_time
    
    def get_completion_rate(self) -> float:
        """Get completion rate as percentage."""
        if self.length == 0:
            return 0.0
        return (len(self.completed_items) / self.length) * 100
    
    def get_failure_rate(self) -> float:
        """Get failure rate as percentage."""
        if self.length == 0:
            return 0.0
        return (len(self.failed_items) / self.length) * 100
    
    def is_execution_complete(self) -> bool:
        """Check if all items have been processed."""
        total_processed = len(self.completed_items) + len(self.failed_items)
        return total_processed >= self.length and len(self.active_items) == 0
    
    def get_parallel_progress_summary(self) -> Dict[str, Any]:
        """Get comprehensive parallel execution summary."""
        return {
            "queue_id": self.queue_id,
            "total_items": self.length,
            "completed": len(self.completed_items),
            "failed": len(self.failed_items),
            "active": len(self.active_items),
            "completion_rate": self.get_completion_rate(),
            "failure_rate": self.get_failure_rate(),
            "current_parallel_count": self.current_parallel_count,
            "max_parallel": self.max_parallel,
            "is_complete": self.is_execution_complete(),
            "has_until_condition": self.until_condition is not None,
            "has_while_condition": self.while_condition is not None,
        }
    
    def to_template_dict(self, item_index: Optional[int] = None, is_current_loop: bool = False) -> Dict[str, Any]:
        """
        Convert to template dictionary with parallel queue specific variables.
        
        Args:
            item_index: Index of current item being processed
            is_current_loop: If this is the current active loop context
        """
        base_dict = super().to_template_dict(is_current_loop)
        
        # Add parallel queue specific variables
        parallel_dict = {
            f"${self.loop_name}.queue_id": self.queue_id,
            f"${self.loop_name}.max_parallel": self.max_parallel,
            f"${self.loop_name}.current_parallel": self.current_parallel_count,
            f"${self.loop_name}.completed_count": len(self.completed_items),
            f"${self.loop_name}.failed_count": len(self.failed_items),
            f"${self.loop_name}.active_count": len(self.active_items),
            f"${self.loop_name}.completion_rate": self.get_completion_rate(),
            f"${self.loop_name}.failure_rate": self.get_failure_rate(),
            f"${self.loop_name}.is_complete": self.is_execution_complete(),
            f"${self.loop_name}.results": self.item_results,
            f"${self.loop_name}.shared_state": self.shared_state,
        }
        
        # Add current item specific variables if item_index provided
        if item_index is not None and is_current_loop:
            parallel_dict.update({
                "$queue_index": item_index,
                "$queue_item": self.items[item_index] if item_index < len(self.items) else None,
                "$queue_size": self.length,
                "$is_parallel": True,
                "$execution_time": self.get_item_execution_time(item_index),
            })
        
        base_dict.update(parallel_dict)
        return base_dict
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get comprehensive debug information."""
        base_debug = super().get_debug_info()
        
        parallel_debug = {
            "parallel_execution": {
                "queue_id": self.queue_id,
                "max_parallel": self.max_parallel,
                "current_parallel_count": self.current_parallel_count,
                "completed_items": list(self.completed_items),
                "failed_items": list(self.failed_items),
                "active_items": list(self.active_items),
                "completion_rate": self.get_completion_rate(),
                "failure_rate": self.get_failure_rate(),
                "is_complete": self.is_execution_complete(),
            },
            "conditions": {
                "until_condition": self.until_condition,
                "while_condition": self.while_condition,
                "last_evaluated": self.condition_last_evaluated,
                "last_result": self.condition_last_result,
            },
            "performance": {
                "items_with_timing": len(self.item_start_times),
                "average_item_time": self._calculate_average_item_time(),
                "total_results": len(self.item_results),
            }
        }
        
        base_debug.update(parallel_debug)
        return base_debug
    
    def _calculate_average_item_time(self) -> float:
        """Calculate average item execution time."""
        if not self.item_start_times or not self.item_end_times:
            return 0.0
        
        total_time = 0.0
        count = 0
        
        for item_index in self.item_start_times:
            if item_index in self.item_end_times:
                execution_time = self.item_end_times[item_index] - self.item_start_times[item_index]
                total_time += execution_time
                count += 1
        
        return total_time / count if count > 0 else 0.0


class ParallelLoopContextManager(GlobalLoopContextManager):
    """Extended loop context manager with parallel queue support."""
    
    def __init__(self):
        super().__init__()
        self.parallel_queues: Dict[str, ParallelQueueContext] = {}
    
    def create_parallel_queue_context(self,
                                    queue_id: str,
                                    items: List[Any],
                                    max_parallel: int,
                                    explicit_loop_name: Optional[str] = None,
                                    until_condition: Optional[str] = None,
                                    while_condition: Optional[str] = None) -> ParallelQueueContext:
        """Create a new parallel queue context."""
        
        # Generate or use explicit loop name
        if explicit_loop_name:
            loop_name = explicit_loop_name
        else:
            loop_name = f"parallel_queue_{queue_id}"
        
        # Ensure unique loop name
        if loop_name in self.active_loops:
            counter = 1
            base_name = loop_name
            while f"{base_name}_{counter}" in self.active_loops:
                counter += 1
            loop_name = f"{base_name}_{counter}"
        
        # Create parallel queue context
        context = ParallelQueueContext(
            item=items[0] if items else None,
            index=0,
            items=items,
            length=len(items),
            loop_name=loop_name,
            loop_id=queue_id,
            is_auto_generated=explicit_loop_name is None,
            nesting_depth=0,
            is_first=True,
            is_last=len(items) <= 1,
            queue_id=queue_id,
            max_parallel=max_parallel,
            until_condition=until_condition,
            while_condition=while_condition,
        )
        
        # Add to tracking
        self.parallel_queues[queue_id] = context
        
        logger.debug(f"Created parallel queue context: {loop_name} with {len(items)} items, max_parallel={max_parallel}")
        
        return context
    
    def push_parallel_queue(self, context: ParallelQueueContext) -> None:
        """Push a parallel queue context to active loops."""
        self.active_loops[context.loop_name] = context
        self.loop_history[context.loop_name] = context
        
        logger.debug(f"Pushed parallel queue context: {context.loop_name}")
    
    def get_parallel_queue_context(self, queue_id: str) -> Optional[ParallelQueueContext]:
        """Get parallel queue context by queue ID."""
        return self.parallel_queues.get(queue_id)
    
    def update_parallel_item_context(self, 
                                   queue_id: str, 
                                   item_index: int, 
                                   item: Any) -> Optional[ParallelQueueContext]:
        """Update context for a specific parallel item execution."""
        if queue_id not in self.parallel_queues:
            return None
        
        context = self.parallel_queues[queue_id]
        
        # Update current item context for this execution
        context.item = item
        context.index = item_index
        
        return context
    
    def get_parallel_template_variables(self, 
                                      queue_id: str, 
                                      item_index: int) -> Dict[str, Any]:
        """Get template variables for a specific parallel item."""
        if queue_id not in self.parallel_queues:
            return {}
        
        context = self.parallel_queues[queue_id]
        return context.to_template_dict(item_index=item_index, is_current_loop=True)
    
    def cleanup_parallel_queue(self, queue_id: str) -> None:
        """Clean up parallel queue context."""
        if queue_id in self.parallel_queues:
            context = self.parallel_queues[queue_id]
            
            # Remove from active loops
            if context.loop_name in self.active_loops:
                del self.active_loops[context.loop_name]
            
            # Remove from parallel queues
            del self.parallel_queues[queue_id]
            
            logger.debug(f"Cleaned up parallel queue context: {queue_id}")
    
    def get_all_parallel_contexts(self) -> Dict[str, ParallelQueueContext]:
        """Get all active parallel queue contexts."""
        return self.parallel_queues.copy()
    
    def get_parallel_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all parallel executions."""
        return {
            "total_parallel_queues": len(self.parallel_queues),
            "active_queues": [
                context.get_parallel_progress_summary()
                for context in self.parallel_queues.values()
            ],
            "total_active_items": sum(
                len(context.active_items) 
                for context in self.parallel_queues.values()
            ),
            "total_completed_items": sum(
                len(context.completed_items) 
                for context in self.parallel_queues.values()
            ),
            "total_failed_items": sum(
                len(context.failed_items) 
                for context in self.parallel_queues.values()
            ),
        }