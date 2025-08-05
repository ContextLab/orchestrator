"""
Enhanced Loop Context Management System with Named Loops.

This module provides comprehensive loop context variables with:
- Named loop system for unlimited nesting depth
- Auto-generated loop names when not specified
- Cross-step loop variable access
- Advanced list item access with $items[index] support
- Template integration with Jinja2
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ItemListAccessor:
    """Advanced list accessor supporting complex operations for loop variables."""
    
    def __init__(self, items: List[Any], loop_name: str = ""):
        self.items = items
        self.loop_name = loop_name
    
    def __getitem__(self, key):
        """Support $items[index] and slice access in templates."""
        if isinstance(key, int):
            return self._safe_get_item(key)
        elif isinstance(key, slice):
            try:
                return self.items[key]
            except (IndexError, TypeError):
                return []
        return None
    
    def _safe_get_item(self, index: int) -> Any:
        """Safe item access with bounds checking."""
        if not self.items:
            return None
        if index < 0:
            index = len(self.items) + index  # Handle negative indices
        if 0 <= index < len(self.items):
            return self.items[index]
        return None
    
    def __len__(self):
        """Return length for len() calls."""
        return len(self.items)
    
    def __iter__(self):
        """Support iteration in templates."""
        return iter(self.items)
    
    def __repr__(self):
        """String representation for debugging."""
        return f"ItemListAccessor({self.items})"
    
    def __str__(self):
        """String conversion for template rendering."""
        return str(self.items)
    
    # Advanced operations for templates
    def get(self, index: int, default=None):
        """Safe get with default: $items.get(2, 'fallback')"""
        result = self._safe_get_item(index)
        return result if result is not None else default
    
    def next_item(self, current_index: int):
        """Get next item: $items.next_item($index)"""
        return self._safe_get_item(current_index + 1)
    
    def prev_item(self, current_index: int):
        """Get previous item: $items.prev_item($index)"""
        return self._safe_get_item(current_index - 1)
    
    def find(self, value) -> int:
        """Find index of value: $items.find('target')"""
        try:
            return self.items.index(value)
        except (ValueError, AttributeError):
            return -1
    
    def contains(self, value) -> bool:
        """Check if contains value: $items.contains('target')"""
        try:
            return value in self.items
        except TypeError:
            return False
    
    def slice(self, start: int, end: Optional[int] = None) -> List[Any]:
        """Get slice of items: $items.slice(1, 3)"""
        try:
            if end is None:
                return self.items[start:]
            return self.items[start:end]
        except (IndexError, TypeError):
            return []
    
    # Properties for common operations
    @property
    def first(self):
        """First item in list: $items.first"""
        return self._safe_get_item(0)
    
    @property 
    def last(self):
        """Last item in list: $items.last"""
        return self._safe_get_item(-1)
    
    @property
    def second(self):
        """Second item in list: $items.second"""
        return self._safe_get_item(1)
    
    @property
    def second_last(self):
        """Second to last item: $items.second_last"""
        return self._safe_get_item(-2)
    
    @property
    def middle(self):
        """Middle item (or closest to middle): $items.middle"""
        if not self.items:
            return None
        mid_index = len(self.items) // 2
        return self._safe_get_item(mid_index)
    
    def is_empty(self) -> bool:
        """Check if list is empty: $items.is_empty()"""
        return len(self.items) == 0
    
    def has_multiple(self) -> bool:
        """Check if list has multiple items: $items.has_multiple()"""
        return len(self.items) > 1


@dataclass
class LoopContextVariables:
    """Named loop context variables with cross-step persistence."""
    
    # Core iteration variables
    item: Any
    index: int
    items: List[Any] 
    length: int
    
    # Loop identification and persistence
    loop_name: str
    loop_id: str
    is_auto_generated: bool
    nesting_depth: int
    
    # Derived variables
    is_first: bool
    is_last: bool
    
    @property
    def position(self) -> int:
        """One-based position ($position = $index + 1)."""
        return self.index + 1
    
    @property 
    def remaining(self) -> int:
        """Items remaining after current ($remaining)."""
        return max(0, self.length - self.position)
    
    @property
    def has_next(self) -> bool:
        """Check if there's a next item ($has_next)."""
        return self.index + 1 < self.length
    
    @property
    def has_prev(self) -> bool:
        """Check if there's a previous item ($has_prev)."""
        return self.index > 0
    
    @classmethod
    def generate_loop_name(cls, step_id: str, nesting_depth: int) -> str:
        """Generate automatic loop name using step ID and depth."""
        # Clean step_id to be safe for variable names
        clean_step_id = step_id.replace("-", "_").replace(".", "_")
        return f"{clean_step_id}_loop_{nesting_depth}"
    
    def to_template_dict(self, is_current_loop: bool = False) -> Dict[str, Any]:
        """Convert to template variables with named and default access.
        
        Args:
            is_current_loop: If True, also provides $ variables for current loop
            
        Returns:
            Dictionary of template variables
        """
        
        # Create ItemListAccessor for advanced list operations
        items_accessor = ItemListAccessor(self.items, self.loop_name)
        
        # Named loop variables: $loop_name.variable
        named_vars = {
            f"${self.loop_name}.item": self.item,
            f"${self.loop_name}.index": self.index,
            f"${self.loop_name}.items": items_accessor,
            f"${self.loop_name}.length": self.length,
            f"${self.loop_name}.is_first": self.is_first,
            f"${self.loop_name}.is_last": self.is_last,
            f"${self.loop_name}.position": self.position,
            f"${self.loop_name}.remaining": self.remaining,
            f"${self.loop_name}.has_next": self.has_next,
            f"${self.loop_name}.has_prev": self.has_prev,
        }
        
        # If this is the current loop, also provide default $ variables
        if is_current_loop:
            default_vars = {
                "$item": self.item,
                "$index": self.index,
                "$items": items_accessor,
                "$length": self.length,
                "$is_first": self.is_first,
                "$is_last": self.is_last,
                "$position": self.position,
                "$remaining": self.remaining,
                "$has_next": self.has_next,
                "$has_prev": self.has_prev,
            }
            named_vars.update(default_vars)
        
        return named_vars
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about this loop context."""
        return {
            "loop_name": self.loop_name,
            "loop_id": self.loop_id,
            "is_auto_generated": self.is_auto_generated,
            "nesting_depth": self.nesting_depth,
            "index": self.index,
            "length": self.length,
            "item_type": type(self.item).__name__,
            "is_current_first": self.is_first,
            "is_current_last": self.is_last,
        }


class GlobalLoopContextManager:
    """Manages loop contexts across entire pipeline execution.
    
    This class handles:
    - Auto-generation of loop names when not explicitly specified
    - Tracking active loops and loop history for cross-step access
    - Providing template variables for all accessible loops
    - Managing nesting depth and loop scope
    """
    
    def __init__(self):
        self.active_loops: Dict[str, LoopContextVariables] = {}
        self.loop_history: Dict[str, LoopContextVariables] = {}
        self.current_loop_stack: List[str] = []
        self.nesting_depth = 0
        self._step_loop_counters: Dict[str, int] = {}  # Track auto-generated names per step
    
    def create_loop_context(
        self, 
        step_id: str, 
        item: Any, 
        index: int, 
        items: List[Any],
        explicit_loop_name: Optional[str] = None
    ) -> LoopContextVariables:
        """Create loop context with auto-generated name if needed.
        
        Args:
            step_id: The step ID containing this loop
            item: Current item in iteration
            index: Zero-based index of current item
            items: Full list of items being iterated
            explicit_loop_name: Optional explicit name for the loop
            
        Returns:
            LoopContextVariables instance
        """
        
        if explicit_loop_name:
            loop_name = explicit_loop_name
            is_auto_generated = False
        else:
            # Generate unique auto name for this step
            if step_id not in self._step_loop_counters:
                self._step_loop_counters[step_id] = 0
            else:
                self._step_loop_counters[step_id] += 1
            
            counter = self._step_loop_counters[step_id]
            loop_name = LoopContextVariables.generate_loop_name(
                step_id, self.nesting_depth + counter
            )
            is_auto_generated = True
        
        # Validate loop name doesn't conflict
        if loop_name in self.active_loops:
            if not is_auto_generated:
                logger.warning(f"Loop name '{loop_name}' is already active, this may cause conflicts")
        
        loop_context = LoopContextVariables(
            item=item,
            index=index,
            items=items,
            length=len(items),
            loop_name=loop_name,
            loop_id=step_id,
            is_auto_generated=is_auto_generated,
            nesting_depth=self.nesting_depth,
            is_first=(index == 0),
            is_last=(index == len(items) - 1)
        )
        
        logger.debug(f"Created loop context: {loop_context.get_debug_info()}")
        return loop_context
    
    def push_loop(self, loop_context: LoopContextVariables):
        """Add new loop context and make it active.
        
        Args:
            loop_context: The loop context to make active
        """
        logger.debug(f"Pushing loop context: {loop_context.loop_name}")
        
        self.active_loops[loop_context.loop_name] = loop_context
        
        # Only add to history if it's explicitly named (can be referenced later)
        if not loop_context.is_auto_generated:
            self.loop_history[loop_context.loop_name] = loop_context
            logger.debug(f"Added loop '{loop_context.loop_name}' to history for cross-step access")
        
        self.current_loop_stack.append(loop_context.loop_name)
        self.nesting_depth += 1
    
    def pop_loop(self, loop_name: str):
        """Remove loop from active contexts.
        
        Args:
            loop_name: Name of the loop to deactivate
        """
        logger.debug(f"Popping loop context: {loop_name}")
        
        if loop_name in self.active_loops:
            loop_context = self.active_loops[loop_name]
            del self.active_loops[loop_name]
            
            # Remove from stack
            if loop_name in self.current_loop_stack:
                self.current_loop_stack.remove(loop_name)
            
            self.nesting_depth = max(0, self.nesting_depth - 1)
            
            logger.debug(f"Removed loop '{loop_name}' from active contexts")
        else:
            logger.warning(f"Attempted to pop non-existent loop: {loop_name}")
    
    def get_current_loop(self) -> Optional[LoopContextVariables]:
        """Get the most recently pushed active loop.
        
        Returns:
            Current loop context or None if no active loops
        """
        if not self.current_loop_stack:
            return None
        
        current_loop_name = self.current_loop_stack[-1]
        return self.active_loops.get(current_loop_name)
    
    def get_loop_by_name(self, loop_name: str) -> Optional[LoopContextVariables]:
        """Get specific loop context by name (active or historical).
        
        Args:
            loop_name: Name of the loop to retrieve
            
        Returns:
            Loop context or None if not found
        """
        # Check active loops first
        if loop_name in self.active_loops:
            return self.active_loops[loop_name]
        
        # Check historical loops
        if loop_name in self.loop_history:
            return self.loop_history[loop_name]
        
        return None
    
    def get_accessible_loop_variables(self) -> Dict[str, Any]:
        """Get template variables for all accessible loops.
        
        Returns:
            Dictionary of all accessible loop variables for templates
        """
        all_vars = {}
        
        # Determine current loop
        current_loop_name = self.current_loop_stack[-1] if self.current_loop_stack else None
        
        # Add variables from all active loops
        for loop_name, loop_context in self.active_loops.items():
            is_current = (loop_name == current_loop_name)
            loop_vars = loop_context.to_template_dict(is_current)
            all_vars.update(loop_vars)
        
        # Add variables from accessible historical loops (explicitly named only)
        for loop_name, loop_context in self.loop_history.items():
            if loop_name not in self.active_loops:
                # Historical loop - only named access, no defaults
                named_vars = loop_context.to_template_dict(is_current=False)
                all_vars.update(named_vars)
        
        logger.debug(f"Providing {len(all_vars)} loop variables to templates")
        return all_vars
    
    def get_active_loop_names(self) -> List[str]:
        """Get list of currently active loop names.
        
        Returns:
            List of active loop names
        """
        return list(self.active_loops.keys())
    
    def get_historical_loop_names(self) -> List[str]:
        """Get list of historical (explicitly named) loop names.
        
        Returns:
            List of historical loop names that can be referenced cross-step
        """
        return list(self.loop_history.keys())
    
    def clear_all_contexts(self):
        """Clear all loop contexts (useful for testing)."""
        logger.debug("Clearing all loop contexts")
        self.active_loops.clear()
        self.loop_history.clear()
        self.current_loop_stack.clear()
        self.nesting_depth = 0
        self._step_loop_counters.clear()
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get comprehensive debug information about loop state.
        
        Returns:
            Debug information dictionary
        """
        return {
            "active_loops": list(self.active_loops.keys()),
            "historical_loops": list(self.loop_history.keys()),
            "current_stack": self.current_loop_stack.copy(),
            "nesting_depth": self.nesting_depth,
            "total_variables": len(self.get_accessible_loop_variables()),
            "step_counters": dict(self._step_loop_counters),
        }