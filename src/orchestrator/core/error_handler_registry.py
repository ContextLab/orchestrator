"""
Error handler registry for managing error handlers and execution statistics.
Provides centralized management of error handlers with pattern matching and analytics.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from .error_handling import ErrorHandler, ErrorContext

logger = logging.getLogger(__name__)


class ErrorHandlerRegistry:
    """Registry for managing error handlers and their execution with analytics."""
    
    def __init__(self):
        self.handlers: Dict[str, ErrorHandler] = {}
        self.task_handlers: Dict[str, List[str]] = defaultdict(list)  # task_id -> [handler_ids]
        self.global_handlers: List[str] = []  # handlers that apply to all tasks
        
        # Statistics and analytics
        self.error_statistics: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.handler_statistics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'executions': 0,
            'successes': 0,
            'failures': 0,
            'avg_execution_time': 0.0,
            'last_execution': None,
            'error_types_handled': set()
        })
        self.recovery_patterns: Dict[str, List[str]] = defaultdict(list)
        self.error_frequency: Dict[Tuple[str, str], int] = defaultdict(int)  # (task_id, error_type) -> count
        
        # Handler execution history
        self.execution_history: List[Dict[str, Any]] = []
        self.max_history_size: int = 1000
    
    def register_handler(self, handler_id: str, handler: ErrorHandler, task_id: Optional[str] = None) -> None:
        """Register an error handler for a specific task or globally."""
        if not isinstance(handler, ErrorHandler):
            raise ValueError("handler must be an ErrorHandler instance")
        
        if handler_id in self.handlers:
            logger.warning(f"Overriding existing error handler: {handler_id}")
        
        self.handlers[handler_id] = handler
        
        if task_id:
            # Register for specific task
            if handler_id not in self.task_handlers[task_id]:
                self.task_handlers[task_id].append(handler_id)
                logger.debug(f"Registered error handler '{handler_id}' for task '{task_id}'")
        else:
            # Register globally
            if handler_id not in self.global_handlers:
                self.global_handlers.append(handler_id)
                logger.debug(f"Registered global error handler '{handler_id}'")
        
        # Sort handlers by priority
        if task_id:
            self.task_handlers[task_id].sort(key=lambda h_id: self.handlers[h_id].priority)
        else:
            self.global_handlers.sort(key=lambda h_id: self.handlers[h_id].priority)
    
    def unregister_handler(self, handler_id: str, task_id: Optional[str] = None) -> bool:
        """Unregister an error handler."""
        if handler_id not in self.handlers:
            return False
        
        if task_id:
            if handler_id in self.task_handlers[task_id]:
                self.task_handlers[task_id].remove(handler_id)
                logger.debug(f"Unregistered error handler '{handler_id}' from task '{task_id}'")
        else:
            if handler_id in self.global_handlers:
                self.global_handlers.remove(handler_id)
                logger.debug(f"Unregistered global error handler '{handler_id}'")
        
        # Remove from handlers dict if no longer referenced
        if not self._is_handler_referenced(handler_id):
            del self.handlers[handler_id]
            logger.debug(f"Removed error handler '{handler_id}' from registry")
        
        return True
    
    def get_handler(self, handler_id: str) -> Optional[ErrorHandler]:
        """Get error handler by ID."""
        return self.handlers.get(handler_id)
    
    def find_matching_handlers(self, error: Exception, task_id: str) -> List[Tuple[str, ErrorHandler]]:
        """Find handlers that match the error type and patterns for a specific task."""
        matching_handlers = []
        
        # Check task-specific handlers first
        task_handler_ids = self.task_handlers.get(task_id, [])
        for handler_id in task_handler_ids:
            handler = self.handlers.get(handler_id)
            if handler and handler.matches_error(error, task_id):
                matching_handlers.append((handler_id, handler))
        
        # Check global handlers
        for handler_id in self.global_handlers:
            handler = self.handlers.get(handler_id)
            if handler and handler.matches_error(error, task_id):
                matching_handlers.append((handler_id, handler))
        
        # Sort by priority (lower number = higher priority)
        matching_handlers.sort(key=lambda x: x[1].priority)
        
        logger.debug(f"Found {len(matching_handlers)} matching handlers for error {type(error).__name__} in task {task_id}")
        return matching_handlers
    
    def record_error_occurrence(self, task_id: str, error_type: str, handled: bool = False) -> None:
        """Record error occurrence for statistics."""
        self.error_statistics[task_id]['total_errors'] += 1
        self.error_statistics[task_id][error_type] += 1
        
        if handled:
            self.error_statistics[task_id]['handled_errors'] += 1
            self.error_statistics[task_id][f'{error_type}_handled'] += 1
        else:
            self.error_statistics[task_id]['unhandled_errors'] += 1
        
        # Update frequency tracking
        self.error_frequency[(task_id, error_type)] += 1
        
        logger.debug(f"Recorded error occurrence: task={task_id}, type={error_type}, handled={handled}")
    
    def record_handler_execution(
        self,
        handler_id: str,
        success: bool,
        execution_time: float,
        error_type: str,
        task_id: str
    ) -> None:
        """Record handler execution statistics."""
        stats = self.handler_statistics[handler_id]
        stats['executions'] += 1
        
        if success:
            stats['successes'] += 1
        else:
            stats['failures'] += 1
        
        # Update average execution time
        current_avg = stats['avg_execution_time']
        executions = stats['executions']
        stats['avg_execution_time'] = ((current_avg * (executions - 1)) + execution_time) / executions
        
        stats['last_execution'] = task_id
        stats['error_types_handled'].add(error_type)
        
        # Record in execution history
        self.execution_history.append({
            'handler_id': handler_id,
            'task_id': task_id,
            'error_type': error_type,
            'success': success,
            'execution_time': execution_time,
            'timestamp': None  # Will be set by caller
        })
        
        # Maintain history size limit
        if len(self.execution_history) > self.max_history_size:
            self.execution_history.pop(0)
        
        logger.debug(f"Recorded handler execution: {handler_id}, success={success}, time={execution_time:.3f}s")
    
    def get_recovery_suggestions(self, error_type: str, task_id: str) -> List[str]:
        """Get recovery suggestions based on historical data."""
        suggestions = []
        
        # Get suggestions from recovery patterns
        pattern_key = f"{task_id}:{error_type}"
        if pattern_key in self.recovery_patterns:
            suggestions.extend(self.recovery_patterns[pattern_key])
        
        # Get general suggestions for error type
        general_key = f"*:{error_type}"
        if general_key in self.recovery_patterns:
            suggestions.extend(self.recovery_patterns[general_key])
        
        # Add suggestions based on successful handlers
        for handler_id, stats in self.handler_statistics.items():
            if error_type in stats['error_types_handled'] and stats['successes'] > 0:
                handler = self.handlers.get(handler_id)
                if handler:
                    suggestion = f"Use handler '{handler_id}'"
                    if handler.handler_action:
                        suggestion += f" (action: {handler.handler_action[:50]}...)"
                    suggestions.append(suggestion)
        
        return list(set(suggestions))  # Remove duplicates
    
    def add_recovery_pattern(self, task_id: str, error_type: str, pattern: str) -> None:
        """Add a recovery pattern for future suggestions."""
        pattern_key = f"{task_id}:{error_type}"
        if pattern not in self.recovery_patterns[pattern_key]:
            self.recovery_patterns[pattern_key].append(pattern)
        
        logger.debug(f"Added recovery pattern for {pattern_key}: {pattern}")
    
    def get_error_statistics(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Get error statistics for a task or all tasks."""
        if task_id:
            return dict(self.error_statistics.get(task_id, {}))
        else:
            # Aggregate statistics for all tasks
            total_stats = defaultdict(int)
            for task_stats in self.error_statistics.values():
                for key, value in task_stats.items():
                    total_stats[key] += value
            return dict(total_stats)
    
    def get_handler_statistics(self, handler_id: Optional[str] = None) -> Dict[str, Any]:
        """Get handler execution statistics."""
        if handler_id:
            stats = self.handler_statistics.get(handler_id, {})
            # Convert set to list for serialization
            if 'error_types_handled' in stats:
                stats = stats.copy()
                stats['error_types_handled'] = list(stats['error_types_handled'])
            return stats
        else:
            # Return all handler statistics
            all_stats = {}
            for h_id, stats in self.handler_statistics.items():
                stats_copy = stats.copy()
                if 'error_types_handled' in stats_copy:
                    stats_copy['error_types_handled'] = list(stats_copy['error_types_handled'])
                all_stats[h_id] = stats_copy
            return all_stats
    
    def get_most_common_errors(self, limit: int = 10) -> List[Tuple[Tuple[str, str], int]]:
        """Get most common errors across all tasks."""
        return sorted(self.error_frequency.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def get_handler_success_rate(self, handler_id: str) -> float:
        """Get success rate for a specific handler."""
        stats = self.handler_statistics.get(handler_id, {})
        executions = stats.get('executions', 0)
        if executions == 0:
            return 0.0
        
        successes = stats.get('successes', 0)
        return successes / executions
    
    def cleanup_statistics(self, max_age_days: int = 30) -> None:
        """Clean up old statistics and execution history."""
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Clean up execution history
        self.execution_history = [
            record for record in self.execution_history
            if record.get('timestamp') and record['timestamp'] > cutoff_date
        ]
        
        logger.info(f"Cleaned up error handler statistics older than {max_age_days} days")
    
    def export_statistics(self) -> Dict[str, Any]:
        """Export all statistics for analysis or persistence."""
        return {
            'error_statistics': dict(self.error_statistics),
            'handler_statistics': self.get_handler_statistics(),
            'recovery_patterns': dict(self.recovery_patterns),
            'error_frequency': {f"{k[0]}:{k[1]}": v for k, v in self.error_frequency.items()},
            'execution_history': self.execution_history[-100:],  # Last 100 executions
            'registered_handlers': len(self.handlers),
            'global_handlers_count': len(self.global_handlers),
            'task_specific_handlers_count': sum(len(handlers) for handlers in self.task_handlers.values())
        }
    
    def _is_handler_referenced(self, handler_id: str) -> bool:
        """Check if a handler is referenced by any task or globally."""
        if handler_id in self.global_handlers:
            return True
        
        for task_handlers in self.task_handlers.values():
            if handler_id in task_handlers:
                return True
        
        return False
    
    def clear_all_handlers(self) -> None:
        """Clear all registered handlers (useful for testing)."""
        self.handlers.clear()
        self.task_handlers.clear()
        self.global_handlers.clear()
        logger.info("Cleared all error handlers from registry")
    
    def validate_handlers(self) -> List[str]:
        """Validate all registered handlers and return list of issues."""
        issues = []
        
        for handler_id, handler in self.handlers.items():
            try:
                # Validate handler configuration
                if not handler.enabled:
                    continue
                
                if not handler.handler_task_id and not handler.handler_action and not handler.fallback_value:
                    issues.append(f"Handler '{handler_id}' has no action, task, or fallback configured")
                
                if handler.timeout is not None and handler.timeout <= 0:
                    issues.append(f"Handler '{handler_id}' has invalid timeout: {handler.timeout}")
                
                if handler.max_handler_retries < 0:
                    issues.append(f"Handler '{handler_id}' has invalid retry count: {handler.max_handler_retries}")
                
            except Exception as e:
                issues.append(f"Handler '{handler_id}' validation failed: {e}")
        
        return issues
    
    def __len__(self) -> int:
        """Return number of registered handlers."""
        return len(self.handlers)
    
    def __contains__(self, handler_id: str) -> bool:
        """Check if handler is registered."""
        return handler_id in self.handlers