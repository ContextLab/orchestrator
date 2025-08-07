"""Enhanced Pipeline class with automatic graph generation support.

This module extends the core Pipeline class to support StateGraph execution
from the AutomaticGraphGenerator (Issue #200). It maintains full backwards
compatibility while adding optimized graph execution capabilities.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING, Union

from .pipeline import Pipeline
from .task import Task, TaskStatus
from .exceptions import PipelineExecutionError

if TYPE_CHECKING:
    from langgraph.graph import StateGraph

logger = logging.getLogger(__name__)


class EnhancedPipeline(Pipeline):
    """
    Enhanced Pipeline with automatic graph generation and StateGraph execution support.
    
    This class extends the core Pipeline to support:
    - StateGraph execution from AutomaticGraphGenerator
    - Enhanced YAML syntax metadata
    - Optimized parallel execution
    - Self-healing integration with AutoDebugger
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize enhanced pipeline."""
        super().__init__(*args, **kwargs)
        
        # StateGraph integration
        self._state_graph: Optional[StateGraph] = None
        self._graph_execution_enabled = False
        
        # Enhanced execution tracking
        self._execution_history: List[Dict[str, Any]] = []
        self._performance_metrics: Dict[str, Any] = {}
        
        logger.debug(f"EnhancedPipeline created: {self.id}")
        
    @property
    def state_graph(self) -> Optional[StateGraph]:
        """Get attached StateGraph if available."""
        if self._state_graph is None and 'state_graph' in self.metadata:
            self._state_graph = self.metadata['state_graph']
        return self._state_graph
        
    @state_graph.setter  
    def state_graph(self, graph: StateGraph) -> None:
        """Set StateGraph for optimized execution."""
        self._state_graph = graph
        self.metadata['state_graph'] = graph
        self._graph_execution_enabled = True
        logger.info(f"StateGraph attached to pipeline: {self.id}")
        
    def has_state_graph(self) -> bool:
        """Check if pipeline has an attached StateGraph."""
        return self.state_graph is not None
        
    def is_enhanced_format(self) -> bool:
        """Check if pipeline was created from enhanced YAML format."""
        return self.metadata.get('enhanced_syntax', False)
        
    def get_compilation_method(self) -> str:
        """Get the compilation method used to create this pipeline."""
        return self.metadata.get('compilation_method', 'unknown')
        
    async def execute_with_state_graph(self, 
                                     initial_context: Optional[Dict[str, Any]] = None,
                                     stream_results: bool = False) -> Dict[str, Any]:
        """
        Execute pipeline using attached StateGraph for optimal performance.
        
        Args:
            initial_context: Initial context/state for execution
            stream_results: Whether to stream intermediate results
            
        Returns:
            Final execution state/results
            
        Raises:
            PipelineExecutionError: If StateGraph execution fails
        """
        if not self.has_state_graph():
            raise PipelineExecutionError(f"No StateGraph attached to pipeline: {self.id}")
            
        logger.info(f"Executing pipeline with StateGraph: {self.id}")
        start_time = time.time()
        
        try:
            # Prepare initial state
            initial_state = self._prepare_initial_state(initial_context)
            
            # Execute StateGraph
            if stream_results:
                final_state = await self._execute_with_streaming(initial_state)
            else:
                final_state = await self._execute_state_graph_direct(initial_state)
                
            # Record performance metrics
            execution_time = time.time() - start_time
            self._record_execution_metrics(execution_time, True, final_state)
            
            logger.info(f"StateGraph execution completed successfully: {self.id} ({execution_time:.3f}s)")
            return final_state
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_execution_metrics(execution_time, False, {'error': str(e)})
            
            logger.error(f"StateGraph execution failed: {self.id} - {e}")
            raise PipelineExecutionError(f"StateGraph execution failed: {e}") from e
            
    async def _execute_state_graph_direct(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute StateGraph directly without streaming."""
        try:
            # Check if StateGraph is a real LangGraph object or placeholder
            if hasattr(self.state_graph, 'ainvoke'):
                # Real LangGraph StateGraph
                result = await self.state_graph.ainvoke(initial_state)
                return result
            elif isinstance(self.state_graph, dict):
                # Placeholder StateGraph - simulate execution
                logger.info("Executing placeholder StateGraph")
                return await self._execute_placeholder_graph(initial_state)
            else:
                raise PipelineExecutionError(f"Invalid StateGraph type: {type(self.state_graph)}")
                
        except Exception as e:
            logger.error(f"Direct StateGraph execution failed: {e}")
            raise
            
    async def _execute_with_streaming(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute StateGraph with streaming support."""
        try:
            if hasattr(self.state_graph, 'astream'):
                # Real streaming execution
                final_state = None
                async for chunk in self.state_graph.astream(initial_state):
                    # Process streaming chunk
                    logger.debug(f"Streaming chunk: {chunk}")
                    final_state = chunk
                    
                return final_state or initial_state
            else:
                # Fallback to direct execution
                return await self._execute_state_graph_direct(initial_state)
                
        except Exception as e:
            logger.error(f"Streaming StateGraph execution failed: {e}")
            raise
            
    async def _execute_placeholder_graph(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute placeholder StateGraph for testing/development."""
        logger.info(f"Executing placeholder StateGraph: {self.state_graph.get('type')}")
        
        # Simulate execution by processing tasks in order
        results = {}
        current_state = initial_state.copy()
        
        # Process each task
        for task_id, task in self.tasks.items():
            logger.debug(f"Simulating task execution: {task_id}")
            
            # Simulate task execution
            task_result = {
                'task_id': task_id,
                'status': 'completed',
                'result': f'simulated_result_from_{task_id}',
                'execution_time': 0.1  # Simulated execution time
            }
            
            results[task_id] = task_result
            current_state[f'{task_id}_result'] = task_result
            
        return {
            'execution_type': 'placeholder',
            'pipeline_id': self.id,
            'results': results,
            'final_state': current_state
        }
        
    def _prepare_initial_state(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare initial state for StateGraph execution."""
        initial_state = {}
        
        # Add pipeline context
        if self.context:
            initial_state.update(self.context)
            
        # Add execution context
        if context:
            initial_state.update(context)
            
        # Add pipeline metadata
        initial_state['_pipeline_metadata'] = {
            'id': self.id,
            'name': self.name,
            'version': self.version,
            'task_count': len(self.tasks),
            'enhanced_format': self.is_enhanced_format(),
            'compilation_method': self.get_compilation_method()
        }
        
        return initial_state
        
    def _record_execution_metrics(self, 
                                execution_time: float, 
                                success: bool,
                                final_state: Dict[str, Any]) -> None:
        """Record execution performance metrics."""
        metrics = {
            'timestamp': time.time(),
            'execution_time': execution_time,
            'success': success,
            'task_count': len(self.tasks),
            'state_graph_type': type(self.state_graph).__name__ if self.state_graph else None,
            'final_state_size': len(str(final_state)) if final_state else 0
        }
        
        self._performance_metrics = metrics
        self._execution_history.append(metrics)
        
        # Keep only last 10 executions in history
        if len(self._execution_history) > 10:
            self._execution_history = self._execution_history[-10:]
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from last execution."""
        return self._performance_metrics.copy()
        
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self._execution_history.copy()
        
    def get_graph_generation_stats(self) -> Optional[Dict[str, Any]]:
        """Get graph generation statistics if available."""
        return self.metadata.get('graph_generation_stats')
        
    async def execute_hybrid(self, 
                           initial_context: Optional[Dict[str, Any]] = None,
                           prefer_state_graph: bool = True) -> Dict[str, Any]:
        """
        Execute pipeline using hybrid approach - StateGraph if available, otherwise fallback.
        
        Args:
            initial_context: Initial context for execution
            prefer_state_graph: Whether to prefer StateGraph execution
            
        Returns:
            Execution results
        """
        if prefer_state_graph and self.has_state_graph():
            try:
                return await self.execute_with_state_graph(initial_context)
            except Exception as e:
                logger.warning(f"StateGraph execution failed, falling back to legacy: {e}")
                
        # Fallback to legacy task-by-task execution
        return await self._execute_legacy_method(initial_context)
        
    async def _execute_legacy_method(self, 
                                   initial_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute pipeline using legacy task-by-task method."""
        logger.info(f"Executing pipeline using legacy method: {self.id}")
        start_time = time.time()
        
        results = {}
        completed_tasks = set()
        context = initial_context.copy() if initial_context else {}
        
        # Get execution order
        execution_order = self.get_execution_order()
        
        for task_id in execution_order:
            task = self.tasks[task_id]
            
            # Check if task is ready (dependencies satisfied)
            if not task.is_ready(completed_tasks):
                raise PipelineExecutionError(f"Task {task_id} not ready - missing dependencies")
                
            # Simulate task execution (in real implementation, this would execute the actual task)
            logger.debug(f"Executing task (legacy): {task_id}")
            
            task_result = {
                'task_id': task_id,
                'status': 'completed',
                'result': f'legacy_result_from_{task_id}',
                'execution_time': 0.1
            }
            
            results[task_id] = task_result
            completed_tasks.add(task_id)
            
            # Update context with task results
            context[f'{task_id}_result'] = task_result
            
        execution_time = time.time() - start_time
        self._record_execution_metrics(execution_time, True, results)
        
        logger.info(f"Legacy execution completed: {self.id} ({execution_time:.3f}s)")
        
        return {
            'execution_type': 'legacy',
            'pipeline_id': self.id,
            'results': results,
            'context': context,
            'execution_time': execution_time
        }
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary with enhanced metadata."""
        base_dict = super().__dict__.copy()
        
        # Add enhanced metadata
        base_dict.update({
            'has_state_graph': self.has_state_graph(),
            'is_enhanced_format': self.is_enhanced_format(),
            'compilation_method': self.get_compilation_method(),
            'performance_metrics': self.get_performance_metrics(),
            'execution_history_count': len(self._execution_history)
        })
        
        # Don't serialize the actual StateGraph object
        if '_state_graph' in base_dict:
            del base_dict['_state_graph']
            
        return base_dict
        
    def __repr__(self) -> str:
        """String representation with enhanced information."""
        state_graph_info = f", StateGraph={self.has_state_graph()}"
        enhanced_info = f", Enhanced={self.is_enhanced_format()}"
        method_info = f", Method={self.get_compilation_method()}"
        
        return (
            f"EnhancedPipeline(id='{self.id}', name='{self.name}', "
            f"tasks={len(self.tasks)}{state_graph_info}{enhanced_info}{method_info})"
        )


def create_enhanced_pipeline_from_legacy(legacy_pipeline: Pipeline) -> EnhancedPipeline:
    """
    Convert a legacy Pipeline object to EnhancedPipeline.
    
    Args:
        legacy_pipeline: Legacy Pipeline object
        
    Returns:
        EnhancedPipeline object with same data
    """
    enhanced = EnhancedPipeline(
        id=legacy_pipeline.id,
        name=legacy_pipeline.name,
        tasks=legacy_pipeline.tasks,
        context=legacy_pipeline.context,
        metadata=legacy_pipeline.metadata,
        created_at=legacy_pipeline.created_at,
        version=legacy_pipeline.version,
        description=legacy_pipeline.description
    )
    
    logger.info(f"Converted legacy pipeline to enhanced: {legacy_pipeline.id}")
    return enhanced