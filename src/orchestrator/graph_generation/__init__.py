"""
Graph generation package for automatic LangGraph creation from declarative pipelines.

This package implements the automatic graph generation system from Issue #200,
which converts declarative pipeline definitions into optimized LangGraph StateGraph
structures without requiring users to understand graph concepts.

Core Components:
- AutomaticGraphGenerator: Main entry point for graph generation
- DeclarativeSyntaxParser: Parses enhanced YAML syntax from Issue #199
- EnhancedDependencyResolver: Resolves explicit and implicit dependencies
- ParallelExecutionDetector: Identifies parallel execution opportunities
- ControlFlowAnalyzer: Handles conditions, loops, and dynamic routing
- DataFlowValidator: Type safety and variable reference validation
- StateGraphConstructor: Generates optimized LangGraph StateGraph
"""

from .automatic_generator import AutomaticGraphGenerator
from .syntax_parser import DeclarativeSyntaxParser
from .dependency_resolver import EnhancedDependencyResolver
from .parallel_detector import ParallelExecutionDetector
from .control_flow_analyzer import ControlFlowAnalyzer
from .data_flow_validator import DataFlowValidator
from .state_graph_constructor import StateGraphConstructor

__all__ = [
    "AutomaticGraphGenerator",
    "DeclarativeSyntaxParser", 
    "EnhancedDependencyResolver",
    "ParallelExecutionDetector",
    "ControlFlowAnalyzer",
    "DataFlowValidator",
    "StateGraphConstructor",
]