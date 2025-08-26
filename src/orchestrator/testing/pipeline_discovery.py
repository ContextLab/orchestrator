"""Pipeline discovery system for automatic detection and categorization of example pipelines."""

import os
import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineInfo:
    """Information about a discovered pipeline."""
    
    name: str
    path: Path
    category: str
    description: Optional[str] = None
    complexity: str = "medium"  # simple, medium, complex
    estimated_runtime: int = 60  # seconds
    dependencies: List[str] = field(default_factory=list)
    has_loops: bool = False
    has_conditions: bool = False
    has_sub_pipelines: bool = False
    requires_external_apis: bool = False
    model_types: Set[str] = field(default_factory=set)
    input_requirements: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_test_safe(self) -> bool:
        """Check if pipeline is safe for automated testing."""
        # Skip known problematic pipelines
        unsafe_patterns = [
            'interactive',  # Requires user interaction
            'terminal_automation',  # May affect system
            'timeout_test',  # Specifically for timeout testing
        ]
        
        name_lower = self.name.lower()
        return not any(pattern in name_lower for pattern in unsafe_patterns)
    
    @property
    def test_priority(self) -> int:
        """Get test priority (1=highest, 3=lowest)."""
        if self.complexity == "simple":
            return 1
        elif self.complexity == "medium":
            return 2
        else:
            return 3


class PipelineDiscovery:
    """
    Automatic pipeline discovery and categorization system.
    
    Discovers all example pipelines and categorizes them by:
    - Type (data processing, research, creative, etc.)
    - Complexity (simple, medium, complex)
    - Requirements (external APIs, special inputs, etc.)
    - Test suitability (safe for automated testing)
    """
    
    def __init__(self, examples_dir: Optional[Path] = None):
        """
        Initialize pipeline discovery system.
        
        Args:
            examples_dir: Directory containing example pipelines
        """
        self.examples_dir = examples_dir or Path("examples")
        self.discovered_pipelines: Dict[str, PipelineInfo] = {}
        self._category_patterns = self._init_category_patterns()
        self._complexity_indicators = self._init_complexity_indicators()
        
    def discover_all_pipelines(self) -> Dict[str, PipelineInfo]:
        """
        Discover all example pipelines in the examples directory.
        
        Returns:
            Dict[str, PipelineInfo]: Mapping of pipeline names to pipeline info
        """
        logger.info(f"Discovering pipelines in {self.examples_dir}")
        
        if not self.examples_dir.exists():
            logger.warning(f"Examples directory does not exist: {self.examples_dir}")
            return {}
        
        self.discovered_pipelines = {}
        
        # Find all YAML files in examples directory
        for yaml_file in self.examples_dir.glob("*.yaml"):
            if self._should_skip_file(yaml_file):
                continue
                
            try:
                pipeline_info = self._analyze_pipeline(yaml_file)
                if pipeline_info:
                    self.discovered_pipelines[pipeline_info.name] = pipeline_info
                    logger.debug(f"Discovered pipeline: {pipeline_info.name} ({pipeline_info.category})")
            except Exception as e:
                logger.warning(f"Failed to analyze pipeline {yaml_file.name}: {e}")
        
        logger.info(f"Discovered {len(self.discovered_pipelines)} pipelines")
        return self.discovered_pipelines
    
    def get_pipelines_by_category(self, category: str) -> List[PipelineInfo]:
        """
        Get all pipelines in a specific category.
        
        Args:
            category: Pipeline category
            
        Returns:
            List[PipelineInfo]: Pipelines in the category
        """
        return [pipeline for pipeline in self.discovered_pipelines.values() 
                if pipeline.category == category]
    
    def get_test_safe_pipelines(self) -> List[PipelineInfo]:
        """
        Get all pipelines that are safe for automated testing.
        
        Returns:
            List[PipelineInfo]: Test-safe pipelines
        """
        return [pipeline for pipeline in self.discovered_pipelines.values()
                if pipeline.is_test_safe]
    
    def get_pipelines_by_complexity(self, complexity: str) -> List[PipelineInfo]:
        """
        Get all pipelines with specified complexity.
        
        Args:
            complexity: Pipeline complexity (simple, medium, complex)
            
        Returns:
            List[PipelineInfo]: Pipelines with specified complexity
        """
        return [pipeline for pipeline in self.discovered_pipelines.values()
                if pipeline.complexity == complexity]
    
    def get_core_test_pipelines(self) -> List[PipelineInfo]:
        """
        Get core pipelines for essential testing (15-20 pipelines).
        
        Returns:
            List[PipelineInfo]: Core test pipelines
        """
        core_pipelines = []
        
        # Include one pipeline from each major category
        categories_covered = set()
        
        # Sort by priority and select best representatives
        all_safe = sorted(self.get_test_safe_pipelines(), 
                         key=lambda p: (p.test_priority, p.name))
        
        for pipeline in all_safe:
            if pipeline.category not in categories_covered or len(core_pipelines) < 10:
                core_pipelines.append(pipeline)
                categories_covered.add(pipeline.category)
                
            if len(core_pipelines) >= 20:
                break
        
        return core_pipelines
    
    def get_quick_test_pipelines(self) -> List[PipelineInfo]:
        """
        Get quick test pipelines (5-10 fastest, simplest pipelines).
        
        Returns:
            List[PipelineInfo]: Quick test pipelines
        """
        simple_pipelines = self.get_pipelines_by_complexity("simple")
        test_safe = [p for p in simple_pipelines if p.is_test_safe]
        
        # Sort by estimated runtime and select fastest
        quick_pipelines = sorted(test_safe, 
                               key=lambda p: (p.estimated_runtime, p.name))[:10]
        
        # Ensure we have at least 5 pipelines
        if len(quick_pipelines) < 5:
            # Add medium complexity pipelines if needed
            medium_safe = [p for p in self.get_pipelines_by_complexity("medium") 
                          if p.is_test_safe]
            additional = sorted(medium_safe, 
                              key=lambda p: (p.estimated_runtime, p.name))
            quick_pipelines.extend(additional[:5 - len(quick_pipelines)])
        
        return quick_pipelines[:10]
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """
        Check if a file should be skipped during discovery.
        
        Args:
            file_path: Path to the YAML file
            
        Returns:
            bool: True if file should be skipped
        """
        name = file_path.name.lower()
        
        # Skip test files, backups, and disabled files
        skip_patterns = [
            'test_',
            '_test',
            '.bak',
            '_backup',
            '_disabled',
            '.disabled',
        ]
        
        return any(pattern in name for pattern in skip_patterns)
    
    def _analyze_pipeline(self, file_path: Path) -> Optional[PipelineInfo]:
        """
        Analyze a pipeline file and extract information.
        
        Args:
            file_path: Path to the pipeline YAML file
            
        Returns:
            Optional[PipelineInfo]: Pipeline information if analysis succeeds
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse YAML
            pipeline_data = yaml.safe_load(content)
            
            if not isinstance(pipeline_data, dict):
                return None
            
            # Extract basic information
            name = file_path.stem
            description = pipeline_data.get('description', '')
            
            # Determine category
            category = self._determine_category(name, content, pipeline_data)
            
            # Analyze complexity
            complexity = self._determine_complexity(content, pipeline_data)
            
            # Extract dependencies and features
            dependencies = self._extract_dependencies(pipeline_data)
            has_loops = self._has_loops(content)
            has_conditions = self._has_conditions(content)
            has_sub_pipelines = self._has_sub_pipelines(pipeline_data)
            requires_external_apis = self._requires_external_apis(content)
            model_types = self._extract_model_types(content, pipeline_data)
            
            # Estimate runtime
            estimated_runtime = self._estimate_runtime(complexity, has_loops, 
                                                     requires_external_apis)
            
            # Determine input requirements
            input_requirements = self._determine_input_requirements(name, category)
            
            return PipelineInfo(
                name=name,
                path=file_path,
                category=category,
                description=description,
                complexity=complexity,
                estimated_runtime=estimated_runtime,
                dependencies=dependencies,
                has_loops=has_loops,
                has_conditions=has_conditions,
                has_sub_pipelines=has_sub_pipelines,
                requires_external_apis=requires_external_apis,
                model_types=model_types,
                input_requirements=input_requirements
            )
            
        except Exception as e:
            logger.error(f"Error analyzing pipeline {file_path}: {e}")
            return None
    
    def _init_category_patterns(self) -> Dict[str, List[str]]:
        """Initialize category detection patterns."""
        return {
            "data_processing": [
                "data_processing", "statistical", "analysis", "csv", "json",
                "transform", "clean", "filter", "aggregate"
            ],
            "research": [
                "research", "search", "web_research", "fact_check", "investigate",
                "academic", "study", "analysis"
            ],
            "creative": [
                "creative", "image", "generation", "art", "design", "story",
                "writing", "content"
            ],
            "control_flow": [
                "control_flow", "conditional", "loop", "for_loop", "while_loop",
                "if", "then", "else", "iterate", "until"
            ],
            "multimodal": [
                "multimodal", "image", "video", "audio", "vision", "speech"
            ],
            "integration": [
                "mcp", "integration", "api", "service", "external", "tool"
            ],
            "optimization": [
                "optimization", "optimize", "performance", "code", "refactor"
            ],
            "automation": [
                "automation", "terminal", "system", "workflow", "batch"
            ],
            "validation": [
                "validation", "test", "verify", "check", "quality"
            ]
        }
    
    def _init_complexity_indicators(self) -> Dict[str, List[str]]:
        """Initialize complexity detection indicators."""
        return {
            "simple": [
                "simple", "basic", "minimal", "quick", "easy", "demo"
            ],
            "complex": [
                "advanced", "complex", "comprehensive", "enhanced", "sophisticated",
                "for_each", "while", "nested", "parallel", "async"
            ]
        }
    
    def _determine_category(self, name: str, content: str, data: Dict[str, Any]) -> str:
        """
        Determine pipeline category based on name and content.
        
        Args:
            name: Pipeline name
            content: Pipeline YAML content
            data: Parsed pipeline data
            
        Returns:
            str: Pipeline category
        """
        name_lower = name.lower()
        content_lower = content.lower()
        
        # Check category patterns
        for category, patterns in self._category_patterns.items():
            for pattern in patterns:
                if pattern in name_lower or pattern in content_lower:
                    return category
        
        # Default category
        return "general"
    
    def _determine_complexity(self, content: str, data: Dict[str, Any]) -> str:
        """
        Determine pipeline complexity.
        
        Args:
            content: Pipeline YAML content
            data: Parsed pipeline data
            
        Returns:
            str: Complexity level (simple, medium, complex)
        """
        content_lower = content.lower()
        
        # Check for complexity indicators
        for level, indicators in self._complexity_indicators.items():
            for indicator in indicators:
                if indicator in content_lower:
                    return level
        
        # Analyze structure complexity
        tasks = data.get('tasks', [])
        
        if len(tasks) <= 2:
            return "simple"
        elif len(tasks) <= 5:
            return "medium"
        else:
            return "complex"
    
    def _extract_dependencies(self, data: Dict[str, Any]) -> List[str]:
        """
        Extract task dependencies from pipeline data.
        
        Args:
            data: Parsed pipeline data
            
        Returns:
            List[str]: List of dependency names
        """
        dependencies = []
        
        for task in data.get('tasks', []):
            if isinstance(task, dict):
                task_deps = task.get('dependencies', [])
                if isinstance(task_deps, list):
                    dependencies.extend(task_deps)
        
        return list(set(dependencies))
    
    def _has_loops(self, content: str) -> bool:
        """Check if pipeline contains loop constructs."""
        loop_patterns = [
            r'\bfor_each\b',
            r'\bwhile\b',
            r'\buntil\b',
            r'\biterate\b',
            r'\bloop\b'
        ]
        
        content_lower = content.lower()
        return any(re.search(pattern, content_lower) for pattern in loop_patterns)
    
    def _has_conditions(self, content: str) -> bool:
        """Check if pipeline contains conditional logic."""
        condition_patterns = [
            r'\bcondition\s*:',
            r'\bif\b',
            r'\belse\b',
            r'\bwhen\b'
        ]
        
        return any(re.search(pattern, content, re.IGNORECASE) 
                  for pattern in condition_patterns)
    
    def _has_sub_pipelines(self, data: Dict[str, Any]) -> bool:
        """Check if pipeline contains sub-pipeline references."""
        content_str = str(data).lower()
        return 'sub_pipeline' in content_str or 'pipeline:' in content_str
    
    def _requires_external_apis(self, content: str) -> bool:
        """Check if pipeline requires external API access."""
        api_indicators = [
            'web_search', 'search_web', 'google', 'bing',
            'anthropic', 'openai', 'google', 'api_key',
            'external', 'service', 'endpoint'
        ]
        
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in api_indicators)
    
    def _extract_model_types(self, content: str, data: Dict[str, Any]) -> Set[str]:
        """
        Extract model types used in pipeline.
        
        Args:
            content: Pipeline YAML content
            data: Parsed pipeline data
            
        Returns:
            Set[str]: Set of model types
        """
        model_types = set()
        
        # Look for model specifications
        model_patterns = [
            r'anthropic:', r'openai:', r'google:', r'ollama:',
            r'gpt-', r'claude-', r'gemini-', r'llama'
        ]
        
        for pattern in model_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                model_type = pattern.replace(':', '').replace('-', '')
                model_types.add(model_type)
        
        # Check task types
        for task in data.get('tasks', []):
            if isinstance(task, dict):
                task_type = task.get('type', '').lower()
                if task_type in ['llm', 'chat', 'completion']:
                    model_types.add('llm')
                elif task_type in ['image', 'vision']:
                    model_types.add('multimodal')
        
        return model_types
    
    def _estimate_runtime(self, complexity: str, has_loops: bool, 
                         requires_apis: bool) -> int:
        """
        Estimate pipeline runtime in seconds.
        
        Args:
            complexity: Pipeline complexity
            has_loops: Whether pipeline has loops
            requires_apis: Whether pipeline requires external APIs
            
        Returns:
            int: Estimated runtime in seconds
        """
        base_times = {
            "simple": 30,
            "medium": 60,
            "complex": 120
        }
        
        runtime = base_times.get(complexity, 60)
        
        # Adjust for loops
        if has_loops:
            runtime *= 2
        
        # Adjust for external APIs
        if requires_apis:
            runtime *= 1.5
        
        return int(runtime)
    
    def _determine_input_requirements(self, name: str, category: str) -> Dict[str, Any]:
        """
        Determine input requirements for a pipeline.
        
        Args:
            name: Pipeline name
            category: Pipeline category
            
        Returns:
            Dict[str, Any]: Input requirements
        """
        # Default inputs by category
        category_inputs = {
            "data_processing": {
                "input_file": "examples/data/sample_data.csv",
                "data_file": "examples/data/sales_data.csv"
            },
            "research": {
                "topic": "artificial intelligence",
                "query": "machine learning applications",
                "search_terms": ["AI", "ML"]
            },
            "creative": {
                "description": "futuristic city with flying cars",
                "prompt": "creative writing prompt",
                "style": "modern"
            },
            "control_flow": {
                "threshold": 50,
                "items": ["item1", "item2", "item3"],
                "condition": True
            },
            "multimodal": {
                "image_path": "examples/data/test_image.jpg",
                "video_path": "examples/data/test_video.mp4"
            }
        }
        
        # Get base inputs for category
        base_inputs = category_inputs.get(category, {})
        
        # Pipeline-specific overrides
        specific_inputs = self._get_pipeline_specific_inputs(name)
        
        # Merge inputs
        inputs = base_inputs.copy()
        inputs.update(specific_inputs)
        
        return inputs
    
    def _get_pipeline_specific_inputs(self, name: str) -> Dict[str, Any]:
        """Get pipeline-specific input overrides."""
        specific_inputs = {
            "simple_data_processing": {
                "input_file": "examples/data/sample_data.csv"
            },
            "data_processing_pipeline": {
                "data_file": "examples/data/sales_data.csv"
            },
            "control_flow_conditional": {
                "threshold": 50
            },
            "control_flow_for_loop": {
                "items": ["file1.txt", "file2.txt", "file3.txt"]
            },
            "creative_image_pipeline": {
                "description": "futuristic city with flying cars"
            },
            "research_minimal": {
                "topic": "quantum computing"
            },
            "research_basic": {
                "research_query": "artificial intelligence in healthcare"
            }
        }
        
        return specific_inputs.get(name, {})