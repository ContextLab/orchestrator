"""Test input management system for pipeline testing."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class TestInputManager:
    """
    Manages test inputs for different pipeline types and categories.
    
    Provides appropriate test data, file paths, and parameters for
    automated pipeline testing based on pipeline characteristics.
    """
    
    def __init__(self, examples_dir: Optional[Path] = None):
        """
        Initialize test input manager.
        
        Args:
            examples_dir: Directory containing example pipelines and data
        """
        self.examples_dir = examples_dir or Path("examples")
        self.data_dir = self.examples_dir / "data"
        
        # Initialize input templates
        self._category_inputs = self._init_category_inputs()
        self._pipeline_specific_inputs = self._init_pipeline_specific_inputs()
        self._safe_test_data = self._init_safe_test_data()
        
        logger.debug("Initialized TestInputManager")
    
    def get_inputs_for_pipeline(self, 
                               pipeline_name: str, 
                               category: str = "general",
                               complexity: str = "medium") -> Dict[str, Any]:
        """
        Get appropriate test inputs for a specific pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            category: Pipeline category
            complexity: Pipeline complexity level
            
        Returns:
            Dict[str, Any]: Test input parameters
        """
        # Start with category-based inputs
        inputs = self._category_inputs.get(category, {}).copy()
        
        # Add complexity-based adjustments
        complexity_adjustments = self._get_complexity_adjustments(complexity)
        inputs.update(complexity_adjustments)
        
        # Apply pipeline-specific overrides
        specific_inputs = self._pipeline_specific_inputs.get(pipeline_name, {})
        inputs.update(specific_inputs)
        
        # Ensure file paths exist and are valid
        inputs = self._validate_and_fix_file_paths(inputs)
        
        # Add common defaults
        defaults = self._get_common_defaults()
        for key, value in defaults.items():
            if key not in inputs:
                inputs[key] = value
        
        logger.debug(f"Generated inputs for {pipeline_name}: {list(inputs.keys())}")
        return inputs
    
    def get_safe_test_data(self, data_type: str) -> Any:
        """
        Get safe test data for automated testing.
        
        Args:
            data_type: Type of test data needed
            
        Returns:
            Any: Safe test data
        """
        return self._safe_test_data.get(data_type)
    
    def create_test_files(self, temp_dir: Path) -> Dict[str, Path]:
        """
        Create temporary test files in specified directory.
        
        Args:
            temp_dir: Temporary directory for test files
            
        Returns:
            Dict[str, Path]: Mapping of file types to created paths
        """
        created_files = {}
        
        # Create sample CSV file
        csv_path = temp_dir / "test_data.csv"
        csv_content = """name,age,city,score
Alice,25,New York,85.5
Bob,30,San Francisco,92.1
Charlie,35,Chicago,78.3
Diana,28,Boston,89.7
Eve,32,Seattle,91.2"""
        
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(csv_content)
        created_files['csv'] = csv_path
        
        # Create sample JSON file
        json_path = temp_dir / "test_data.json"
        json_data = {
            "users": [
                {"id": 1, "name": "Alice", "active": True},
                {"id": 2, "name": "Bob", "active": False},
                {"id": 3, "name": "Charlie", "active": True}
            ],
            "config": {
                "version": "1.0",
                "features": ["auth", "analytics"]
            }
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        created_files['json'] = json_path
        
        # Create sample text file
        txt_path = temp_dir / "test_document.txt"
        txt_content = """This is a sample document for testing text processing pipelines.
It contains multiple paragraphs with various content types.

The document includes:
- Plain text content
- Multiple sentences and paragraphs
- Various punctuation marks
- Numbers like 123 and 45.67
- Common words for natural language processing

This content is designed to be safe for automated testing while providing
realistic text data for pipeline validation."""
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(txt_content)
        created_files['txt'] = txt_path
        
        # Create sample markdown file
        md_path = temp_dir / "test_article.md"
        md_content = """# Test Article

This is a **test article** for markdown processing pipelines.

## Overview

The article contains:

- Markdown formatting
- Headers and lists
- *Italic* and **bold** text
- Code blocks and links

## Content Section

Here is some content that can be processed by various pipelines:

```python
def hello_world():
    print("Hello, World!")
```

This is useful for testing content transformation and analysis pipelines.

## Conclusion

This test document provides safe, predictable content for automated testing."""
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        created_files['md'] = md_path
        
        logger.debug(f"Created {len(created_files)} test files in {temp_dir}")
        return created_files
    
    def _init_category_inputs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize category-based input templates."""
        return {
            "data_processing": {
                "input_file": "examples/data/sample_data.csv",
                "data_file": "examples/data/sales_data.csv",
                "format": "csv",
                "delimiter": ",",
                "has_header": True
            },
            "research": {
                "topic": "artificial intelligence applications",
                "query": "machine learning in healthcare",
                "search_terms": ["AI", "machine learning", "healthcare"],
                "max_results": 5,
                "language": "en"
            },
            "creative": {
                "description": "a futuristic city with flying cars and neon lights",
                "prompt": "Write a short story about space exploration",
                "style": "modern",
                "tone": "optimistic",
                "length": "short"
            },
            "control_flow": {
                "threshold": 50,
                "items": ["item1", "item2", "item3", "item4", "item5"],
                "condition": True,
                "max_iterations": 10,
                "target_value": 42
            },
            "multimodal": {
                "image_description": "test image for analysis",
                "text_content": "Sample text for multimodal processing",
                "analysis_type": "comprehensive"
            },
            "integration": {
                "service_name": "test_service",
                "api_timeout": 30,
                "retry_attempts": 3,
                "test_mode": True
            },
            "optimization": {
                "code_file": "examples/data/sample_code.py",
                "optimization_level": "basic",
                "preserve_functionality": True,
                "target_metrics": ["performance", "readability"]
            },
            "automation": {
                "task_type": "file_processing",
                "batch_size": 5,
                "parallel_execution": False,
                "dry_run": True
            },
            "validation": {
                "schema_file": "examples/config/validation_schema.json",
                "strict_mode": False,
                "report_format": "json"
            }
        }
    
    def _init_pipeline_specific_inputs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize pipeline-specific input overrides."""
        return {
            # Data processing pipelines
            "simple_data_processing": {
                "input_file": "examples/data/sample_data.csv",
                "output_format": "processed"
            },
            "data_processing_pipeline": {
                "data_file": "examples/data/sales_data.csv",
                "analysis_type": "comprehensive"
            },
            "statistical_analysis": {
                "dataset": "examples/data/test_data.csv",
                "analysis_methods": ["mean", "median", "correlation"]
            },
            
            # Control flow pipelines
            "control_flow_conditional": {
                "threshold": 50,
                "test_value": 75,
                "condition_type": "greater_than"
            },
            "control_flow_for_loop": {
                "items": ["file1.txt", "file2.txt", "file3.txt"],
                "operation": "process"
            },
            "control_flow_while_loop": {
                "target_number": 42,
                "max_attempts": 10,
                "increment": 5
            },
            "control_flow_advanced": {
                "documents": [
                    "artificial intelligence is transforming healthcare",
                    "climate change is affecting global weather patterns",
                    "test text for processing"
                ],
                "languages": ["Spanish", "French", "German"],
                "batch_process": True
            },
            
            # Creative pipelines
            "creative_image_pipeline": {
                "description": "futuristic city with flying cars",
                "styles": ["cyberpunk", "impressionist", "photorealistic"],
                "image_count": 3
            },
            
            # Research pipelines
            "research_minimal": {
                "topic": "quantum computing basics",
                "depth": "overview"
            },
            "research_basic": {
                "research_query": "artificial intelligence in healthcare applications",
                "sources": "academic",
                "max_results": 5
            },
            "research_advanced_tools": {
                "topics": [
                    "artificial intelligence in healthcare",
                    "quantum computing applications",
                    "sustainable agriculture practices",
                    "climate change mitigation strategies"
                ],
                "research_depth": "comprehensive"
            },
            
            # Integration pipelines
            "mcp_integration_pipeline": {
                "search_queries": [
                    "artificial intelligence ethics",
                    "machine learning algorithms",
                    "quantum computing breakthrough"
                ],
                "max_results_per_query": 3
            },
            "mcp_memory_workflow": {
                "users": [
                    {"name": "Alice Johnson", "interests": ["AI", "robotics"]},
                    {"name": "Bob Smith", "interests": ["quantum computing", "physics"]}
                ]
            },
            
            # Optimization pipelines
            "code_optimization": {
                "files": [
                    "examples/data/sample_code.py",
                    "examples/data/sample_javascript.js",
                    "examples/data/sample_java.java"
                ],
                "optimization_focus": "performance"
            },
            
            # Fact checking pipelines
            "fact_checker": {
                "document_to_check": "Climate change is caused by human activities and leads to global warming.",
                "check_level": "basic"
            },
            "iterative_fact_checker": {
                "input_document": "examples/test_data/test_climate_document.md",
                "fact_check_depth": "thorough"
            },
            
            # Model routing pipelines
            "model_routing_demo": {
                "tasks": [
                    {"text": "Hello world", "complexity": "simple"},
                    {"text": "Explain quantum computing", "complexity": "medium"},
                    {"text": "Design a distributed system", "complexity": "complex"}
                ]
            },
            "llm_routing_pipeline": {
                "prompts": [
                    "Hello world",
                    "Write a Python function to sort an array",
                    "Design and implement a complete microservices architecture"
                ]
            }
        }
    
    def _init_safe_test_data(self) -> Dict[str, Any]:
        """Initialize safe test data for various data types."""
        return {
            "text_short": "This is a short test text for processing.",
            "text_medium": """This is a medium-length test document. It contains multiple sentences 
            and can be used for various text processing tasks. The content is designed to be 
            safe and predictable for automated testing purposes.""",
            "text_long": """This is a longer test document that contains multiple paragraphs and 
            various types of content. It is designed for testing text processing pipelines that 
            require more substantial input data.
            
            The document includes different sentence structures, punctuation marks, and common 
            vocabulary that would be found in typical text processing scenarios. This ensures 
            that pipelines can be tested with realistic but controlled input data.
            
            Finally, the content is intentionally generic and safe, avoiding any potentially 
            problematic or sensitive information that might cause issues in automated testing 
            environments.""",
            
            "topics": [
                "artificial intelligence",
                "machine learning applications", 
                "quantum computing basics",
                "sustainable technology",
                "data science methods"
            ],
            
            "search_queries": [
                "AI applications in healthcare",
                "renewable energy technologies",
                "machine learning algorithms",
                "quantum computing breakthrough",
                "climate change solutions"
            ],
            
            "file_items": [
                "document1.txt",
                "report2.pdf", 
                "data3.csv",
                "analysis4.json",
                "summary5.md"
            ],
            
            "categories": [
                "technology",
                "science", 
                "research",
                "analysis",
                "development"
            ],
            
            "test_numbers": [10, 25, 50, 75, 100],
            
            "sample_data_rows": [
                {"id": 1, "name": "Alice", "score": 85.5, "active": True},
                {"id": 2, "name": "Bob", "score": 92.1, "active": False},
                {"id": 3, "name": "Charlie", "score": 78.3, "active": True},
                {"id": 4, "name": "Diana", "score": 89.7, "active": True},
                {"id": 5, "name": "Eve", "score": 91.2, "active": False}
            ]
        }
    
    def _get_complexity_adjustments(self, complexity: str) -> Dict[str, Any]:
        """Get input adjustments based on complexity level."""
        adjustments = {
            "simple": {
                "timeout": 60,
                "max_iterations": 3,
                "batch_size": 2,
                "complexity_level": "basic"
            },
            "medium": {
                "timeout": 180,
                "max_iterations": 5,
                "batch_size": 5,
                "complexity_level": "standard"
            },
            "complex": {
                "timeout": 300,
                "max_iterations": 10,
                "batch_size": 10,
                "complexity_level": "advanced"
            }
        }
        
        return adjustments.get(complexity, adjustments["medium"])
    
    def _get_common_defaults(self) -> Dict[str, Any]:
        """Get common default values for all pipelines."""
        return {
            "model": "anthropic:claude-sonnet-4-20250514",  # Fast, reliable model
            "temperature": 0.7,
            "max_tokens": 2000,
            "test_mode": True,
            "dry_run": False,  # Actually execute for real testing
            "verbose": False,
            "save_intermediate": False,  # Don't clutter with intermediate files
            "timeout": 180,  # 3 minutes default
            "max_cost": 0.50  # $0.50 max cost per pipeline
        }
    
    def _validate_and_fix_file_paths(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fix file paths in inputs, creating alternatives if needed.
        
        Args:
            inputs: Input dictionary potentially containing file paths
            
        Returns:
            Dict[str, Any]: Updated inputs with valid file paths
        """
        # File path keys that need validation
        file_keys = [
            'input_file', 'data_file', 'document_file', 'csv_file',
            'json_file', 'config_file', 'schema_file', 'code_file'
        ]
        
        for key in file_keys:
            if key in inputs:
                file_path = Path(inputs[key])
                
                # Check if file exists
                if not file_path.exists():
                    # Try to find alternative or create fallback
                    alternative = self._find_alternative_file(file_path)
                    if alternative:
                        inputs[key] = str(alternative)
                        logger.debug(f"Replaced missing file {file_path} with {alternative}")
                    else:
                        # Remove the file input if no alternative found
                        logger.warning(f"Removing missing file input: {key} = {file_path}")
                        del inputs[key]
        
        return inputs
    
    def _find_alternative_file(self, original_path: Path) -> Optional[Path]:
        """
        Find an alternative file if the original doesn't exist.
        
        Args:
            original_path: Original file path that doesn't exist
            
        Returns:
            Optional[Path]: Alternative file path or None
        """
        # Check if it's in the data directory
        if original_path.parent.name == "data":
            # Look for any CSV file in the data directory
            if original_path.suffix == ".csv":
                data_dir = self.data_dir
                if data_dir.exists():
                    csv_files = list(data_dir.glob("*.csv"))
                    if csv_files:
                        return csv_files[0]
            
            # Look for any JSON file
            elif original_path.suffix == ".json":
                data_dir = self.data_dir
                if data_dir.exists():
                    json_files = list(data_dir.glob("*.json"))
                    if json_files:
                        return json_files[0]
        
        # Check common alternative names
        alternatives = [
            "sample_data.csv",
            "test_data.csv", 
            "sales_data.csv",
            "sample_code.py",
            "test_image.jpg"
        ]
        
        for alt_name in alternatives:
            alt_path = self.data_dir / alt_name
            if alt_path.exists() and alt_path.suffix == original_path.suffix:
                return alt_path
        
        return None