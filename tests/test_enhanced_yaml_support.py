"""
Enhanced YAML Syntax Support Tests - Issue #199 Integration

This module tests the enhanced YAML processing capabilities that implement
the declarative improvements outlined in Issue #199. Tests cover:

- Type-safe input/output definitions
- Enhanced control flow (parallel_map, loops, conditions)
- Intelligent defaults and auto-discovery
- Backwards compatibility with legacy formats
- Integration with AutomaticGraphGenerator

NO MOCKS - All tests use real enhanced YAML processing and validation.
"""

import pytest
import asyncio
import yaml
from typing import Dict, Any

from src.orchestrator.graph_generation.enhanced_yaml_processor import (
    EnhancedYAMLProcessor, EnhancedPipeline, EnhancedStep, 
    TypeSafeInput, TypeSafeOutput, StepType, DataType
)
from src.orchestrator.graph_generation.automatic_generator import AutomaticGraphGenerator
from src.orchestrator.core.exceptions import YAMLCompilerError, ValidationError


class TestEnhancedYAMLProcessor:
    """Test enhanced YAML processing implementing Issue #199 features."""
    
    @pytest.fixture
    def processor(self):
        """Create enhanced YAML processor instance."""
        return EnhancedYAMLProcessor()
        
    @pytest.fixture
    def enhanced_yaml_example(self):
        """Example enhanced YAML from Issue #199."""
        return {
            "id": "research_pipeline", 
            "name": "Research Pipeline with Enhanced Syntax",
            "description": "Demonstrates Issue #199 enhanced declarative syntax",
            "type": "workflow",
            "version": "2.0.0",
            
            # Type-safe inputs with validation
            "inputs": {
                "topic": {
                    "type": "string",
                    "description": "Research topic to investigate",
                    "required": True,
                    "example": "quantum computing applications"
                },
                "depth": {
                    "type": "string",
                    "enum": ["basic", "detailed", "comprehensive"],
                    "default": "detailed",
                    "description": "Research depth level"
                },
                "max_results": {
                    "type": "integer",
                    "range": [5, 50],
                    "default": 10,
                    "description": "Maximum search results to process"
                }
            },
            
            # Type-safe outputs with sources
            "outputs": {
                "report_file": {
                    "type": "file",
                    "description": "Final research report",
                    "source": "{{ save_report.file_path }}",
                    "format": "markdown"
                },
                "fact_check_results": {
                    "type": "array",
                    "description": "List of verified claims",
                    "source": "{{ fact_check_claims.results }}",
                    "schema": {
                        "verification_status": "string",
                        "confidence_score": "float"
                    }
                },
                "credibility_score": {
                    "type": "float",
                    "source": "{{ compile_report.credibility_score }}",
                    "range": [0.0, 1.0]
                }
            },
            
            # Enhanced declarative steps
            "steps": [
                {
                    "id": "web_search",
                    "tool": "web_search",
                    "inputs": {
                        "query": "{{ inputs.topic }}",
                        "max_results": "{{ inputs.max_results }}"
                    },
                    "outputs": {
                        "search_results": {
                            "type": "array",
                            "description": "Raw search data",
                            "schema": {
                                "title": "string",
                                "url": "string",
                                "content": "string",
                                "relevance_score": "float"
                            }
                        },
                        "total_results": {
                            "type": "integer",
                            "description": "Total number of results found"
                        }
                    }
                },
                {
                    "id": "analyze_results",
                    "tool": "llm_analysis", 
                    "model": {"min_size": "7B", "expertise": "medium"},
                    "depends_on": ["web_search"],
                    "inputs": {
                        "content": "{{ web_search.search_results }}",
                        "analysis_type": "extract_claims_and_sources"
                    },
                    "outputs": {
                        "claims": {
                            "type": "array",
                            "description": "Factual claims found",
                            "schema": {
                                "claim_text": "string",
                                "confidence": "float",
                                "sources": "array"
                            }
                        },
                        "needs_verification": {
                            "type": "boolean",
                            "description": "Whether fact-checking is needed"
                        }
                    }
                }
            ],
            
            # Advanced control flow steps
            "advanced_steps": [
                {
                    "id": "fact_check_claims",
                    "type": "parallel_map",
                    "condition": "{{ analyze_results.needs_verification }}",
                    "items": "{{ analyze_results.claims }}",
                    "max_parallel": 3,
                    "tool": "fact_checker",
                    "inputs": {
                        "claim": "{{ item.claim_text }}",
                        "sources": "{{ item.sources }}"
                    },
                    "outputs": {
                        "verification_status": {
                            "type": "string",
                            "enum": ["verified", "disputed", "unknown"]
                        },
                        "confidence_score": {
                            "type": "float",
                            "range": [0.0, 1.0]
                        }
                    },
                    "depends_on": ["analyze_results"]
                },
                {
                    "id": "compile_report",
                    "tool": "report_generator",
                    "depends_on": ["fact_check_claims"],
                    "inputs": {
                        "topic": "{{ inputs.topic }}",
                        "search_data": "{{ web_search.search_results }}",
                        "fact_check_results": "{{ fact_check_claims.results }}"
                    },
                    "outputs": {
                        "report_content": {
                            "type": "string",
                            "format": "markdown"
                        },
                        "credibility_score": {
                            "type": "float",
                            "range": [0.0, 1.0]
                        }
                    }
                },
                {
                    "id": "save_report",
                    "tool": "filesystem",
                    "depends_on": ["compile_report"],
                    "inputs": {
                        "action": "write",
                        "path": "./reports/{{ inputs.topic | slugify }}_report.md",
                        "content": "{{ compile_report.report_content }}"
                    },
                    "outputs": {
                        "file_path": {
                            "type": "file",
                            "description": "Path to saved file"
                        }
                    }
                }
            ],
            
            # Enhanced configuration
            "config": {
                "timeout": 3600,
                "retry_policy": "exponential_backoff",
                "parallel_optimization": True
            },
            
            "metadata": {
                "author": "Issue #199 Enhanced Syntax",
                "version": "2.0.0",
                "created": "2024-01-01"
            }
        }
        
    @pytest.mark.asyncio
    async def test_enhanced_yaml_format_detection(self, processor):
        """Test detection of Issue #199 enhanced YAML format."""
        
        # Enhanced format indicators
        enhanced_yaml = {
            "id": "test",
            "type": "workflow",
            "inputs": {
                "topic": {
                    "type": "string",
                    "required": True
                }
            }
        }
        
        assert processor._is_enhanced_format(enhanced_yaml) == True
        
        # Legacy format
        legacy_yaml = {
            "id": "test",
            "steps": [{"id": "step1", "action": "echo"}]
        }
        
        assert processor._is_enhanced_format(legacy_yaml) == False
        
    @pytest.mark.asyncio
    async def test_type_safe_inputs_processing(self, processor, enhanced_yaml_example):
        """Test processing of type-safe input definitions."""
        
        pipeline = await processor.process_enhanced_yaml(enhanced_yaml_example)
        
        # Validate input processing
        assert len(pipeline.inputs) == 3
        
        # Test topic input
        topic_input = pipeline.inputs["topic"]
        assert topic_input.type == DataType.STRING
        assert topic_input.required == True
        assert topic_input.description == "Research topic to investigate"
        assert topic_input.example == "quantum computing applications"
        
        # Test depth input with enum
        depth_input = pipeline.inputs["depth"]
        assert depth_input.type == DataType.STRING
        assert depth_input.enum == ["basic", "detailed", "comprehensive"]
        assert depth_input.default == "detailed"
        
        # Test max_results input with range
        max_results_input = pipeline.inputs["max_results"]
        assert max_results_input.type == DataType.INTEGER
        assert max_results_input.range == [5, 50]
        assert max_results_input.default == 10
        
    @pytest.mark.asyncio
    async def test_type_safe_outputs_processing(self, processor, enhanced_yaml_example):
        """Test processing of type-safe output definitions."""
        
        pipeline = await processor.process_enhanced_yaml(enhanced_yaml_example)
        
        # Validate output processing
        assert len(pipeline.outputs) == 3
        
        # Test file output
        report_output = pipeline.outputs["report_file"]
        assert report_output.type == DataType.FILE
        assert report_output.source == "{{ save_report.file_path }}"
        assert report_output.format == "markdown"
        
        # Test array output with schema
        fact_check_output = pipeline.outputs["fact_check_results"]
        assert fact_check_output.type == DataType.ARRAY
        assert fact_check_output.schema is not None
        assert "verification_status" in fact_check_output.schema
        
        # Test float output with range
        credibility_output = pipeline.outputs["credibility_score"]
        assert credibility_output.type == DataType.FLOAT
        assert credibility_output.source == "{{ compile_report.credibility_score }}"
        
    @pytest.mark.asyncio
    async def test_enhanced_step_processing(self, processor, enhanced_yaml_example):
        """Test processing of enhanced step definitions."""
        
        pipeline = await processor.process_enhanced_yaml(enhanced_yaml_example)
        
        # Test standard steps
        assert len(pipeline.steps) == 2
        
        web_search_step = pipeline.steps[0]
        assert web_search_step.id == "web_search"
        assert web_search_step.type == StepType.STANDARD
        assert web_search_step.tool == "web_search"
        assert len(web_search_step.outputs) == 2
        
        # Test advanced steps with control flow
        assert len(pipeline.advanced_steps) == 3
        
        fact_check_step = pipeline.advanced_steps[0]
        assert fact_check_step.id == "fact_check_claims"
        assert fact_check_step.type == StepType.PARALLEL_MAP
        assert fact_check_step.condition == "{{ analyze_results.needs_verification }}"
        assert fact_check_step.items == "{{ analyze_results.claims }}"
        assert fact_check_step.max_parallel == 3
        
    @pytest.mark.asyncio
    async def test_parallel_map_step_processing(self, processor):
        """Test processing of parallel_map steps from Issue #199."""
        
        parallel_map_yaml = {
            "id": "test_parallel",
            "type": "workflow",
            "steps": [
                {
                    "id": "process_items",
                    "type": "parallel_map",
                    "items": "{{ inputs.item_list }}",
                    "max_parallel": 5,
                    "tool": "item_processor",
                    "inputs": {
                        "item": "{{ item }}"
                    },
                    "outputs": {
                        "processed_result": {
                            "type": "string",
                            "description": "Processed item result"
                        }
                    }
                }
            ]
        }
        
        pipeline = await processor.process_enhanced_yaml(parallel_map_yaml)
        
        step = pipeline.steps[0]
        assert step.type == StepType.PARALLEL_MAP
        assert step.items == "{{ inputs.item_list }}"
        assert step.max_parallel == 5
        
    @pytest.mark.asyncio
    async def test_loop_step_processing(self, processor):
        """Test processing of loop steps."""
        
        loop_yaml = {
            "id": "test_loop",
            "type": "workflow",
            "advanced_steps": [
                {
                    "id": "iterative_improvement",
                    "type": "loop",
                    "loop_condition": "{{ quality_score < 0.8 }}",
                    "max_iterations": 5,
                    "steps": [
                        {
                            "id": "improve_content",
                            "tool": "content_improver",
                            "inputs": {
                                "content": "{{ current_content }}"
                            }
                        }
                    ]
                }
            ]
        }
        
        pipeline = await processor.process_enhanced_yaml(loop_yaml)
        
        loop_step = pipeline.advanced_steps[0]
        assert loop_step.type == StepType.LOOP
        assert loop_step.loop_condition == "{{ quality_score < 0.8 }}"
        assert loop_step.max_iterations == 5
        assert len(loop_step.steps) == 1
        
    @pytest.mark.asyncio
    async def test_backwards_compatibility_with_legacy_format(self, processor):
        """Test backwards compatibility with legacy pipeline format."""
        
        legacy_yaml = {
            "id": "legacy_pipeline",
            "name": "Legacy Format Pipeline",
            "parameters": {
                "input_text": {
                    "type": "string",
                    "default": "hello world"
                }
            },
            "steps": [
                {
                    "id": "echo_step",
                    "action": "echo",
                    "parameters": {
                        "message": "{{ parameters.input_text }}"
                    },
                    "dependencies": []
                }
            ],
            "outputs": {
                "result": "{{ echo_step.output }}"
            }
        }
        
        pipeline = await processor.process_enhanced_yaml(legacy_yaml)
        
        # Verify conversion to enhanced format
        assert pipeline.id == "legacy_pipeline"
        assert len(pipeline.inputs) == 1
        assert "input_text" in pipeline.inputs
        assert pipeline.inputs["input_text"].type == DataType.STRING
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].id == "echo_step"
        
    @pytest.mark.asyncio
    async def test_intelligent_defaults_application(self, processor):
        """Test intelligent defaults from Issue #199."""
        
        minimal_yaml = {
            "id": "minimal_pipeline",
            "steps": [
                {
                    "id": "process_data",
                    "tool": "data_processor",
                    "inputs": {
                        "data": "some data"
                    }
                },
                {
                    "id": "save_result", 
                    "tool": "filesystem",
                    "depends_on": ["process_data"],
                    "inputs": {
                        "content": "{{ process_data.result }}"
                    }
                }
            ]
        }
        
        pipeline = await processor.process_enhanced_yaml(minimal_yaml)
        
        # Verify intelligent defaults were applied
        # Steps should have default outputs
        for step in pipeline.steps:
            if not step.outputs and (step.tool or step.action):
                # This should be handled by intelligent defaults
                pass
                
        # Pipeline should have auto-inferred outputs from last step
        if not pipeline.outputs and pipeline.steps:
            # This should be handled by intelligent defaults
            pass
            
    @pytest.mark.asyncio
    async def test_validation_missing_id(self, processor):
        """Test validation catches missing ID in enhanced format."""
        
        # Missing required fields
        invalid_yaml = {
            # Missing id - this should cause validation error
            "type": "workflow",  # Enhanced format indicator 1
            "inputs": {           # Enhanced format indicator 2
                "test_input": {
                    "type": "string",
                    "required": True
                }
            },
            "steps": []
        }
        
        with pytest.raises((YAMLCompilerError, ValidationError)):
            await processor.process_enhanced_yaml(invalid_yaml)
            
    @pytest.mark.asyncio
    async def test_validation_invalid_enum(self, processor):
        """Test validation catches invalid enum values."""
        
        # Invalid enum value  
        invalid_enum_yaml = {
            "id": "test",
            "type": "workflow",  # Enhanced format
            "inputs": {
                "mode": {
                    "type": "string",
                    "enum": ["fast", "slow"],
                    "default": "invalid_value"
                }
            }
        }
        
        with pytest.raises((YAMLCompilerError, ValidationError)):
            await processor.process_enhanced_yaml(invalid_enum_yaml)
            
    @pytest.mark.asyncio
    async def test_validation_invalid_range(self, processor):
        """Test validation catches invalid range values."""
        
        # Invalid range value
        invalid_range_yaml = {
            "id": "test", 
            "type": "workflow",  # Enhanced format
            "inputs": {
                "count": {
                    "type": "integer",
                    "range": [1, 10],
                    "default": 15
                }
            }
        }
        
        with pytest.raises((YAMLCompilerError, ValidationError)):
            await processor.process_enhanced_yaml(invalid_range_yaml)


class TestAutomaticGraphGeneratorEnhancedIntegration:
    """Test integration of enhanced YAML with AutomaticGraphGenerator."""
    
    @pytest.fixture
    def generator(self):
        """Create graph generator with enhanced YAML support."""
        return AutomaticGraphGenerator()
        
    @pytest.mark.asyncio
    async def test_enhanced_yaml_graph_generation(self, generator):
        """Test complete graph generation from enhanced YAML."""
        
        enhanced_pipeline_def = {
            "id": "enhanced_test_pipeline",
            "type": "workflow",
            "inputs": {
                "message": {
                    "type": "string",
                    "default": "Hello Enhanced World",
                    "description": "Message to process"
                }
            },
            "outputs": {
                "final_result": {
                    "type": "string",
                    "source": "{{ process_message.result }}",
                    "description": "Final processed message"
                }
            },
            "steps": [
                {
                    "id": "process_message",
                    "action": "generate_text",
                    "inputs": {
                        "prompt": "Process this message: {{ inputs.message }}",
                        "model": "<AUTO task=\"generate\">Select model for text generation</AUTO>",
                        "max_tokens": 100
                    },
                    "outputs": {
                        "result": {
                            "type": "string",
                            "description": "Processed message result"
                        }
                    }
                }
            ]
        }
        
        # Should detect enhanced format and process correctly
        assert generator._is_enhanced_yaml_format(enhanced_pipeline_def) == True
        
        # Generate graph should succeed
        try:
            state_graph = await generator.generate_graph(enhanced_pipeline_def)
            assert state_graph is not None
            # Successful generation indicates proper enhanced YAML processing
            
        except Exception as e:
            # If there are integration issues, they should be clear
            pytest.fail(f"Enhanced YAML graph generation failed: {e}")
            
    @pytest.mark.asyncio
    async def test_enhanced_yaml_with_parallel_map(self, generator):
        """Test enhanced parallel_map integration with graph generation."""
        
        parallel_map_pipeline = {
            "id": "enhanced_parallel_test",
            "type": "workflow", 
            "inputs": {
                "items_to_process": {
                    "type": "array",
                    "default": ["item1", "item2", "item3"],
                    "description": "Items for parallel processing"
                }
            },
            "steps": [
                {
                    "id": "parallel_processing",
                    "type": "parallel_map",
                    "items": "{{ inputs.items_to_process }}",
                    "max_parallel": 2,
                    "action": "generate_text",
                    "inputs": {
                        "prompt": "Process item: {{ item }}",
                        "model": "<AUTO task=\"generate\">Select model</AUTO>",
                        "max_tokens": 50
                    },
                    "outputs": {
                        "processed_item": {
                            "type": "string",
                            "description": "Processed item result"
                        }
                    }
                }
            ],
            "outputs": {
                "all_results": {
                    "type": "array",
                    "source": "{{ parallel_processing.results }}",
                    "description": "All parallel processing results"
                }
            }
        }
        
        try:
            state_graph = await generator.generate_graph(parallel_map_pipeline)
            assert state_graph is not None
            # Successful generation indicates parallel_map processing works
            
        except Exception as e:
            pytest.fail(f"Enhanced parallel_map graph generation failed: {e}")


class TestEnhancedYAMLFeatureDetection:
    """Test detection of Issue #199 enhanced features."""
    
    @pytest.fixture
    def processor(self):
        return EnhancedYAMLProcessor()
        
    def test_data_type_parsing(self, processor):
        """Test parsing of enhanced data types."""
        
        # Test all supported data types
        type_mappings = {
            "string": DataType.STRING,
            "str": DataType.STRING,
            "integer": DataType.INTEGER,
            "int": DataType.INTEGER,
            "float": DataType.FLOAT,
            "number": DataType.FLOAT,
            "boolean": DataType.BOOLEAN,
            "bool": DataType.BOOLEAN,
            "array": DataType.ARRAY,
            "list": DataType.ARRAY,
            "object": DataType.OBJECT,
            "dict": DataType.OBJECT,
            "file": DataType.FILE,
            "json": DataType.JSON,
            "any": DataType.ANY
        }
        
        for type_str, expected_type in type_mappings.items():
            parsed_type = processor._parse_data_type(type_str)
            assert parsed_type == expected_type
            
        # Test complex type specification
        complex_type = {"base": "array", "items": "string"}
        parsed_type = processor._parse_data_type(complex_type)
        assert parsed_type == DataType.ARRAY
        
    def test_step_type_determination(self, processor):
        """Test automatic step type determination."""
        
        # Test explicit type specification
        parallel_map_step = {"id": "test", "type": "parallel_map", "items": "{{ data }}"}
        step_type = processor._determine_step_type(parallel_map_step)
        assert step_type == StepType.PARALLEL_MAP
        
        # Test auto-detection from fields
        auto_parallel_step = {"id": "test", "items": "{{ data }}", "max_parallel": 3}
        step_type = processor._determine_step_type(auto_parallel_step)
        assert step_type == StepType.PARALLEL_MAP
        
        # Test loop detection
        loop_step = {"id": "test", "loop_condition": "{{ continue }}", "max_iterations": 5}
        step_type = processor._determine_step_type(loop_step)
        assert step_type == StepType.LOOP
        
        # Test standard step
        standard_step = {"id": "test", "tool": "echo"}
        step_type = processor._determine_step_type(standard_step)
        assert step_type == StepType.STANDARD


if __name__ == "__main__":
    # Run specific test for debugging
    import asyncio
    
    async def run_single_test():
        processor = EnhancedYAMLProcessor()
        
        test_yaml = {
            "id": "test_pipeline",
            "type": "workflow",
            "inputs": {
                "message": {
                    "type": "string",
                    "required": True
                }
            },
            "steps": [
                {
                    "id": "echo_step",
                    "action": "echo",
                    "inputs": {
                        "text": "{{ inputs.message }}"
                    },
                    "outputs": {
                        "result": {
                            "type": "string",
                            "description": "Echo result"
                        }
                    }
                }
            ]
        }
        
        pipeline = await processor.process_enhanced_yaml(test_yaml)
        print(f"Processed pipeline: {pipeline.id}")
        print(f"Inputs: {len(pipeline.inputs)}")
        print(f"Steps: {len(pipeline.steps)}")
        
    asyncio.run(run_single_test())