#!/usr/bin/env python3
"""
Tutorial Documentation Generator for Orchestrator Pipelines

This script analyzes validated example pipelines and generates comprehensive
tutorial documentation to enable effective user onboarding and pipeline remixing.

Usage:
    python scripts/tutorial_generator.py [options]
    
Options:
    --output-dir PATH    Output directory for generated tutorials (default: docs/tutorials)
    --pipeline-dir PATH  Directory containing pipeline YAML files (default: examples/)
    --analyze-only       Only analyze pipelines, don't generate tutorials
    --verbose           Enable verbose logging
"""

import os
import sys
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import re
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@dataclass
class PipelineAnalysis:
    """Analysis results for a single pipeline"""
    name: str
    file_path: str
    complexity_level: str  # 'beginner', 'intermediate', 'advanced'
    features_used: List[str]
    template_patterns: List[str]
    control_flow_patterns: List[str]
    data_sources: List[str]
    output_types: List[str]
    dependencies: List[str]
    estimated_runtime: str
    use_cases: List[str]
    key_concepts: List[str]
    difficulty_score: int  # 0-100
    
@dataclass 
class FeatureMatrix:
    """Matrix of features demonstrated across all pipelines"""
    feature_map: Dict[str, List[str]]  # feature -> list of pipelines
    coverage_stats: Dict[str, int]     # feature -> count of demonstrations
    gaps: List[str]                    # features not demonstrated
    
@dataclass
class LearningModule:
    """A learning module containing related pipelines"""
    name: str
    description: str
    pipelines: List[str]
    prerequisites: List[str]
    key_concepts: List[str]
    estimated_time: str

@dataclass
class TutorialSuite:
    """Complete tutorial documentation suite"""
    tutorials: Dict[str, Dict[str, Any]]
    feature_matrix: FeatureMatrix
    learning_path: List[LearningModule]
    metadata: Dict[str, Any]

class PipelineAnalyzer:
    """Analyzes pipeline YAML files to extract features and characteristics"""
    
    # Define comprehensive feature categories
    TOOLBOX_FEATURES = {
        # Core Features
        'template_variables': ['{{', 'input.', 'step.', 'outputs.'],
        'data_flow': ['input:', 'output:', 'outputs:'],
        'error_handling': ['on_error:', 'try_catch:', 'error_handling:'],
        
        # Control Flow
        'conditional_execution': ['if:', 'condition:', 'when:'],
        'for_loops': ['for:', 'iterate:', 'loop:'],
        'while_loops': ['while:', 'while_condition:'],
        'until_conditions': ['until:', 'until_condition:'],
        'dynamic_execution': ['dynamic:', 'runtime_config:'],
        
        # Data Processing
        'csv_processing': ['read-csv', 'write-csv', '.csv'],
        'json_handling': ['read-json', 'write-json', '.json'],
        'data_transformation': ['transform:', 'map:', 'filter:'],
        'statistical_analysis': ['statistics:', 'analyze:', 'stats:'],
        'data_validation': ['validate:', 'check:', 'verify:'],
        
        # Content Generation
        'llm_integration': ['llm-chat', 'generate-text', 'model:'],
        'multi_model_routing': ['route-model', 'model_routing:', 'providers:'],
        'content_synthesis': ['synthesize:', 'combine:', 'merge:'],
        'research_workflows': ['research:', 'search:', 'gather:'],
        'fact_checking': ['fact-check', 'verify-facts', 'validate-claims'],
        
        # Creative & Multimodal
        'image_generation': ['generate-image', 'create-image', 'dall-e'],
        'image_processing': ['process-image', 'analyze-image'],
        'multimodal_content': ['multimodal:', 'vision:', 'audio:'],
        'visual_outputs': ['chart:', 'plot:', 'visualization:'],
        
        # Integration & Tools
        'mcp_integration': ['mcp:', 'mcp-', 'external-tool:'],
        'web_search': ['web-search', 'search-web', 'google-search'],
        'api_integration': ['api-call', 'http-request', 'rest:'],
        'file_operations': ['read-file', 'write-file', 'file-ops'],
        'system_automation': ['terminal:', 'command:', 'execute:'],
        
        # Advanced Patterns
        'iterative_processing': ['iterate:', 'repeat:', 'recursive:'],
        'file_inclusion': ['include:', 'import:', 'extend:'],
        'modular_architecture': ['modules:', 'components:', 'include:'],
        'interactive_workflows': ['interactive:', 'user_input:', 'prompt:'],
        'performance_optimization': ['cache:', 'parallel:', 'optimize:']
    }
    
    COMPLEXITY_INDICATORS = {
        'beginner': {
            'max_steps': 5,
            'features': ['template_variables', 'data_flow', 'llm_integration'],
            'patterns': ['simple_input_output', 'basic_processing']
        },
        'intermediate': {
            'max_steps': 15,
            'features': ['control_flow', 'data_processing', 'multiple_models'],
            'patterns': ['multi_step', 'conditional_logic', 'iteration']
        },
        'advanced': {
            'max_steps': float('inf'),
            'features': ['system_integration', 'complex_control_flow', 'optimization'],
            'patterns': ['recursive', 'modular', 'error_handling', 'security']
        }
    }
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
    def log(self, message: str):
        """Log message if verbose mode enabled"""
        if self.verbose:
            print(f"[PipelineAnalyzer] {message}")
            
    def analyze_pipeline(self, file_path: Path) -> PipelineAnalysis:
        """Analyze a single pipeline YAML file"""
        self.log(f"Analyzing pipeline: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                config = yaml.safe_load(content)
        except Exception as e:
            self.log(f"Error loading {file_path}: {e}")
            return self._create_error_analysis(file_path, str(e))
            
        name = file_path.stem
        
        # Extract features
        features = self._extract_features(content, config)
        templates = self._extract_template_patterns(content)
        control_flow = self._extract_control_flow_patterns(content, config)
        data_sources = self._extract_data_sources(config)
        output_types = self._extract_output_types(config)
        dependencies = self._extract_dependencies(config)
        
        # Assess complexity
        complexity, difficulty = self._assess_complexity(config, features, control_flow)
        
        # Generate use cases and concepts
        use_cases = self._generate_use_cases(name, features, complexity)
        concepts = self._generate_key_concepts(features, control_flow)
        
        return PipelineAnalysis(
            name=name,
            file_path=str(file_path),
            complexity_level=complexity,
            features_used=features,
            template_patterns=templates,
            control_flow_patterns=control_flow,
            data_sources=data_sources,
            output_types=output_types,
            dependencies=dependencies,
            estimated_runtime=self._estimate_runtime(config, complexity),
            use_cases=use_cases,
            key_concepts=concepts,
            difficulty_score=difficulty
        )
        
    def _extract_features(self, content: str, config: Dict) -> List[str]:
        """Extract toolbox features used in the pipeline"""
        features = []
        content_lower = content.lower()
        
        for feature, indicators in self.TOOLBOX_FEATURES.items():
            for indicator in indicators:
                if indicator.lower() in content_lower:
                    features.append(feature)
                    break
                    
        return sorted(list(set(features)))
        
    def _extract_template_patterns(self, content: str) -> List[str]:
        """Extract template variable patterns used"""
        patterns = []
        
        # Find template variables
        template_vars = re.findall(r'\{\{([^}]+)\}\}', content)
        for var in template_vars:
            var_clean = var.strip()
            if '.' in var_clean:
                pattern_type = var_clean.split('.')[0]
                patterns.append(f"template_access_{pattern_type}")
            else:
                patterns.append("simple_variable")
                
        return sorted(list(set(patterns)))
        
    def _extract_control_flow_patterns(self, content: str, config: Dict) -> List[str]:
        """Extract control flow patterns used"""
        patterns = []
        
        if 'steps' in config:
            steps = config['steps']
            if isinstance(steps, dict):
                for step_name, step_config in steps.items():
                    if isinstance(step_config, dict):
                        # Check for control flow keywords
                        if 'if' in step_config or 'condition' in step_config:
                            patterns.append('conditional')
                        if 'for' in step_config or 'iterate' in step_config:
                            patterns.append('iteration')
                        if 'while' in step_config:
                            patterns.append('while_loop')
                        if 'until' in step_config:
                            patterns.append('until_condition')
                            
        return sorted(list(set(patterns)))
        
    def _extract_data_sources(self, config: Dict) -> List[str]:
        """Extract data sources used by the pipeline"""
        sources = []
        
        # Check input section
        if 'input' in config:
            input_config = config['input']
            if isinstance(input_config, dict):
                for key, value in input_config.items():
                    if isinstance(value, str):
                        if '.csv' in value.lower():
                            sources.append('CSV files')
                        elif '.json' in value.lower():
                            sources.append('JSON files')
                        elif '.txt' in value.lower():
                            sources.append('Text files')
                        elif 'http' in value.lower():
                            sources.append('Web APIs')
                            
        return sorted(list(set(sources)))
        
    def _extract_output_types(self, config: Dict) -> List[str]:
        """Extract output types produced by the pipeline"""
        outputs = []
        
        # Check for common output patterns
        config_str = str(config).lower()
        if 'csv' in config_str:
            outputs.append('CSV data')
        if 'json' in config_str:
            outputs.append('JSON data')
        if 'markdown' in config_str or '.md' in config_str:
            outputs.append('Markdown documents')
        if 'image' in config_str or 'png' in config_str:
            outputs.append('Images')
        if 'report' in config_str:
            outputs.append('Reports')
        if 'analysis' in config_str:
            outputs.append('Analysis results')
            
        return sorted(list(set(outputs))) if outputs else ['Text output']
        
    def _extract_dependencies(self, config: Dict) -> List[str]:
        """Extract external dependencies and requirements"""
        deps = []
        
        config_str = str(config).lower()
        
        # External services
        if 'openai' in config_str or 'gpt' in config_str:
            deps.append('OpenAI API')
        if 'anthropic' in config_str or 'claude' in config_str:
            deps.append('Anthropic API')
        if 'google' in config_str:
            deps.append('Google APIs')
        if 'web-search' in config_str:
            deps.append('Web search APIs')
        if 'mcp' in config_str:
            deps.append('MCP tools')
        if 'terminal' in config_str:
            deps.append('System access')
            
        return sorted(list(set(deps)))
        
    def _assess_complexity(self, config: Dict, features: List[str], control_flow: List[str]) -> Tuple[str, int]:
        """Assess pipeline complexity level and difficulty score"""
        score = 0
        
        # Count steps
        step_count = 0
        if 'steps' in config and isinstance(config['steps'], dict):
            step_count = len(config['steps'])
        score += min(step_count * 5, 30)  # Max 30 points for steps
        
        # Feature complexity
        advanced_features = [
            'iterative_processing', 'system_automation', 'error_handling',
            'modular_architecture', 'performance_optimization'
        ]
        intermediate_features = [
            'control_flow', 'data_processing', 'multi_model_routing',
            'mcp_integration', 'api_integration'
        ]
        
        for feature in features:
            if feature in advanced_features:
                score += 15
            elif feature in intermediate_features:
                score += 10
            else:
                score += 5
                
        # Control flow complexity
        score += len(control_flow) * 8
        
        # Template complexity
        config_str = str(config)
        template_count = config_str.count('{{')
        score += min(template_count * 2, 20)
        
        # Determine complexity level
        if score <= 25:
            return 'beginner', score
        elif score <= 60:
            return 'intermediate', score
        else:
            return 'advanced', score
            
    def _generate_use_cases(self, name: str, features: List[str], complexity: str) -> List[str]:
        """Generate realistic use cases for the pipeline"""
        use_cases = []
        
        # Name-based use cases
        if 'research' in name:
            use_cases.extend([
                'Academic research and literature review',
                'Market research and competitive analysis',
                'Fact-checking and information verification'
            ])
        if 'data' in name:
            use_cases.extend([
                'Business data analysis and reporting',
                'Data quality assessment and cleaning',
                'Automated data processing workflows'
            ])
        if 'creative' in name:
            use_cases.extend([
                'Content creation and marketing materials',
                'Creative writing and ideation',
                'Visual content generation'
            ])
        if 'control_flow' in name:
            use_cases.extend([
                'Conditional workflow automation',
                'Batch processing with logic',
                'Dynamic pipeline execution'
            ])
            
        # Feature-based use cases
        if 'web_search' in features:
            use_cases.append('Information gathering and research')
        if 'image_generation' in features:
            use_cases.append('Visual content creation')
        if 'llm_integration' in features:
            use_cases.append('AI-powered content generation')
        if 'system_automation' in features:
            use_cases.append('System administration and automation')
            
        return sorted(list(set(use_cases))) if use_cases else ['General automation tasks']
        
    def _generate_key_concepts(self, features: List[str], control_flow: List[str]) -> List[str]:
        """Generate key concepts demonstrated by the pipeline"""
        concepts = []
        
        # Feature-based concepts
        concept_map = {
            'template_variables': 'Template variable substitution',
            'data_flow': 'Data flow between pipeline steps',
            'conditional_execution': 'Conditional logic and branching',
            'for_loops': 'Iterative processing with loops',
            'llm_integration': 'Large language model integration',
            'error_handling': 'Error handling and recovery',
            'mcp_integration': 'External tool integration',
            'file_operations': 'File system operations',
            'api_integration': 'REST API communication'
        }
        
        for feature in features:
            if feature in concept_map:
                concepts.append(concept_map[feature])
                
        # Control flow concepts
        if 'conditional' in control_flow:
            concepts.append('Conditional execution patterns')
        if 'iteration' in control_flow:
            concepts.append('Loop-based processing')
        if 'while_loop' in control_flow:
            concepts.append('While loop conditions')
        if 'until_condition' in control_flow:
            concepts.append('Until condition patterns')
            
        return sorted(list(set(concepts))) if concepts else ['Basic pipeline structure']
        
    def _estimate_runtime(self, config: Dict, complexity: str) -> str:
        """Estimate pipeline runtime"""
        base_times = {
            'beginner': '< 5 minutes',
            'intermediate': '5-15 minutes', 
            'advanced': '15+ minutes'
        }
        
        # Adjust for specific patterns
        config_str = str(config).lower()
        if 'research' in config_str or 'search' in config_str:
            if complexity == 'beginner':
                return '2-5 minutes'
            elif complexity == 'intermediate':
                return '10-30 minutes'
            else:
                return '30+ minutes'
                
        return base_times[complexity]
        
    def _create_error_analysis(self, file_path: Path, error: str) -> PipelineAnalysis:
        """Create analysis for a pipeline that couldn't be loaded"""
        return PipelineAnalysis(
            name=file_path.stem,
            file_path=str(file_path),
            complexity_level='unknown',
            features_used=[],
            template_patterns=[],
            control_flow_patterns=[],
            data_sources=[],
            output_types=[],
            dependencies=[],
            estimated_runtime='unknown',
            use_cases=[f'Pipeline analysis failed: {error}'],
            key_concepts=[],
            difficulty_score=0
        )

class TutorialGenerator:
    """Generates tutorial documentation from pipeline analysis"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.analyzer = PipelineAnalyzer(verbose)
        
    def log(self, message: str):
        """Log message if verbose mode enabled"""
        if self.verbose:
            print(f"[TutorialGenerator] {message}")
            
    def generate_tutorial_suite(self, pipeline_dir: Path, output_dir: Path) -> TutorialSuite:
        """Generate complete tutorial documentation suite"""
        self.log("Generating comprehensive tutorial suite...")
        
        # Analyze all pipelines
        analyses = self._analyze_all_pipelines(pipeline_dir)
        
        # Generate individual tutorials
        tutorials = self._generate_tutorials(analyses)
        
        # Create feature coverage matrix
        feature_matrix = self._create_feature_matrix(analyses)
        
        # Build progressive learning path
        learning_path = self._create_learning_path(analyses)
        
        # Create tutorial suite
        suite = TutorialSuite(
            tutorials=tutorials,
            feature_matrix=feature_matrix,
            learning_path=learning_path,
            metadata={
                'generated_at': datetime.now().isoformat(),
                'pipeline_count': len(analyses),
                'feature_count': len(feature_matrix.feature_map),
                'learning_modules': len(learning_path)
            }
        )
        
        # Write tutorial files
        self._write_tutorial_suite(suite, output_dir)
        
        return suite
        
    def _analyze_all_pipelines(self, pipeline_dir: Path) -> List[PipelineAnalysis]:
        """Analyze all pipeline YAML files in the directory"""
        self.log(f"Analyzing pipelines in {pipeline_dir}")
        
        analyses = []
        yaml_files = list(pipeline_dir.glob('*.yaml')) + list(pipeline_dir.glob('*.yml'))
        
        for yaml_file in sorted(yaml_files):
            analysis = self.analyzer.analyze_pipeline(yaml_file)
            analyses.append(analysis)
            
        self.log(f"Analyzed {len(analyses)} pipelines")
        return analyses
        
    def _generate_tutorials(self, analyses: List[PipelineAnalysis]) -> Dict[str, Dict[str, Any]]:
        """Generate tutorial content for all analyzed pipelines"""
        tutorials = {}
        
        for analysis in analyses:
            self.log(f"Generating tutorial for {analysis.name}")
            tutorials[analysis.name] = self._generate_single_tutorial(analysis)
            
        return tutorials
        
    def _generate_single_tutorial(self, analysis: PipelineAnalysis) -> Dict[str, Any]:
        """Generate tutorial content for a single pipeline"""
        
        # Load the actual pipeline for code examples
        try:
            with open(analysis.file_path, 'r', encoding='utf-8') as f:
                pipeline_yaml = f.read()
        except:
            pipeline_yaml = "# Pipeline file could not be loaded"
            
        tutorial = {
            'metadata': {
                'pipeline_name': analysis.name,
                'complexity_level': analysis.complexity_level,
                'difficulty_score': analysis.difficulty_score,
                'estimated_runtime': analysis.estimated_runtime,
                'generated_at': datetime.now().isoformat()
            },
            'overview': {
                'purpose': self._generate_purpose(analysis),
                'use_cases': analysis.use_cases,
                'prerequisites': self._generate_prerequisites(analysis),
                'key_concepts': analysis.key_concepts
            },
            'pipeline_breakdown': {
                'yaml_content': pipeline_yaml,
                'configuration_analysis': self._analyze_configuration(analysis),
                'data_flow': self._describe_data_flow(analysis),
                'control_flow': self._describe_control_flow(analysis)
            },
            'customization_guide': {
                'input_modifications': self._generate_input_guide(analysis),
                'parameter_tuning': self._generate_parameter_guide(analysis),
                'step_modifications': self._generate_step_guide(analysis),
                'output_customization': self._generate_output_guide(analysis)
            },
            'remixing_instructions': {
                'compatible_patterns': self._find_compatible_patterns(analysis),
                'extension_ideas': self._generate_extension_ideas(analysis),
                'combination_examples': self._generate_combination_examples(analysis),
                'advanced_variations': self._generate_advanced_variations(analysis)
            },
            'hands_on_exercise': {
                'execution_instructions': self._generate_execution_instructions(analysis),
                'expected_outputs': self._generate_expected_outputs(analysis),
                'troubleshooting': self._generate_troubleshooting(analysis),
                'verification_steps': self._generate_verification_steps(analysis)
            }
        }
        
        return tutorial
        
    def _generate_purpose(self, analysis: PipelineAnalysis) -> str:
        """Generate purpose description for the pipeline"""
        if 'research' in analysis.name:
            return f"This pipeline demonstrates how to build automated research workflows using the orchestrator toolbox. It showcases {', '.join(analysis.features_used[:3])} and provides a foundation for building more sophisticated research applications."
        elif 'data' in analysis.name:
            return f"This pipeline shows how to process and analyze data using orchestrator's data processing capabilities. It demonstrates {', '.join(analysis.features_used[:3])} for building robust data workflows."
        elif 'control_flow' in analysis.name:
            return f"This pipeline illustrates advanced control flow patterns in orchestrator. It demonstrates {', '.join(analysis.control_flow_patterns)} for building dynamic, conditional workflows."
        elif 'creative' in analysis.name:
            return f"This pipeline showcases creative content generation capabilities. It demonstrates {', '.join(analysis.features_used[:3])} for building AI-powered creative workflows."
        else:
            return f"This pipeline demonstrates {', '.join(analysis.features_used[:3])} and provides a practical example of orchestrator's capabilities for {analysis.complexity_level}-level workflows."
            
    def _generate_prerequisites(self, analysis: PipelineAnalysis) -> List[str]:
        """Generate prerequisites based on complexity and features"""
        prereqs = ["Basic understanding of YAML syntax"]
        
        if analysis.complexity_level == 'intermediate':
            prereqs.extend([
                "Familiarity with template variables and data flow",
                "Understanding of basic control flow concepts"
            ])
        elif analysis.complexity_level == 'advanced':
            prereqs.extend([
                "Experience with intermediate pipeline patterns",
                "Understanding of error handling and system integration",
                "Familiarity with external APIs and tools"
            ])
            
        # Feature-specific prerequisites
        if 'system_automation' in analysis.features_used:
            prereqs.append("Understanding of command-line interfaces and system security")
        if 'mcp_integration' in analysis.features_used:
            prereqs.append("Familiarity with Model Context Protocol (MCP)")
        if 'api_integration' in analysis.features_used:
            prereqs.append("Basic understanding of REST APIs")
            
        return prereqs
        
    def _analyze_configuration(self, analysis: PipelineAnalysis) -> Dict[str, str]:
        """Analyze and explain configuration sections"""
        return {
            'input_section': 'Defines the data inputs and parameters for the pipeline',
            'steps_section': 'Contains the sequence of operations to be executed',
            'output_section': 'Specifies how results are formatted and stored',
            'template_usage': f"Uses {len(analysis.template_patterns)} template patterns for dynamic content",
            'feature_highlights': f"Demonstrates {len(analysis.features_used)} key orchestrator features"
        }
        
    def _describe_data_flow(self, analysis: PipelineAnalysis) -> str:
        """Describe how data flows through the pipeline"""
        if analysis.data_sources:
            sources = ', '.join(analysis.data_sources)
            outputs = ', '.join(analysis.output_types)
            return f"Data flows from {sources} through {len(analysis.features_used)} processing steps to produce {outputs}."
        else:
            return f"This pipeline processes input parameters through {len(analysis.features_used)} steps to generate the specified outputs."
            
    def _describe_control_flow(self, analysis: PipelineAnalysis) -> str:
        """Describe control flow patterns used"""
        if analysis.control_flow_patterns:
            patterns = ', '.join(analysis.control_flow_patterns)
            return f"Uses {patterns} patterns to control execution flow based on conditions and data."
        else:
            return "Follows linear execution flow from first step to last step."
            
    def _generate_input_guide(self, analysis: PipelineAnalysis) -> List[str]:
        """Generate input modification guide"""
        guides = [
            "Modify input parameters to match your specific data sources",
            "Adjust file paths and data formats as needed for your environment"
        ]
        
        if analysis.data_sources:
            guides.append(f"This pipeline works with {', '.join(analysis.data_sources)} - adapt the input section accordingly")
            
        return guides
        
    def _generate_parameter_guide(self, analysis: PipelineAnalysis) -> List[str]:
        """Generate parameter tuning guide"""
        guides = []
        
        if 'llm_integration' in analysis.features_used:
            guides.extend([
                "Adjust model parameters (temperature, max_tokens) for different output styles",
                "Modify prompts to change the tone and focus of generated content"
            ])
            
        if 'control_flow' in str(analysis.control_flow_patterns):
            guides.append("Tune conditional parameters to change when branches execute")
            
        if analysis.complexity_level == 'advanced':
            guides.append("Fine-tune performance parameters for your specific use case")
            
        return guides if guides else ["Adjust step parameters to customize behavior for your needs"]
        
    def _generate_step_guide(self, analysis: PipelineAnalysis) -> List[str]:
        """Generate step modification guide"""
        return [
            "Add new steps by following the same pattern as existing ones",
            "Remove steps that aren't needed for your specific use case",
            "Reorder steps if your workflow requires different sequencing",
            "Replace tool actions with alternatives that provide similar functionality"
        ]
        
    def _generate_output_guide(self, analysis: PipelineAnalysis) -> List[str]:
        """Generate output customization guide"""
        guides = [
            "Change output file paths and formats to match your requirements",
            "Modify output templates to customize the structure and content"
        ]
        
        if analysis.output_types:
            types = ', '.join(analysis.output_types)
            guides.append(f"This pipeline produces {types} - adjust output configuration accordingly")
            
        return guides
        
    def _find_compatible_patterns(self, analysis: PipelineAnalysis) -> List[str]:
        """Find patterns that work well with this pipeline"""
        compatible = []
        
        # Feature-based compatibility
        if 'data_processing' in analysis.features_used:
            compatible.extend([
                "statistical_analysis.yaml - for analyzing processed data",
                "data_validation pipelines - for quality assurance"
            ])
            
        if 'llm_integration' in analysis.features_used:
            compatible.extend([
                "fact_checker.yaml - for content verification",
                "research workflows - for information gathering"
            ])
            
        if 'control_flow' in str(analysis.control_flow_patterns):
            compatible.append("error_handling patterns - for robust execution")
            
        return compatible if compatible else ["Most basic pipelines can be combined with this pattern"]
        
    def _generate_extension_ideas(self, analysis: PipelineAnalysis) -> List[str]:
        """Generate ideas for extending this pipeline"""
        ideas = []
        
        if analysis.complexity_level == 'beginner':
            ideas.extend([
                "Add error handling and recovery steps",
                "Implement conditional logic for different scenarios",
                "Include data validation and quality checks"
            ])
        elif analysis.complexity_level == 'intermediate':
            ideas.extend([
                "Add iterative processing for continuous improvement",
                "Implement parallel processing for better performance",
                "Include advanced error recovery mechanisms"
            ])
        else:
            ideas.extend([
                "Build modular components for reusability",
                "Add performance monitoring and optimization",
                "Implement advanced security and access controls"
            ])
            
        return ideas
        
    def _generate_combination_examples(self, analysis: PipelineAnalysis) -> List[str]:
        """Generate specific combination examples"""
        examples = []
        
        if 'research' in analysis.name:
            examples.extend([
                "Combine with fact_checker.yaml to verify research claims",
                "Use with creative_image_pipeline.yaml to generate visual research summaries",
                "Integrate with data_processing.yaml to analyze research data"
            ])
        elif 'data' in analysis.name:
            examples.extend([
                "Combine with research workflows to gather additional data",
                "Use with statistical analysis for comprehensive insights",
                "Integrate with visualization tools for data presentation"
            ])
            
        return examples if examples else ["Can be combined with most other pipeline patterns"]
        
    def _generate_advanced_variations(self, analysis: PipelineAnalysis) -> List[str]:
        """Generate advanced variation ideas"""
        variations = [
            f"Scale to handle larger datasets and more complex processing",
            f"Add real-time processing capabilities for streaming data",
            f"Implement distributed processing across multiple systems"
        ]
        
        if 'llm_integration' in analysis.features_used:
            variations.append("Use multiple AI models for comparison and validation")
            
        if 'api_integration' in analysis.features_used:
            variations.append("Add rate limiting and retry logic for production use")
            
        return variations
        
    def _generate_execution_instructions(self, analysis: PipelineAnalysis) -> List[str]:
        """Generate step-by-step execution instructions"""
        instructions = [
            f"1. Navigate to your orchestrator project directory",
            f"2. Run: python scripts/run_pipeline.py examples/{analysis.name}.yaml",
            f"3. Monitor the output for progress and any error messages",
            f"4. Check the output directory for generated results"
        ]
        
        if analysis.dependencies:
            deps = ', '.join(analysis.dependencies)
            instructions.insert(1, f"1.5. Ensure you have access to required services: {deps}")
            
        return instructions
        
    def _generate_expected_outputs(self, analysis: PipelineAnalysis) -> List[str]:
        """Generate description of expected outputs"""
        outputs = []
        
        if analysis.output_types:
            for output_type in analysis.output_types:
                outputs.append(f"Generated {output_type} in the specified output directory")
        else:
            outputs.append("Text-based results printed to console")
            
        outputs.extend([
            "Execution logs showing step-by-step progress",
            f"Completion message with runtime statistics",
            f"No error messages or warnings (successful execution)"
        ])
        
        return outputs
        
    def _generate_troubleshooting(self, analysis: PipelineAnalysis) -> Dict[str, str]:
        """Generate troubleshooting guide"""
        issues = {}
        
        if analysis.dependencies:
            issues["API Authentication Errors"] = "Ensure all required API keys are properly configured in your environment"
            
        if 'template_variables' in analysis.features_used:
            issues["Template Resolution Errors"] = "Check that all input parameters are provided and template syntax is correct"
            
        if analysis.complexity_level == 'advanced':
            issues["Complex Logic Errors"] = "Review the pipeline configuration and ensure all advanced features are properly configured"
            
        issues["General Execution Errors"] = "Check the logs for specific error messages and verify your orchestrator installation"
        
        return issues
        
    def _generate_verification_steps(self, analysis: PipelineAnalysis) -> List[str]:
        """Generate steps to verify successful execution"""
        steps = [
            "Check that the pipeline completed without errors",
            "Verify all expected output files were created",
            "Review the output content for quality and accuracy"
        ]
        
        if 'data_processing' in analysis.features_used:
            steps.append("Validate the processed data matches your expectations")
            
        if 'llm_integration' in analysis.features_used:
            steps.append("Review generated content for relevance and quality")
            
        return steps
        
    def _create_feature_matrix(self, analyses: List[PipelineAnalysis]) -> FeatureMatrix:
        """Create comprehensive feature coverage matrix"""
        self.log("Creating feature coverage matrix...")
        
        feature_map = defaultdict(list)
        
        # Map each feature to pipelines that demonstrate it
        for analysis in analyses:
            for feature in analysis.features_used:
                feature_map[feature].append(analysis.name)
                
        # Calculate coverage statistics
        coverage_stats = {feature: len(pipelines) for feature, pipelines in feature_map.items()}
        
        # Identify gaps (features with no or minimal demonstration)
        gaps = [feature for feature, count in coverage_stats.items() if count < 2]
        
        return FeatureMatrix(
            feature_map=dict(feature_map),
            coverage_stats=coverage_stats,
            gaps=gaps
        )
        
    def _create_learning_path(self, analyses: List[PipelineAnalysis]) -> List[LearningModule]:
        """Create progressive learning path"""
        self.log("Creating progressive learning path...")
        
        # Group pipelines by complexity
        beginner_pipelines = [a.name for a in analyses if a.complexity_level == 'beginner']
        intermediate_pipelines = [a.name for a in analyses if a.complexity_level == 'intermediate']
        advanced_pipelines = [a.name for a in analyses if a.complexity_level == 'advanced']
        
        # Create learning modules
        modules = []
        
        if beginner_pipelines:
            modules.append(LearningModule(
                name="Getting Started with Orchestrator",
                description="Learn the fundamentals of pipeline creation with simple, practical examples",
                pipelines=sorted(beginner_pipelines),
                prerequisites=["Basic YAML knowledge"],
                key_concepts=[
                    "Pipeline structure and syntax",
                    "Template variables and data flow",
                    "Basic tool integration",
                    "Input/output handling"
                ],
                estimated_time="2-4 hours"
            ))
            
        if intermediate_pipelines:
            modules.append(LearningModule(
                name="Intermediate Pipeline Patterns",
                description="Build more sophisticated workflows with control flow, data processing, and AI integration",
                pipelines=sorted(intermediate_pipelines),
                prerequisites=["Completion of Getting Started module"],
                key_concepts=[
                    "Control flow patterns (conditionals, loops)",
                    "Data processing and transformation",
                    "LLM integration and prompt engineering",
                    "Multi-step workflow design"
                ],
                estimated_time="4-8 hours"
            ))
            
        if advanced_pipelines:
            modules.append(LearningModule(
                name="Advanced Orchestrator Techniques",
                description="Master complex patterns including system integration, error handling, and optimization",
                pipelines=sorted(advanced_pipelines),
                prerequisites=["Completion of Intermediate module"],
                key_concepts=[
                    "System integration and automation",
                    "Advanced error handling and recovery",
                    "Performance optimization",
                    "Modular architecture patterns",
                    "Security and production considerations"
                ],
                estimated_time="6-12 hours"
            ))
            
        return modules
        
    def _write_tutorial_suite(self, suite: TutorialSuite, output_dir: Path):
        """Write tutorial suite to files"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write individual tutorials
        tutorials_dir = output_dir / "pipelines"
        tutorials_dir.mkdir(exist_ok=True)
        
        for pipeline_name, tutorial in suite.tutorials.items():
            tutorial_file = tutorials_dir / f"{pipeline_name}.md"
            self._write_tutorial_markdown(tutorial, tutorial_file)
            
        # Write feature matrix
        matrix_file = output_dir / "feature_matrix.json"
        with open(matrix_file, 'w') as f:
            json.dump(asdict(suite.feature_matrix), f, indent=2)
            
        # Write learning path
        learning_file = output_dir / "learning_path.md"
        self._write_learning_path_markdown(suite.learning_path, learning_file)
        
        # Write overview and index
        index_file = output_dir / "README.md"
        self._write_tutorial_index(suite, index_file)
        
        # Write metadata
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(suite.metadata, f, indent=2)
            
        self.log(f"Tutorial suite written to {output_dir}")
        
    def _write_tutorial_markdown(self, tutorial: Dict[str, Any], file_path: Path):
        """Write individual tutorial as markdown file"""
        metadata = tutorial['metadata']
        overview = tutorial['overview']
        breakdown = tutorial['pipeline_breakdown']
        customization = tutorial['customization_guide']
        remixing = tutorial['remixing_instructions']
        exercise = tutorial['hands_on_exercise']
        
        content = f"""# Pipeline Tutorial: {metadata['pipeline_name']}

## Overview

**Complexity Level**: {metadata['complexity_level'].title()}  
**Difficulty Score**: {metadata['difficulty_score']}/100  
**Estimated Runtime**: {metadata['estimated_runtime']}  

### Purpose
{overview['purpose']}

### Use Cases
{self._format_list(overview['use_cases'])}

### Prerequisites
{self._format_list(overview['prerequisites'])}

### Key Concepts
{self._format_list(overview['key_concepts'])}

## Pipeline Breakdown

### Configuration Analysis
{self._format_dict(breakdown['configuration_analysis'])}

### Data Flow
{breakdown['data_flow']}

### Control Flow
{breakdown['control_flow']}

### Pipeline Configuration
```yaml
{breakdown['yaml_content']}
```

## Customization Guide

### Input Modifications
{self._format_list(customization['input_modifications'])}

### Parameter Tuning
{self._format_list(customization['parameter_tuning'])}

### Step Modifications
{self._format_list(customization['step_modifications'])}

### Output Customization
{self._format_list(customization['output_customization'])}

## Remixing Instructions

### Compatible Patterns
{self._format_list(remixing['compatible_patterns'])}

### Extension Ideas
{self._format_list(remixing['extension_ideas'])}

### Combination Examples
{self._format_list(remixing['combination_examples'])}

### Advanced Variations
{self._format_list(remixing['advanced_variations'])}

## Hands-On Exercise

### Execution Instructions
{self._format_list(exercise['execution_instructions'])}

### Expected Outputs
{self._format_list(exercise['expected_outputs'])}

### Troubleshooting
{self._format_dict(exercise['troubleshooting'])}

### Verification Steps
{self._format_list(exercise['verification_steps'])}

---

*Tutorial generated on {metadata['generated_at']}*
"""
        
        with open(file_path, 'w') as f:
            f.write(content)
            
    def _write_learning_path_markdown(self, learning_path: List[LearningModule], file_path: Path):
        """Write learning path as markdown file"""
        content = """# Orchestrator Learning Path

A progressive guide to mastering the orchestrator toolbox through hands-on examples.

"""
        
        for i, module in enumerate(learning_path, 1):
            content += f"""## Module {i}: {module.name}

**Description**: {module.description}  
**Estimated Time**: {module.estimated_time}  

### Prerequisites
{self._format_list(module.prerequisites)}

### Key Concepts
{self._format_list(module.key_concepts)}

### Pipelines in This Module
{self._format_list([f"[{pipeline}](pipelines/{pipeline}.md)" for pipeline in module.pipelines])}

"""
        
        with open(file_path, 'w') as f:
            f.write(content)
            
    def _write_tutorial_index(self, suite: TutorialSuite, file_path: Path):
        """Write tutorial index/overview file"""
        metadata = suite.metadata
        
        content = f"""# Orchestrator Pipeline Tutorials

Comprehensive tutorial documentation for all validated orchestrator example pipelines.

## Overview

**Pipeline Count**: {metadata['pipeline_count']}  
**Feature Count**: {metadata['feature_count']}  
**Learning Modules**: {metadata['learning_modules']}  
**Generated**: {metadata['generated_at']}  

## Getting Started

1. **[Learning Path](learning_path.md)** - Progressive skill-building guide
2. **[Feature Matrix](feature_matrix.json)** - Complete feature coverage mapping
3. **[Individual Tutorials](pipelines/)** - Detailed pipeline documentation

## Tutorial Categories

### By Complexity Level
"""
        
        # Group by complexity
        by_complexity = defaultdict(list)
        for pipeline_name, tutorial in suite.tutorials.items():
            complexity = tutorial['metadata']['complexity_level']
            by_complexity[complexity].append(pipeline_name)
            
        for complexity in ['beginner', 'intermediate', 'advanced']:
            if complexity in by_complexity:
                pipelines = sorted(by_complexity[complexity])
                content += f"\n#### {complexity.title()} ({len(pipelines)} pipelines)\n"
                for pipeline in pipelines:
                    content += f"- [{pipeline}](pipelines/{pipeline}.md)\n"
                    
        content += f"""

### By Feature Category

See the [feature matrix](feature_matrix.json) for complete feature-to-pipeline mapping.

## How to Use These Tutorials

1. **Start with the Learning Path** if you're new to orchestrator
2. **Browse by complexity** if you have specific skill level requirements  
3. **Search by feature** if you need examples of specific capabilities
4. **Follow remixing guides** to combine patterns for custom workflows

## Contributing

These tutorials are automatically generated from pipeline analysis. To update:

1. Modify example pipelines in the `examples/` directory
2. Run `python scripts/tutorial_generator.py` to regenerate documentation
3. Review and commit the updated tutorials

---

*Documentation generated on {metadata['generated_at']}*
"""
        
        with open(file_path, 'w') as f:
            f.write(content)
            
    def _format_list(self, items: List[str]) -> str:
        """Format list as markdown bullets"""
        if not items:
            return "- None specified"
        return '\n'.join(f"- {item}" for item in items)
        
    def _format_dict(self, items: Dict[str, str]) -> str:
        """Format dictionary as markdown list"""
        if not items:
            return "- None specified"
        return '\n'.join(f"- **{key}**: {value}" for key, value in items.items())

def main():
    """Main entry point for tutorial generator"""
    parser = argparse.ArgumentParser(description="Generate tutorial documentation for orchestrator pipelines")
    parser.add_argument('--output-dir', default='docs/tutorials', help='Output directory for tutorials')
    parser.add_argument('--pipeline-dir', default='examples', help='Directory containing pipeline YAML files')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze pipelines, don\'t generate tutorials')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Convert paths
    pipeline_dir = Path(args.pipeline_dir)
    output_dir = Path(args.output_dir)
    
    if not pipeline_dir.exists():
        print(f"Error: Pipeline directory {pipeline_dir} does not exist")
        return 1
        
    # Initialize generator
    generator = TutorialGenerator(verbose=args.verbose)
    
    try:
        if args.analyze_only:
            # Only perform analysis
            print("Analyzing pipelines...")
            analyses = generator._analyze_all_pipelines(pipeline_dir)
            
            # Print analysis summary
            print(f"\nAnalysis Summary:")
            print(f"Total pipelines: {len(analyses)}")
            
            by_complexity = defaultdict(int)
            for analysis in analyses:
                by_complexity[analysis.complexity_level] += 1
                
            for complexity, count in sorted(by_complexity.items()):
                print(f"  {complexity}: {count}")
                
        else:
            # Generate complete tutorial suite
            print("Generating tutorial documentation suite...")
            suite = generator.generate_tutorial_suite(pipeline_dir, output_dir)
            
            print(f"\nTutorial Generation Complete:")
            print(f"  Pipelines documented: {suite.metadata['pipeline_count']}")
            print(f"  Features covered: {suite.metadata['feature_count']}")
            print(f"  Learning modules: {suite.metadata['learning_modules']}")
            print(f"  Output directory: {output_dir}")
            
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
        
    return 0

if __name__ == '__main__':
    sys.exit(main())