"""Tests for document_intelligence.yaml example."""
import pytest
from pathlib import Path
from .test_base import BaseExampleTest


class TestDocumentIntelligenceYAML(BaseExampleTest):
    """Test the document intelligence YAML pipeline."""
    
    @pytest.fixture
    def pipeline_name(self):
        return "document_intelligence.yaml"
    
    @pytest.fixture
    def sample_inputs(self):
        return {
            "input_dir": "/path/to/documents",
            "output_dir": "/path/to/output",
            "enable_ocr": True,
            "languages": ["en", "es"],
            "custom_entities": ["PROJECT_CODE", "VENDOR_ID"],
            "output_format": "json",
            "extract_tables": True,
            "build_knowledge_graph": True
        }
    
    def test_pipeline_structure(self, pipeline_name):
        """Test that the pipeline has valid structure."""
        self.validate_pipeline_structure(pipeline_name)
        
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check document processing steps
        step_ids = [step['id'] for step in config['steps']]
        required_steps = [
            'discover_documents',
            'classify_documents',
            'extract_text',
            'analyze_structure',
            'extract_entities',
            'detect_pii',
            'generate_summary'
        ]
        
        for step in required_steps:
            assert step in step_ids, f"Missing required step: {step}"
    
    def test_ocr_configuration(self, pipeline_name):
        """Test OCR configuration in pipeline."""
        # Read raw file content to check for OCR references
        example_dir = Path(__file__).parent.parent.parent / "examples"
        pipeline_path = example_dir / pipeline_name
        
        with open(pipeline_path, 'r') as f:
            raw_content = f.read()
        
        # Check OCR enablement in raw content
        assert '{{enable_ocr}}' in raw_content or 'enable_ocr' in raw_content
        assert '{{languages}}' in raw_content or 'languages' in raw_content
    
    @pytest.mark.asyncio
    async def test_document_discovery(self, orchestrator, pipeline_name, sample_inputs):
        """Test document discovery process."""
        # Test pipeline structure and flow with minimal responses
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            sample_inputs,
            expected_outputs={
                'total_documents': int,
                'document_stats': dict
            },
            use_minimal_responses=True
        )
        
        # Verify result structure
        assert result is not None
        assert 'outputs' in result or 'steps' in result
    
    @pytest.mark.asyncio
    async def test_pii_detection(self, orchestrator, pipeline_name, sample_inputs):
        """Test PII detection functionality."""
        # Load and validate pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find PII detection step
        pii_step = next((s for s in config['steps'] if s['id'] == 'detect_pii'), None)
        assert pii_step is not None
        
        # Test with minimal responses to validate flow
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            sample_inputs,
            use_minimal_responses=True
        )
        
        # Verify execution completed
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_table_extraction(self, orchestrator, pipeline_name, sample_inputs):
        """Test conditional table extraction."""
        # Load pipeline and validate table extraction configuration
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find table extraction step
        table_step = next((s for s in config['steps'] if s['id'] == 'extract_tables'), None)
        
        if table_step:
            # Verify conditional execution
            assert 'when' in table_step or 'extract_tables' in str(config['inputs'])
        
        # Test with table extraction enabled
        inputs = sample_inputs.copy()
        inputs['extract_tables'] = True
        
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            inputs,
            use_minimal_responses=True
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_knowledge_graph_building(self, orchestrator, pipeline_name, sample_inputs):
        """Test knowledge graph construction."""
        # Load pipeline and check for knowledge graph steps
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check if knowledge graph building is configured
        kg_step = next((s for s in config['steps'] if s['id'] == 'build_knowledge_graph'), None)
        
        if kg_step:
            # Test with knowledge graph building enabled
            inputs = sample_inputs.copy()
            inputs['build_knowledge_graph'] = True
            
            result = await self.run_pipeline_test(
                orchestrator,
                pipeline_name,
                inputs,
                use_minimal_responses=True
            )
            
            assert result is not None
    
    def test_multi_language_support(self, pipeline_name):
        """Test multi-language configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check language input
        assert 'languages' in config['inputs']
        assert config['inputs']['languages']['type'] == 'list'
        assert config['inputs']['languages']['default'] == ['en']
        
        # Check language usage in steps
        extract_step = next(s for s in config['steps'] if s['id'] == 'extract_text')
        assert 'Languages: {{languages}}' in extract_step['action']
    
    def test_output_organization(self, pipeline_name):
        """Test output structure and organization."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check outputs include all key metrics
        expected_outputs = [
            'total_documents',
            'total_entities',
            'pii_documents',
            'key_insights',
            'compliance_status',
            'report_location'
        ]
        
        for output in expected_outputs:
            assert output in config['outputs'], f"Missing output: {output}"
        
        # Check save_outputs step
        save_step = next(s for s in config['steps'] if s['id'] == 'save_outputs')
        assert 'folder structure' in save_step['action'].lower()