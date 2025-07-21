"""Tests for document_intelligence.yaml example."""
import pytest
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
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find extract_text step
        extract_step = next(s for s in config['steps'] if s['id'] == 'extract_text')
        
        # Check OCR enablement
        assert '{{enable_ocr}}' in str(extract_step)
        assert '{{languages}}' in str(extract_step)
    
    @pytest.mark.asyncio
    async def test_document_discovery(self, orchestrator, pipeline_name, sample_inputs):
        """Test document discovery process."""
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                if step.get('id') == 'discover_documents':
                    return {
                        'result': {
                            'document_list': [
                                {
                                    'id': 'doc1',
                                    'name': 'contract.pdf',
                                    'path': '/docs/contract.pdf',
                                    'type': 'pdf',
                                    'size': 1024000
                                },
                                {
                                    'id': 'doc2',
                                    'name': 'invoice.jpg',
                                    'path': '/docs/invoice.jpg',
                                    'type': 'image',
                                    'size': 512000
                                },
                                {
                                    'id': 'doc3',
                                    'name': 'report.docx',
                                    'path': '/docs/report.docx',
                                    'type': 'docx',
                                    'size': 2048000
                                }
                            ],
                            'total_size': 3584000,
                            'file_types': {'pdf': 1, 'image': 1, 'docx': 1}
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=sample_inputs
            )
            
            # Verify discovery was called
            discovery_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'discover_documents'
            ]
            assert len(discovery_calls) == 1
    
    @pytest.mark.asyncio
    async def test_pii_detection(self, orchestrator, pipeline_name):
        """Test PII detection functionality."""
        inputs = {
            "input_dir": "/test/docs",
            "enable_ocr": False
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'detect_pii':
                    return {
                        'result': {
                            'doc1': {
                                'pii_found': True,
                                'types': ['SSN', 'CREDIT_CARD'],
                                'locations': [
                                    {'type': 'SSN', 'page': 1, 'line': 5},
                                    {'type': 'CREDIT_CARD', 'page': 2, 'line': 10}
                                ],
                                'risk_level': 'high'
                            },
                            'doc2': {
                                'pii_found': False,
                                'types': [],
                                'risk_level': 'low'
                            }
                        }
                    }
                elif step_id == 'create_compliance_report':
                    return {
                        'result': {
                            'overall_status': 'ATTENTION_REQUIRED',
                            'documents_with_pii': 1,
                            'high_risk_documents': 1,
                            'recommendations': [
                                'Redact SSN on page 1',
                                'Remove credit card info on page 2'
                            ]
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify PII detection and compliance report
            pii_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'detect_pii'
            ]
            compliance_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'create_compliance_report'
            ]
            
            assert len(pii_calls) > 0
            assert len(compliance_calls) > 0
    
    @pytest.mark.asyncio
    async def test_table_extraction(self, orchestrator, pipeline_name):
        """Test conditional table extraction."""
        inputs = {
            "input_dir": "/test/docs",
            "extract_tables": True
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'analyze_structure':
                    return {
                        'result': {
                            'tables': [
                                {
                                    'document': 'doc1',
                                    'table_locations': [
                                        {'page': 1, 'bbox': [100, 200, 500, 400]}
                                    ]
                                }
                            ]
                        }
                    }
                elif step_id == 'extract_tables':
                    return {
                        'result': {
                            'doc1': {
                                'tables': [
                                    {
                                        'headers': ['Name', 'Amount', 'Date'],
                                        'rows': [
                                            ['Item A', '100.00', '2024-01-01'],
                                            ['Item B', '200.00', '2024-01-02']
                                        ]
                                    }
                                ]
                            }
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify table extraction was triggered
            table_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'extract_tables'
            ]
            assert len(table_calls) > 0
    
    @pytest.mark.asyncio
    async def test_knowledge_graph_building(self, orchestrator, pipeline_name):
        """Test knowledge graph construction."""
        inputs = {
            "input_dir": "/test/docs",
            "build_knowledge_graph": True
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'extract_entities':
                    return {
                        'result': {
                            'doc1': [
                                {'text': 'John Doe', 'type': 'PERSON'},
                                {'text': 'Acme Corp', 'type': 'ORG'}
                            ],
                            'doc2': [
                                {'text': 'Jane Smith', 'type': 'PERSON'},
                                {'text': 'Acme Corp', 'type': 'ORG'}
                            ]
                        }
                    }
                elif step_id == 'extract_relationships':
                    return {
                        'result': [
                            {
                                'source': 'John Doe',
                                'target': 'Acme Corp',
                                'type': 'WORKS_FOR'
                            },
                            {
                                'source': 'Jane Smith',
                                'target': 'Acme Corp',
                                'type': 'WORKS_FOR'
                            }
                        ]
                    }
                elif step_id == 'build_knowledge_graph':
                    return {
                        'result': {
                            'node_count': 3,
                            'edge_count': 2,
                            'communities': [
                                ['John Doe', 'Jane Smith', 'Acme Corp']
                            ]
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify knowledge graph steps
            kg_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'build_knowledge_graph'
            ]
            assert len(kg_calls) > 0
    
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