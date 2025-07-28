Document Intelligence
=====================

This example demonstrates how to build a comprehensive document intelligence system that extracts insights, analyzes content, and processes various document formats using multi-modal AI capabilities. The system handles everything from simple text extraction to complex document understanding and knowledge extraction.

.. note::
   **Level:** Intermediate  
   **Duration:** 45-60 minutes  
   **Prerequisites:** Basic Python knowledge, understanding of document processing concepts, familiarity with OCR and NLP

Overview
--------

The Document Intelligence system provides:

1. **Multi-Format Support**: Process PDFs, Word docs, images, scanned documents
2. **Text Extraction**: OCR for scanned documents, layout preservation
3. **Structure Analysis**: Identify headers, tables, figures, and sections
4. **Entity Extraction**: Extract names, dates, amounts, and custom entities
5. **Semantic Analysis**: Understand document meaning and relationships
6. **Classification**: Automatically categorize and tag documents
7. **Knowledge Graph**: Build relationships between documents and entities

**Key Features:**
- Multi-modal AI for text and image understanding
- Table extraction and analysis
- Form field detection and extraction
- Multi-language support
- Document summarization
- Question answering on documents
- Compliance and PII detection

Quick Start
-----------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-org/orchestrator.git
   cd orchestrator
   
   # Install dependencies
   pip install -r requirements.txt
   pip install pytesseract pdf2image python-docx openpyxl
   
   # Set up environment variables
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export AZURE_DOCUMENT_INTELLIGENCE_KEY="your-azure-key"  # Optional
   
   # Run the example
   python examples/document_intelligence.py \
     --input-dir ./documents \
     --output-dir ./processed \
     --enable-ocr

Complete Implementation
-----------------------

Pipeline Configuration (YAML)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # document_intelligence_pipeline.yaml
   id: document_intelligence
   name: Multi-Modal Document Intelligence Pipeline
   version: "1.0"
   
   metadata:
     description: "Extract insights from documents using AI"
     author: "Data Science Team"
     tags: ["document-processing", "nlp", "ocr", "knowledge-extraction"]
   
   models:
     text_analyzer:
       provider: "openai"
       model: "gpt-4"
       temperature: 0.1
     vision_analyzer:
       provider: "openai"
       model: "gpt-4-vision-preview"
       temperature: 0.2
     entity_extractor:
       provider: "anthropic"
       model: "claude-opus-4-20250514"
       temperature: 0.1
   
   context:
     supported_formats: ["pdf", "docx", "xlsx", "png", "jpg", "tiff"]
     languages: ["en", "es", "fr", "de", "zh", "ja"]
     extraction_depth: "comprehensive"
   
   tasks:
     - id: discover_documents
       name: "Discover Documents"
       action: "scan_directory"
       parameters:
         path: "{{ inputs.input_dir }}"
         file_patterns: "{{ context.supported_formats }}"
         recursive: true
         include_metadata: true
       outputs:
         - document_list
         - total_size
         - file_types
     
     - id: classify_documents
       name: "Classify Document Types"
       action: "classify_documents"
       model: "text_analyzer"
       parallel: true
       for_each: "{{ discover_documents.document_list }}"
       parameters:
         document: "{{ item }}"
         classification_schema: <AUTO>Determine document categories</AUTO>
         confidence_threshold: 0.8
       dependencies:
         - discover_documents
       outputs:
         - document_types
         - classification_confidence
     
     - id: extract_text
       name: "Extract Text Content"
       action: "extract_document_text"
       parallel: true
       max_workers: 5
       for_each: "{{ discover_documents.document_list }}"
       parameters:
         document: "{{ item }}"
         preserve_layout: true
         enable_ocr: "{{ inputs.enable_ocr }}"
         ocr_languages: <AUTO>Detect languages for OCR</AUTO>
       dependencies:
         - classify_documents
       outputs:
         - extracted_text
         - page_layouts
         - extraction_metadata
     
     - id: analyze_structure
       name: "Analyze Document Structure"
       action: "analyze_document_structure"
       model: "vision_analyzer"
       parallel: true
       for_each: "{{ discover_documents.document_list }}"
       parameters:
         document: "{{ item }}"
         text_content: "{{ extract_text.extracted_text[item] }}"
         identify_elements: ["headers", "paragraphs", "tables", "figures", "lists"]
         extract_hierarchy: true
       dependencies:
         - extract_text
       outputs:
         - document_structure
         - section_hierarchy
         - visual_elements
     
     - id: extract_tables
       name: "Extract and Analyze Tables"
       action: "extract_tables"
       model: "vision_analyzer"
       condition: "analyze_structure.visual_elements.tables | length > 0"
       parameters:
         documents: "{{ analyze_structure.visual_elements.tables }}"
         extraction_mode: <AUTO>Choose between OCR and structured extraction</AUTO>
         clean_data: true
         infer_headers: true
       dependencies:
         - analyze_structure
       outputs:
         - extracted_tables
         - table_metadata
         - data_quality_scores
     
     - id: extract_entities
       name: "Extract Named Entities"
       action: "extract_entities"
       model: "entity_extractor"
       parallel: true
       for_each: "{{ extract_text.extracted_text }}"
       parameters:
         text: "{{ item.content }}"
         entity_types: <AUTO>Detect relevant entity types</AUTO>
         custom_entities: "{{ inputs.custom_entities }}"
         confidence_threshold: 0.7
       dependencies:
         - extract_text
       outputs:
         - entities
         - entity_relationships
         - entity_confidence
     
     - id: detect_pii
       name: "Detect PII and Sensitive Data"
       action: "scan_for_pii"
       parallel: true
       for_each: "{{ extract_text.extracted_text }}"
       parameters:
         text: "{{ item.content }}"
         pii_types: ["ssn", "credit_card", "email", "phone", "address", "medical"]
         redaction_mode: "mask"
       dependencies:
         - extract_text
       outputs:
         - pii_findings
         - redacted_text
         - compliance_report
     
     - id: analyze_content
       name: "Semantic Content Analysis"
       action: "analyze_document_content"
       model: "text_analyzer"
       parallel: true
       for_each: "{{ extract_text.extracted_text }}"
       parameters:
         text: "{{ item.content }}"
         document_type: "{{ classify_documents.document_types[item.id] }}"
         analysis_depth: <AUTO>Determine based on document length and type</AUTO>
         extract_key_points: true
         identify_topics: true
       dependencies:
         - extract_entities
         - analyze_structure
       outputs:
         - content_analysis
         - key_points
         - topic_model
         - sentiment_analysis
     
     - id: generate_summary
       name: "Generate Document Summaries"
       action: "summarize_document"
       model: "text_analyzer"
       parallel: true
       for_each: "{{ extract_text.extracted_text }}"
       parameters:
         content: "{{ item.content }}"
         key_points: "{{ analyze_content.key_points[item.id] }}"
         summary_length: <AUTO>Determine based on document length</AUTO>
         summary_style: "executive"
         include_entities: true
       dependencies:
         - analyze_content
       outputs:
         - summaries
         - summary_metadata
     
     - id: build_knowledge_graph
       name: "Build Knowledge Graph"
       action: "construct_knowledge_graph"
       parameters:
         entities: "{{ extract_entities.entities }}"
         relationships: "{{ extract_entities.entity_relationships }}"
         document_metadata: "{{ classify_documents.document_types }}"
         include_cross_references: true
       dependencies:
         - extract_entities
         - analyze_content
       outputs:
         - knowledge_graph
         - relationship_matrix
         - entity_clusters
     
     - id: generate_insights
       name: "Generate Document Insights"
       action: "generate_insights"
       model: "text_analyzer"
       parameters:
         analyses: "{{ analyze_content.content_analysis }}"
         knowledge_graph: "{{ build_knowledge_graph.knowledge_graph }}"
         document_types: "{{ classify_documents.document_types }}"
         insight_types: <AUTO>Generate relevant insights based on content</AUTO>
       dependencies:
         - build_knowledge_graph
       outputs:
         - insights
         - recommendations
         - anomalies
     
     - id: create_report
       name: "Create Intelligence Report"
       action: "compile_intelligence_report"
       parameters:
         summaries: "{{ generate_summary.summaries }}"
         entities: "{{ extract_entities.entities }}"
         insights: "{{ generate_insights.insights }}"
         pii_report: "{{ detect_pii.compliance_report }}"
         format: "{{ inputs.output_format }}"
       dependencies:
         - generate_insights
       outputs:
         - intelligence_report
         - executive_summary
         - detailed_findings

Python Implementation
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # document_intelligence.py
   import asyncio
   import os
   from pathlib import Path
   from typing import Dict, List, Any, Optional, Union
   import json
   from datetime import datetime
   import pytesseract
   from PIL import Image
   import pdfplumber
   import docx
   import pandas as pd
   import networkx as nx
   
   from orchestrator import Orchestrator
   from orchestrator.tools.document_tools import (
       DocumentExtractorTool,
       OCRTool,
       TableExtractorTool,
       EntityExtractorTool
   )
   from orchestrator.tools.nlp_tools import (
       TextAnalyzerTool,
       SummarizerTool,
       PIIDetectorTool
   )
   from orchestrator.integrations.knowledge_graph import KnowledgeGraphBuilder
   
   
   class DocumentIntelligenceSystem:
       """
       Multi-modal document intelligence system for comprehensive document analysis.
       
       Features:
       - Multi-format document processing
       - OCR and layout analysis
       - Entity and relationship extraction
       - Knowledge graph construction
       - Compliance and PII detection
       """
       
       def __init__(self, config: Dict[str, Any]):
           self.config = config
           self.orchestrator = None
           self.knowledge_graph = None
           self._setup_system()
       
       def _setup_system(self):
           """Initialize document intelligence components."""
           self.orchestrator = Orchestrator()
           
           # Register AI models
           self._register_models()
           
           # Initialize tools
           self.tools = {
               'document_extractor': DocumentExtractorTool(),
               'ocr': OCRTool(
                   languages=self.config.get('ocr_languages', ['eng'])
               ),
               'table_extractor': TableExtractorTool(),
               'entity_extractor': EntityExtractorTool(self.config),
               'text_analyzer': TextAnalyzerTool(),
               'summarizer': SummarizerTool(),
               'pii_detector': PIIDetectorTool()
           }
           
           # Initialize knowledge graph
           self.knowledge_graph = KnowledgeGraphBuilder()
       
       async def process_documents(
           self,
           input_dir: str,
           output_dir: str,
           enable_ocr: bool = True,
           custom_entities: Optional[List[str]] = None,
           output_format: str = 'json',
           **kwargs
       ) -> Dict[str, Any]:
           """
           Process documents and extract intelligence.
           
           Args:
               input_dir: Directory containing documents
               output_dir: Output directory for results
               enable_ocr: Enable OCR for scanned documents
               custom_entities: Custom entity types to extract
               output_format: Output format (json, pdf, html)
               
           Returns:
               Document intelligence report
           """
           print(f"📄 Starting document intelligence processing for: {input_dir}")
           
           # Prepare context
           context = {
               'input_dir': input_dir,
               'output_dir': output_dir,
               'enable_ocr': enable_ocr,
               'custom_entities': custom_entities or [],
               'output_format': output_format,
               'timestamp': datetime.now().isoformat(),
               **kwargs
           }
           
           # Execute pipeline
           try:
               results = await self.orchestrator.execute_pipeline(
                   'document_intelligence_pipeline.yaml',
                   context=context,
                   progress_callback=self._progress_callback
               )
               
               # Process results
               intelligence_report = await self._process_results(results)
               
               # Save outputs
               await self._save_outputs(intelligence_report, output_dir, output_format)
               
               # Update knowledge graph
               await self._update_knowledge_graph(results)
               
               return intelligence_report
               
           except Exception as e:
               print(f"❌ Document processing failed: {str(e)}")
               raise
       
       async def _progress_callback(self, task_id: str, progress: float, message: str):
           """Handle progress updates."""
           icons = {
               'discover_documents': '📁',
               'classify_documents': '🏷️',
               'extract_text': '📝',
               'analyze_structure': '🏗️',
               'extract_tables': '📊',
               'extract_entities': '🔍',
               'detect_pii': '🔐',
               'analyze_content': '🧠',
               'generate_summary': '📋',
               'build_knowledge_graph': '🕸️',
               'generate_insights': '💡',
               'create_report': '📄'
           }
           icon = icons.get(task_id, '▶️')
           print(f"{icon} {task_id}: {progress:.0%} - {message}")
       
       async def _process_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
           """Process and format intelligence results."""
           report = {
               'summary': {
                   'total_documents': len(results.get('discover_documents', {}).get('document_list', [])),
                   'document_types': {},
                   'total_entities': 0,
                   'pii_findings': 0,
                   'key_insights': []
               },
               'documents': {},
               'entities': {},
               'knowledge_graph': {},
               'compliance': {},
               'insights': []
           }
           
           # Process document classifications
           if 'classify_documents' in results:
               classifications = results['classify_documents']['document_types']
               for doc_type, count in self._count_document_types(classifications).items():
                   report['summary']['document_types'][doc_type] = count
           
           # Process extracted entities
           if 'extract_entities' in results:
               entities = results['extract_entities']['entities']
               report['entities'] = self._organize_entities(entities)
               report['summary']['total_entities'] = sum(
                   len(ents) for ents in report['entities'].values()
               )
           
           # Process PII findings
           if 'detect_pii' in results:
               pii_findings = results['detect_pii']['pii_findings']
               report['compliance'] = {
                   'pii_summary': self._summarize_pii_findings(pii_findings),
                   'documents_with_pii': len([f for f in pii_findings if f]),
                   'redacted_documents': results['detect_pii'].get('redacted_text', {})
               }
               report['summary']['pii_findings'] = report['compliance']['documents_with_pii']
           
           # Process insights
           if 'generate_insights' in results:
               insights = results['generate_insights']['insights']
               report['insights'] = insights
               report['summary']['key_insights'] = insights[:5]  # Top 5 insights
           
           # Process knowledge graph
           if 'build_knowledge_graph' in results:
               kg_data = results['build_knowledge_graph']
               report['knowledge_graph'] = {
                   'nodes': len(kg_data['knowledge_graph']['nodes']),
                   'edges': len(kg_data['knowledge_graph']['edges']),
                   'clusters': kg_data.get('entity_clusters', [])
               }
           
           # Process individual documents
           for doc_id, doc_data in self._organize_document_results(results).items():
               report['documents'][doc_id] = doc_data
           
           return report
       
       def _organize_document_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
           """Organize results by document."""
           documents = {}
           
           doc_list = results.get('discover_documents', {}).get('document_list', [])
           
           for doc in doc_list:
               doc_id = doc['id']
               documents[doc_id] = {
                   'metadata': doc,
                   'classification': results.get('classify_documents', {}).get('document_types', {}).get(doc_id),
                   'extracted_text': results.get('extract_text', {}).get('extracted_text', {}).get(doc_id),
                   'structure': results.get('analyze_structure', {}).get('document_structure', {}).get(doc_id),
                   'entities': results.get('extract_entities', {}).get('entities', {}).get(doc_id),
                   'summary': results.get('generate_summary', {}).get('summaries', {}).get(doc_id),
                   'insights': results.get('analyze_content', {}).get('content_analysis', {}).get(doc_id)
               }
           
           return documents
       
       async def _save_outputs(
           self,
           report: Dict[str, Any],
           output_dir: str,
           output_format: str
       ):
           """Save processing outputs."""
           output_path = Path(output_dir)
           output_path.mkdir(parents=True, exist_ok=True)
           
           timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
           
           if output_format == 'json':
               output_file = output_path / f'intelligence_report_{timestamp}.json'
               with open(output_file, 'w') as f:
                   json.dump(report, f, indent=2, default=str)
               print(f"✅ Report saved to: {output_file}")
           
           elif output_format == 'html':
               html_content = self._generate_html_report(report)
               output_file = output_path / f'intelligence_report_{timestamp}.html'
               output_file.write_text(html_content)
               print(f"✅ HTML report saved to: {output_file}")
           
           # Save knowledge graph visualization
           if report.get('knowledge_graph'):
               graph_file = output_path / f'knowledge_graph_{timestamp}.png'
               await self._save_knowledge_graph_visualization(
                   report['knowledge_graph'],
                   graph_file
               )
       
       async def _update_knowledge_graph(self, results: Dict[str, Any]):
           """Update the knowledge graph with new information."""
           if 'build_knowledge_graph' not in results:
               return
           
           kg_data = results['build_knowledge_graph']['knowledge_graph']
           
           # Add nodes
           for node in kg_data['nodes']:
               self.knowledge_graph.add_entity(
                   entity_id=node['id'],
                   entity_type=node['type'],
                   properties=node.get('properties', {})
               )
           
           # Add edges
           for edge in kg_data['edges']:
               self.knowledge_graph.add_relationship(
                   source=edge['source'],
                   target=edge['target'],
                   relationship_type=edge['type'],
                   properties=edge.get('properties', {})
               )

Advanced Document Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class AdvancedDocumentProcessor:
       """Advanced document processing capabilities."""
       
       async def process_complex_layout(
           self,
           document_path: str,
           preserve_layout: bool = True
       ) -> Dict[str, Any]:
           """Process documents with complex layouts."""
           # Use vision model for layout understanding
           layout_analysis = await self.analyze_visual_layout(document_path)
           
           # Extract content while preserving structure
           structured_content = {
               'headers': [],
               'paragraphs': [],
               'tables': [],
               'figures': [],
               'sidebars': [],
               'footnotes': []
           }
           
           for element in layout_analysis['elements']:
               content = await self.extract_element_content(element)
               structured_content[element['type']].append({
                   'content': content,
                   'position': element['bbox'],
                   'page': element['page'],
                   'confidence': element['confidence']
               })
           
           return structured_content
       
       async def extract_form_fields(
           self,
           document_path: str
       ) -> Dict[str, Any]:
           """Extract form fields and their values."""
           # Detect form fields using vision model
           form_fields = await self.detect_form_fields(document_path)
           
           extracted_data = {}
           for field in form_fields:
               field_name = field['label']
               field_value = await self.extract_field_value(field)
               extracted_data[field_name] = {
                   'value': field_value,
                   'type': field['field_type'],
                   'confidence': field['confidence'],
                   'location': field['bbox']
               }
           
           return extracted_data
       
       async def compare_documents(
           self,
           doc1_path: str,
           doc2_path: str
       ) -> Dict[str, Any]:
           """Compare two documents for similarities and differences."""
           # Extract content from both documents
           doc1_content = await self.extract_document_content(doc1_path)
           doc2_content = await self.extract_document_content(doc2_path)
           
           # Perform comparison
           comparison = {
               'similarity_score': await self.calculate_similarity(
                   doc1_content,
                   doc2_content
               ),
               'common_entities': await self.find_common_entities(
                   doc1_content['entities'],
                   doc2_content['entities']
               ),
               'differences': await self.identify_differences(
                   doc1_content,
                   doc2_content
               ),
               'version_changes': await self.detect_version_changes(
                   doc1_content,
                   doc2_content
               )
           }
           
           return comparison

Table Processing
^^^^^^^^^^^^^^^^

.. code-block:: python

   class TableProcessor:
       """Advanced table extraction and analysis."""
       
       async def extract_complex_table(
           self,
           table_image: Union[str, Image.Image]
       ) -> pd.DataFrame:
           """Extract complex tables with merged cells and nested headers."""
           # Use vision model to understand table structure
           table_structure = await self.analyze_table_structure(table_image)
           
           # Extract cell contents
           cells = []
           for row in table_structure['rows']:
               row_data = []
               for cell in row['cells']:
                   content = await self.extract_cell_content(cell)
                   row_data.append({
                       'value': content,
                       'colspan': cell.get('colspan', 1),
                       'rowspan': cell.get('rowspan', 1)
                   })
               cells.append(row_data)
           
           # Reconstruct table with proper structure
           df = self.reconstruct_dataframe(cells, table_structure)
           
           # Clean and normalize data
           df = await self.clean_table_data(df)
           
           return df
       
       async def analyze_table_data(
           self,
           df: pd.DataFrame
       ) -> Dict[str, Any]:
           """Analyze extracted table data."""
           analysis = {
               'summary_statistics': df.describe().to_dict(),
               'data_types': df.dtypes.to_dict(),
               'missing_values': df.isnull().sum().to_dict(),
               'anomalies': await self.detect_table_anomalies(df),
               'insights': await self.generate_table_insights(df)
           }
           
           return analysis

Question Answering
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class DocumentQA:
       """Question answering on documents."""
       
       def __init__(self, model_config: Dict[str, Any]):
           self.model = self._init_qa_model(model_config)
           self.document_store = {}
       
       async def answer_question(
           self,
           question: str,
           document_id: Optional[str] = None
       ) -> Dict[str, Any]:
           """Answer questions about documents."""
           # Get relevant context
           if document_id:
               context = self.document_store.get(document_id, {})
           else:
               # Search across all documents
               context = await self.search_relevant_context(question)
           
           # Generate answer
           answer = await self.model.generate_answer(
               question=question,
               context=context
           )
           
           return {
               'question': question,
               'answer': answer['text'],
               'confidence': answer['confidence'],
               'sources': answer['sources'],
               'relevant_excerpts': answer.get('excerpts', [])
           }
       
       async def generate_faq(
           self,
           document_content: str
       ) -> List[Dict[str, str]]:
           """Generate FAQ from document content."""
           # Extract key topics
           topics = await self.extract_key_topics(document_content)
           
           # Generate questions for each topic
           faq = []
           for topic in topics:
               questions = await self.generate_questions_for_topic(
                   topic,
                   document_content
               )
               
               for question in questions:
                   answer = await self.answer_question(
                       question,
                       document_content
                   )
                   faq.append({
                       'question': question,
                       'answer': answer['answer'],
                       'topic': topic
                   })
           
           return faq

Running the System
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # main.py
   import asyncio
   import argparse
   from document_intelligence import DocumentIntelligenceSystem
   
   async def main():
       parser = argparse.ArgumentParser(description='Document Intelligence System')
       parser.add_argument('--input-dir', required=True, help='Input directory')
       parser.add_argument('--output-dir', default='./processed', 
                          help='Output directory')
       parser.add_argument('--enable-ocr', action='store_true',
                          help='Enable OCR for scanned documents')
       parser.add_argument('--languages', nargs='+', default=['en'],
                          help='Languages for processing')
       parser.add_argument('--output-format', choices=['json', 'html', 'pdf'],
                          default='json')
       parser.add_argument('--extract-tables', action='store_true',
                          help='Extract and analyze tables')
       parser.add_argument('--build-knowledge-graph', action='store_true',
                          help='Build knowledge graph from entities')
       
       args = parser.parse_args()
       
       # Configuration
       config = {
           'openai_api_key': os.getenv('OPENAI_API_KEY'),
           'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
           'azure_key': os.getenv('AZURE_DOCUMENT_INTELLIGENCE_KEY'),
           'ocr_languages': args.languages,
           'enable_table_extraction': args.extract_tables,
           'enable_knowledge_graph': args.build_knowledge_graph
       }
       
       # Create system
       doc_intelligence = DocumentIntelligenceSystem(config)
       
       # Process documents
       results = await doc_intelligence.process_documents(
           input_dir=args.input_dir,
           output_dir=args.output_dir,
           enable_ocr=args.enable_ocr,
           output_format=args.output_format
       )
       
       # Display results
       print("\n📊 Document Processing Complete!")
       print(f"Documents Processed: {results['summary']['total_documents']}")
       print(f"Entities Extracted: {results['summary']['total_entities']}")
       print(f"Documents with PII: {results['summary']['pii_findings']}")
       
       print("\n📁 Document Types:")
       for doc_type, count in results['summary']['document_types'].items():
           print(f"  - {doc_type}: {count}")
       
       print("\n💡 Key Insights:")
       for i, insight in enumerate(results['summary']['key_insights'], 1):
           print(f"{i}. {insight}")
       
       if results.get('knowledge_graph'):
           print(f"\n🕸️ Knowledge Graph:")
           print(f"  - Nodes: {results['knowledge_graph']['nodes']}")
           print(f"  - Relationships: {results['knowledge_graph']['edges']}")
   
   if __name__ == "__main__":
       asyncio.run(main())

Best Practices
--------------

1. **Format Handling**: Use appropriate tools for each document format
2. **OCR Quality**: Pre-process images to improve OCR accuracy
3. **Entity Validation**: Validate extracted entities against known databases
4. **Privacy First**: Always check for PII before processing
5. **Incremental Processing**: Process large document sets in batches
6. **Version Control**: Track document versions and changes
7. **Metadata Preservation**: Maintain document metadata throughout processing

Summary
-------

The Document Intelligence system demonstrates:

- Multi-modal AI for comprehensive document understanding
- Automated extraction of structure, entities, and insights
- Knowledge graph construction from document relationships
- PII detection and compliance reporting
- Table extraction and analysis
- Question answering capabilities

This system provides a foundation for building intelligent document processing solutions for various industries and use cases.