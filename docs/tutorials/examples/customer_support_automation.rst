Customer Support Automation
===========================

This example demonstrates how to build a comprehensive customer support automation system that handles ticket management, automated responses, knowledge base integration, and seamless handoff to human agents when needed. The system uses AI to understand customer issues and provide intelligent solutions.

.. note::
   **Level:** Advanced  
   **Duration:** 75-90 minutes  
   **Prerequisites:** Python knowledge, understanding of customer support workflows, familiarity with ticketing systems, API integration experience

Overview
--------

The Customer Support Automation system provides:

1. **Ticket Management**: Automated ticket creation, categorization, and routing
2. **Smart Auto-Response**: AI-powered responses for common issues
3. **Knowledge Base Integration**: Search and suggest relevant articles
4. **Escalation Management**: Intelligent escalation to human agents
5. **Multi-Language Support**: Handle support in multiple languages
6. **Sentiment Tracking**: Monitor customer satisfaction in real-time
7. **Analytics Dashboard**: Comprehensive support metrics and insights

**Key Features:**
- Integration with popular ticketing systems (Zendesk, Freshdesk, etc.)
- Natural language understanding for ticket classification
- Automated workflow triggers
- SLA monitoring and alerts
- Customer history and context awareness
- Proactive support suggestions
- Self-service portal integration

Quick Start
-----------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-org/orchestrator.git
   cd orchestrator
   
   # Install dependencies
   pip install -r requirements.txt
   pip install zenpy freshdesk jira
   
   # Set up environment variables
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export ZENDESK_SUBDOMAIN="your-subdomain"
   export ZENDESK_EMAIL="your-email"
   export ZENDESK_TOKEN="your-token"
   
   # Run the example
   python examples/customer_support_automation.py \
     --ticketing-system zendesk \
     --auto-respond \
     --languages "en,es,fr"

Complete Implementation
-----------------------

Pipeline Configuration (YAML)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # customer_support_automation.yaml
   id: customer_support_automation
   name: AI-Powered Customer Support Automation
   version: "1.0"
   
   metadata:
     description: "Intelligent customer support with automated ticket handling"
     author: "Support Team"
     tags: ["support", "automation", "ticketing", "customer-service"]
   
   models:
     ticket_classifier:
       provider: "openai"
       model: "gpt-4"
       temperature: 0.2
     response_generator:
       provider: "anthropic"
       model: "claude-3-opus"
       temperature: 0.3
     sentiment_analyzer:
       provider: "openai"
       model: "gpt-3.5-turbo"
       temperature: 0.1
   
   context:
     ticketing_system: "{{ inputs.ticketing_system }}"
     auto_respond_enabled: "{{ inputs.auto_respond }}"
     languages: "{{ inputs.languages }}"
     business_hours: "{{ inputs.business_hours }}"
     sla_policies: "{{ inputs.sla_policies }}"
   
   tasks:
     - id: receive_ticket
       name: "Receive Support Ticket"
       action: "fetch_new_ticket"
       parameters:
         source: "{{ context.ticketing_system }}"
         ticket_id: "{{ inputs.ticket_id }}"
         include_history: true
         include_customer_data: true
       outputs:
         - ticket_data
         - customer_info
         - conversation_history
     
     - id: detect_language
       name: "Detect Ticket Language"
       action: "language_detection"
       parameters:
         text: "{{ receive_ticket.ticket_data.description }}"
         supported_languages: "{{ context.languages }}"
         confidence_threshold: 0.8
       dependencies:
         - receive_ticket
       outputs:
         - detected_language
         - confidence_score
         - needs_translation
     
     - id: analyze_sentiment
       name: "Analyze Customer Sentiment"
       action: "sentiment_analysis"
       model: "sentiment_analyzer"
       parameters:
         text: "{{ receive_ticket.ticket_data.description }}"
         history: "{{ receive_ticket.conversation_history }}"
         detect_urgency: true
         detect_frustration: true
       dependencies:
         - receive_ticket
       outputs:
         - sentiment_score
         - emotion_labels
         - urgency_level
         - frustration_detected
     
     - id: extract_entities
       name: "Extract Key Information"
       action: "entity_extraction"
       parameters:
         text: "{{ receive_ticket.ticket_data.description }}"
         entity_types: ["product", "issue_type", "order_number", "account_id"]
         use_customer_context: true
       dependencies:
         - receive_ticket
       outputs:
         - extracted_entities
         - confidence_scores
     
     - id: classify_ticket
       name: "Classify Ticket Category"
       action: "ticket_classification"
       model: "ticket_classifier"
       parameters:
         title: "{{ receive_ticket.ticket_data.subject }}"
         description: "{{ receive_ticket.ticket_data.description }}"
         entities: "{{ extract_entities.extracted_entities }}"
         classification_taxonomy: <AUTO>Use company-specific taxonomy</AUTO>
       dependencies:
         - extract_entities
       outputs:
         - primary_category
         - secondary_categories
         - tags
         - priority_score
     
     - id: search_knowledge_base
       name: "Search Knowledge Base"
       action: "kb_search"
       parameters:
         query: "{{ receive_ticket.ticket_data.description }}"
         category: "{{ classify_ticket.primary_category }}"
         limit: 5
         include_internal_kb: <AUTO>Based on agent availability</AUTO>
       dependencies:
         - classify_ticket
       outputs:
         - relevant_articles
         - solution_found
         - confidence_score
     
     - id: check_automation_eligibility
       name: "Check if Can Auto-Respond"
       action: "automation_check"
       parameters:
         category: "{{ classify_ticket.primary_category }}"
         sentiment: "{{ analyze_sentiment.sentiment_score }}"
         kb_confidence: "{{ search_knowledge_base.confidence_score }}"
         customer_tier: "{{ receive_ticket.customer_info.tier }}"
         previous_interactions: "{{ receive_ticket.customer_info.ticket_count }}"
       dependencies:
         - search_knowledge_base
         - analyze_sentiment
       outputs:
         - can_automate
         - automation_reason
         - risk_score
     
     - id: generate_response
       name: "Generate Automated Response"
       action: "generate_support_response"
       model: "response_generator"
       condition: "check_automation_eligibility.can_automate == true"
       parameters:
         ticket_content: "{{ receive_ticket.ticket_data }}"
         kb_articles: "{{ search_knowledge_base.relevant_articles }}"
         customer_info: "{{ receive_ticket.customer_info }}"
         language: "{{ detect_language.detected_language }}"
         tone: <AUTO>Match customer tone and formality</AUTO>
         include_solution_steps: true
         include_kb_links: true
       dependencies:
         - check_automation_eligibility
       outputs:
         - response_text
         - solution_steps
         - kb_references
         - follow_up_needed
     
     - id: translate_response
       name: "Translate Response if Needed"
       action: "translate_text"
       condition: "detect_language.needs_translation == true"
       parameters:
         text: "{{ generate_response.response_text }}"
         source_language: "en"
         target_language: "{{ detect_language.detected_language }}"
         preserve_formatting: true
       dependencies:
         - generate_response
       outputs:
         - translated_text
         - translation_confidence
     
     - id: update_ticket
       name: "Update Ticket in System"
       action: "update_ticket"
       parameters:
         ticket_id: "{{ receive_ticket.ticket_data.id }}"
         category: "{{ classify_ticket.primary_category }}"
         tags: "{{ classify_ticket.tags }}"
         priority: <AUTO>Calculate based on urgency and customer tier</AUTO>
         status: "{{ 'pending' if not check_automation_eligibility.can_automate else 'open' }}"
       dependencies:
         - classify_ticket
       outputs:
         - update_status
         - ticket_url
     
     - id: send_response
       name: "Send Response to Customer"
       action: "send_ticket_response"
       condition: "generate_response.response_text != null"
       parameters:
         ticket_id: "{{ receive_ticket.ticket_data.id }}"
         response: "{{ translate_response.translated_text or generate_response.response_text }}"
         is_public: true
         close_ticket: "{{ not generate_response.follow_up_needed }}"
       dependencies:
         - generate_response
         - update_ticket
       outputs:
         - response_sent
         - response_id
     
     - id: assign_to_agent
       name: "Assign to Human Agent"
       action: "agent_assignment"
       condition: "check_automation_eligibility.can_automate == false"
       parameters:
         ticket_id: "{{ receive_ticket.ticket_data.id }}"
         category: "{{ classify_ticket.primary_category }}"
         priority: "{{ classify_ticket.priority_score }}"
         required_skills: <AUTO>Determine based on ticket content</AUTO>
         context_package: {
           "classification": "{{ classify_ticket }}",
           "sentiment": "{{ analyze_sentiment }}",
           "kb_search": "{{ search_knowledge_base }}",
           "entities": "{{ extract_entities }}"
         }
       dependencies:
         - check_automation_eligibility
       outputs:
         - assigned_agent
         - assignment_reason
         - estimated_response_time
     
     - id: monitor_sla
       name: "Monitor SLA Compliance"
       action: "sla_monitoring"
       parameters:
         ticket_id: "{{ receive_ticket.ticket_data.id }}"
         customer_tier: "{{ receive_ticket.customer_info.tier }}"
         priority: "{{ classify_ticket.priority_score }}"
         sla_policies: "{{ context.sla_policies }}"
       dependencies:
         - update_ticket
       outputs:
         - sla_status
         - time_remaining
         - escalation_needed
     
     - id: log_interaction
       name: "Log Support Interaction"
       action: "log_to_analytics"
       parameters:
         ticket_id: "{{ receive_ticket.ticket_data.id }}"
         automation_status: "{{ 'automated' if check_automation_eligibility.can_automate else 'manual' }}"
         category: "{{ classify_ticket.primary_category }}"
         sentiment: "{{ analyze_sentiment.sentiment_score }}"
         resolution_time: "{{ calculate_resolution_time() }}"
         kb_helpful: "{{ search_knowledge_base.solution_found }}"
       dependencies:
         - send_response
         - assign_to_agent
       outputs:
         - analytics_logged
         - metrics_updated

Python Implementation
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # customer_support_automation.py
   import asyncio
   import os
   from typing import Dict, List, Any, Optional
   import json
   from datetime import datetime, timedelta
   from dataclasses import dataclass
   import logging
   
   from orchestrator import Orchestrator
   from orchestrator.tools.support_tools import (
       TicketingSystemTool,
       KnowledgeBaseTool,
       ResponseGeneratorTool,
       TranslationTool,
       SLAMonitorTool
   )
   from orchestrator.integrations.ticketing import (
       ZendeskIntegration,
       FreshdeskIntegration,
       JiraServiceDeskIntegration
   )
   from orchestrator.analytics import SupportAnalytics
   
   
   @dataclass
   class SupportTicket:
       """Represents a customer support ticket."""
       id: str
       subject: str
       description: str
       customer_id: str
       status: str
       priority: str
       category: Optional[str] = None
       tags: List[str] = None
       created_at: datetime = None
       updated_at: datetime = None
       
   
   class CustomerSupportAutomation:
       """
       Comprehensive customer support automation system.
       
       Features:
       - Multi-system ticketing integration
       - AI-powered ticket classification and response
       - Knowledge base integration
       - Intelligent routing and escalation
       - Multi-language support
       - Real-time analytics
       """
       
       def __init__(self, config: Dict[str, Any]):
           self.config = config
           self.orchestrator = None
           self.ticketing_system = None
           self.knowledge_base = None
           self.analytics = None
           self._setup_system()
       
       def _setup_system(self):
           """Initialize support automation components."""
           self.orchestrator = Orchestrator()
           
           # Register AI models
           self._register_models()
           
           # Initialize ticketing system
           self.ticketing_system = self._init_ticketing_system()
           
           # Initialize knowledge base
           self.knowledge_base = KnowledgeBaseTool(
               kb_url=self.config.get('kb_url'),
               api_key=self.config.get('kb_api_key')
           )
           
           # Initialize analytics
           self.analytics = SupportAnalytics(
               backend=self.config.get('analytics_backend', 'postgres')
           )
           
           # Initialize tools
           self.tools = {
               'ticketing': TicketingSystemTool(self.ticketing_system),
               'kb_search': self.knowledge_base,
               'response_generator': ResponseGeneratorTool(self.config),
               'translator': TranslationTool(
                   service=self.config.get('translation_service', 'google')
               ),
               'sla_monitor': SLAMonitorTool(self.config.get('sla_policies', {}))
           }
       
       def _init_ticketing_system(self):
           """Initialize the appropriate ticketing system."""
           system_type = self.config['ticketing_system']
           
           if system_type == 'zendesk':
               return ZendeskIntegration(
                   subdomain=self.config['zendesk_subdomain'],
                   email=self.config['zendesk_email'],
                   token=self.config['zendesk_token']
               )
           elif system_type == 'freshdesk':
               return FreshdeskIntegration(
                   domain=self.config['freshdesk_domain'],
                   api_key=self.config['freshdesk_api_key']
               )
           elif system_type == 'jira':
               return JiraServiceDeskIntegration(
                   server=self.config['jira_server'],
                   email=self.config['jira_email'],
                   token=self.config['jira_token']
               )
           else:
               raise ValueError(f"Unsupported ticketing system: {system_type}")
       
       async def process_ticket(
           self,
           ticket_id: str,
           auto_respond: bool = True
       ) -> Dict[str, Any]:
           """
           Process a support ticket through the automation pipeline.
           
           Args:
               ticket_id: ID of the ticket to process
               auto_respond: Whether to enable automated responses
               
           Returns:
               Processing results and actions taken
           """
           logging.info(f"🎫 Processing ticket: {ticket_id}")
           
           # Prepare context
           context = {
               'ticket_id': ticket_id,
               'ticketing_system': self.config['ticketing_system'],
               'auto_respond': auto_respond,
               'languages': self.config.get('languages', ['en']),
               'business_hours': self.config.get('business_hours', {}),
               'sla_policies': self.config.get('sla_policies', {})
           }
           
           # Execute pipeline
           try:
               results = await self.orchestrator.execute_pipeline(
                   'customer_support_automation.yaml',
                   context=context,
                   progress_callback=self._progress_callback
               )
               
               # Process results
               automation_report = await self._process_results(results)
               
               # Track analytics
               await self.analytics.track_ticket_processing(automation_report)
               
               return automation_report
               
           except Exception as e:
               logging.error(f"Error processing ticket {ticket_id}: {e}")
               return await self._handle_processing_error(ticket_id, e)
       
       async def _progress_callback(self, task_id: str, progress: float, message: str):
           """Handle progress updates."""
           icons = {
               'receive_ticket': '📥',
               'detect_language': '🌐',
               'analyze_sentiment': '😊',
               'extract_entities': '🔍',
               'classify_ticket': '🏷️',
               'search_knowledge_base': '📚',
               'check_automation_eligibility': '🤖',
               'generate_response': '✍️',
               'translate_response': '🔤',
               'update_ticket': '🔄',
               'send_response': '📤',
               'assign_to_agent': '👤',
               'monitor_sla': '⏰',
               'log_interaction': '📊'
           }
           icon = icons.get(task_id, '▶️')
           print(f"{icon} {task_id}: {progress:.0%} - {message}")
       
       async def _process_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
           """Process pipeline results into automation report."""
           report = {
               'ticket_id': results['context']['ticket_id'],
               'processing_time': datetime.now().isoformat(),
               'automation_status': 'unknown',
               'actions_taken': [],
               'metrics': {}
           }
           
           # Determine automation status
           if results.get('send_response', {}).get('response_sent'):
               report['automation_status'] = 'automated'
               report['actions_taken'].append('sent_automated_response')
               report['automated_response'] = results['generate_response']['response_text']
           elif results.get('assign_to_agent', {}).get('assigned_agent'):
               report['automation_status'] = 'escalated'
               report['actions_taken'].append('assigned_to_agent')
               report['assigned_agent'] = results['assign_to_agent']['assigned_agent']
           
           # Add classification info
           if 'classify_ticket' in results:
               report['classification'] = {
                   'category': results['classify_ticket']['primary_category'],
                   'tags': results['classify_ticket']['tags'],
                   'priority': results['classify_ticket']['priority_score']
               }
           
           # Add sentiment analysis
           if 'analyze_sentiment' in results:
               report['sentiment'] = {
                   'score': results['analyze_sentiment']['sentiment_score'],
                   'urgency': results['analyze_sentiment']['urgency_level'],
                   'frustration': results['analyze_sentiment']['frustration_detected']
               }
           
           # Add knowledge base results
           if 'search_knowledge_base' in results:
               report['kb_search'] = {
                   'articles_found': len(results['search_knowledge_base']['relevant_articles']),
                   'solution_found': results['search_knowledge_base']['solution_found'],
                   'confidence': results['search_knowledge_base']['confidence_score']
               }
           
           # Add SLA status
           if 'monitor_sla' in results:
               report['sla'] = {
                   'status': results['monitor_sla']['sla_status'],
                   'time_remaining': results['monitor_sla']['time_remaining']
               }
           
           return report
       
       async def batch_process_tickets(
           self,
           ticket_ids: List[str],
           parallel: bool = True
       ) -> Dict[str, Any]:
           """Process multiple tickets in batch."""
           if parallel:
               # Process tickets in parallel
               tasks = [
                   self.process_ticket(ticket_id)
                   for ticket_id in ticket_ids
               ]
               results = await asyncio.gather(*tasks, return_exceptions=True)
           else:
               # Process tickets sequentially
               results = []
               for ticket_id in ticket_ids:
                   result = await self.process_ticket(ticket_id)
                   results.append(result)
           
           # Compile batch report
           batch_report = {
               'total_tickets': len(ticket_ids),
               'processed': sum(1 for r in results if not isinstance(r, Exception)),
               'automated': sum(1 for r in results if not isinstance(r, Exception) and r.get('automation_status') == 'automated'),
               'escalated': sum(1 for r in results if not isinstance(r, Exception) and r.get('automation_status') == 'escalated'),
               'errors': sum(1 for r in results if isinstance(r, Exception)),
               'results': results
           }
           
           return batch_report

Advanced Features
^^^^^^^^^^^^^^^^^

.. code-block:: python

   class IntelligentTicketClassifier:
       """Advanced ticket classification with custom taxonomies."""
       
       def __init__(self, taxonomy_config: Dict[str, Any]):
           self.taxonomy = taxonomy_config
           self.classifier = None
           self._load_classifier()
       
       async def classify_ticket(
           self,
           ticket: SupportTicket,
           entities: Dict[str, Any]
       ) -> Dict[str, Any]:
           """Classify ticket using multi-level taxonomy."""
           # Extract features
           features = await self._extract_features(ticket, entities)
           
           # Primary classification
           primary_category = await self._classify_primary(features)
           
           # Secondary classification
           secondary_categories = await self._classify_secondary(
               features,
               primary_category
           )
           
           # Generate tags
           tags = await self._generate_tags(features, primary_category)
           
           # Calculate priority
           priority_score = await self._calculate_priority(
               ticket,
               primary_category,
               features
           )
           
           return {
               'primary_category': primary_category,
               'secondary_categories': secondary_categories,
               'tags': tags,
               'priority_score': priority_score,
               'confidence': await self._calculate_confidence(features)
           }
       
       async def _extract_features(
           self,
           ticket: SupportTicket,
           entities: Dict[str, Any]
       ) -> Dict[str, Any]:
           """Extract classification features from ticket."""
           return {
               'text_features': await self._extract_text_features(
                   ticket.subject + " " + ticket.description
               ),
               'entity_features': entities,
               'metadata_features': {
                   'channel': ticket.channel if hasattr(ticket, 'channel') else 'unknown',
                   'customer_history': ticket.customer_history if hasattr(ticket, 'customer_history') else None
               }
           }
       
       async def _calculate_priority(
           self,
           ticket: SupportTicket,
           category: str,
           features: Dict[str, Any]
       ) -> int:
           """Calculate ticket priority score (0-10)."""
           priority = 5  # Default medium priority
           
           # Category-based priority
           high_priority_categories = ['payment_issue', 'service_down', 'security']
           if category in high_priority_categories:
               priority += 3
           
           # Sentiment-based priority
           if features.get('sentiment_score', 0) < -0.5:
               priority += 2
           
           # Customer tier priority
           customer_tier = features.get('metadata_features', {}).get('customer_tier')
           if customer_tier in ['platinum', 'gold']:
               priority += 2
           
           # Urgency keywords
           urgency_keywords = ['urgent', 'asap', 'immediately', 'critical']
           if any(keyword in ticket.description.lower() for keyword in urgency_keywords):
               priority += 1
           
           return min(priority, 10)  # Cap at 10
   
   
   class KnowledgeBaseSearchEngine:
       """Advanced knowledge base search with semantic understanding."""
       
       def __init__(self, kb_config: Dict[str, Any]):
           self.config = kb_config
           self.embeddings_model = self._init_embeddings_model()
           self.vector_store = self._init_vector_store()
       
       async def search(
           self,
           query: str,
           category: Optional[str] = None,
           limit: int = 5
       ) -> Dict[str, Any]:
           """Search knowledge base with semantic matching."""
           # Generate query embedding
           query_embedding = await self.embeddings_model.embed(query)
           
           # Search vector store
           results = await self.vector_store.search(
               query_embedding,
               filter={'category': category} if category else None,
               limit=limit * 2  # Get more for re-ranking
           )
           
           # Re-rank results
           reranked = await self._rerank_results(query, results)
           
           # Check if solution found
           solution_found = any(r['score'] > 0.8 for r in reranked[:limit])
           
           return {
               'relevant_articles': reranked[:limit],
               'solution_found': solution_found,
               'confidence_score': reranked[0]['score'] if reranked else 0.0
           }
       
       async def _rerank_results(
           self,
           query: str,
           results: List[Dict[str, Any]]
       ) -> List[Dict[str, Any]]:
           """Re-rank search results for relevance."""
           # Use cross-encoder for re-ranking
           reranked = []
           
           for result in results:
               relevance_score = await self._calculate_relevance(
                   query,
                   result['content']
               )
               
               result['score'] = (result['score'] + relevance_score) / 2
               reranked.append(result)
           
           return sorted(reranked, key=lambda x: x['score'], reverse=True)
   
   
   class AutoResponseGenerator:
       """Generate high-quality automated responses."""
       
       def __init__(self, model_config: Dict[str, Any]):
           self.model_config = model_config
           self.response_templates = self._load_templates()
       
       async def generate_response(
           self,
           ticket: SupportTicket,
           kb_articles: List[Dict[str, Any]],
           customer_info: Dict[str, Any],
           language: str = 'en'
       ) -> Dict[str, Any]:
           """Generate personalized automated response."""
           # Select response strategy
           strategy = await self._select_strategy(ticket, kb_articles)
           
           if strategy == 'kb_based':
               response = await self._generate_kb_response(
                   ticket,
                   kb_articles,
                   customer_info
               )
           elif strategy == 'template_based':
               response = await self._generate_template_response(
                   ticket,
                   customer_info
               )
           else:
               response = await self._generate_custom_response(
                   ticket,
                   customer_info
               )
           
           # Personalize response
           response = await self._personalize_response(
               response,
               customer_info,
               language
           )
           
           # Add solution steps if applicable
           if kb_articles:
               response['solution_steps'] = await self._extract_solution_steps(
                   kb_articles[0]
               )
           
           return response
       
       async def _personalize_response(
           self,
           response: Dict[str, Any],
           customer_info: Dict[str, Any],
           language: str
       ) -> Dict[str, Any]:
           """Personalize response for customer."""
           # Add customer name
           if customer_info.get('name'):
               response['text'] = response['text'].replace(
                   '{customer_name}',
                   customer_info['name']
               )
           
           # Adjust tone based on customer history
           if customer_info.get('ticket_count', 0) > 10:
               # Valued customer - more personal tone
               response['text'] = await self._adjust_tone_for_valued_customer(
                   response['text']
               )
           
           # Add language-specific elements
           if language != 'en':
               response['cultural_adjustments'] = await self._apply_cultural_adjustments(
                   response['text'],
                   language
               )
           
           return response

SLA Monitoring and Escalation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class SLAManager:
       """Manage SLA compliance and escalations."""
       
       def __init__(self, sla_policies: Dict[str, Any]):
           self.policies = sla_policies
           self.escalation_handler = EscalationHandler()
       
       async def monitor_ticket_sla(
           self,
           ticket: SupportTicket,
           customer_tier: str,
           priority: int
       ) -> Dict[str, Any]:
           """Monitor SLA compliance for a ticket."""
           # Get applicable SLA policy
           policy = self._get_applicable_policy(customer_tier, priority)
           
           # Calculate time elapsed
           time_elapsed = datetime.now() - ticket.created_at
           
           # Check SLA targets
           sla_status = {
               'first_response': await self._check_first_response_sla(
                   ticket,
                   policy,
                   time_elapsed
               ),
               'resolution': await self._check_resolution_sla(
                   ticket,
                   policy,
                   time_elapsed
               ),
               'update_frequency': await self._check_update_frequency_sla(
                   ticket,
                   policy
               )
           }
           
           # Determine overall status
           overall_status = 'compliant'
           escalation_needed = False
           
           for target, status in sla_status.items():
               if status['status'] == 'breached':
                   overall_status = 'breached'
                   escalation_needed = True
                   break
               elif status['status'] == 'at_risk':
                   overall_status = 'at_risk'
           
           # Calculate time remaining for next target
           time_remaining = min(
               status['time_remaining']
               for status in sla_status.values()
               if status['time_remaining'] is not None
           )
           
           return {
               'sla_status': overall_status,
               'time_remaining': time_remaining,
               'escalation_needed': escalation_needed,
               'detailed_status': sla_status,
               'policy_applied': policy['name']
           }
       
       async def handle_escalation(
           self,
           ticket: SupportTicket,
           reason: str
       ) -> Dict[str, Any]:
           """Handle SLA escalation."""
           return await self.escalation_handler.escalate(
               ticket,
               reason,
               self.policies['escalation_paths']
           )
   
   
   class EscalationHandler:
       """Handle ticket escalations."""
       
       async def escalate(
           self,
           ticket: SupportTicket,
           reason: str,
           escalation_paths: Dict[str, Any]
       ) -> Dict[str, Any]:
           """Escalate ticket based on reason."""
           # Determine escalation path
           path = self._determine_escalation_path(reason, escalation_paths)
           
           # Get escalation target
           target = await self._get_escalation_target(path, ticket)
           
           # Perform escalation
           escalation_result = {
               'escalated_to': target['name'],
               'escalation_level': target['level'],
               'reason': reason,
               'notification_sent': await self._send_escalation_notification(
                   ticket,
                   target,
                   reason
               )
           }
           
           return escalation_result

Analytics and Reporting
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class SupportAnalyticsDashboard:
       """Real-time analytics dashboard for support metrics."""
       
       def __init__(self, analytics_backend: Any):
           self.backend = analytics_backend
           self.metrics_calculator = MetricsCalculator()
       
       async def get_dashboard_data(
           self,
           timeframe: timedelta = timedelta(days=1)
       ) -> Dict[str, Any]:
           """Get comprehensive dashboard data."""
           end_time = datetime.now()
           start_time = end_time - timeframe
           
           # Get raw metrics
           raw_metrics = await self.backend.query_metrics(
               start_time,
               end_time
           )
           
           # Calculate dashboard metrics
           dashboard = {
               'overview': await self._calculate_overview(raw_metrics),
               'automation_metrics': await self._calculate_automation_metrics(raw_metrics),
               'category_breakdown': await self._analyze_by_category(raw_metrics),
               'sentiment_trends': await self._analyze_sentiment_trends(raw_metrics),
               'sla_compliance': await self._calculate_sla_compliance(raw_metrics),
               'agent_performance': await self._analyze_agent_performance(raw_metrics),
               'kb_effectiveness': await self._analyze_kb_effectiveness(raw_metrics)
           }
           
           return dashboard
       
       async def _calculate_automation_metrics(
           self,
           metrics: List[Dict[str, Any]]
       ) -> Dict[str, Any]:
           """Calculate automation-specific metrics."""
           total_tickets = len(metrics)
           automated_tickets = sum(
               1 for m in metrics
               if m.get('automation_status') == 'automated'
           )
           
           automation_rate = automated_tickets / total_tickets if total_tickets > 0 else 0
           
           # Calculate automation success metrics
           successful_automations = sum(
               1 for m in metrics
               if m.get('automation_status') == 'automated' and
               m.get('customer_satisfaction', 0) >= 4
           )
           
           automation_success_rate = (
               successful_automations / automated_tickets
               if automated_tickets > 0 else 0
           )
           
           return {
               'automation_rate': automation_rate,
               'automation_success_rate': automation_success_rate,
               'total_automated': automated_tickets,
               'time_saved': await self._calculate_time_saved(metrics),
               'cost_savings': await self._calculate_cost_savings(metrics)
           }
       
       async def generate_weekly_report(self) -> Dict[str, Any]:
           """Generate comprehensive weekly report."""
           # Get data for the past week
           dashboard_data = await self.get_dashboard_data(
               timeframe=timedelta(days=7)
           )
           
           # Add trends and comparisons
           report = {
               'period': {
                   'start': (datetime.now() - timedelta(days=7)).isoformat(),
                   'end': datetime.now().isoformat()
               },
               'metrics': dashboard_data,
               'trends': await self._calculate_weekly_trends(),
               'insights': await self._generate_insights(dashboard_data),
               'recommendations': await self._generate_recommendations(dashboard_data)
           }
           
           return report

Running the System
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # main.py
   import asyncio
   import argparse
   from customer_support_automation import CustomerSupportAutomation
   
   async def main():
       parser = argparse.ArgumentParser(description='Customer Support Automation')
       parser.add_argument('--ticketing-system', 
                          choices=['zendesk', 'freshdesk', 'jira'],
                          default='zendesk')
       parser.add_argument('--auto-respond', action='store_true',
                          help='Enable automated responses')
       parser.add_argument('--languages', nargs='+', default=['en'],
                          help='Supported languages')
       parser.add_argument('--mode', choices=['process', 'monitor', 'report'],
                          default='process')
       parser.add_argument('--ticket-id', help='Specific ticket to process')
       parser.add_argument('--batch-size', type=int, default=10,
                          help='Batch size for processing')
       
       args = parser.parse_args()
       
       # Configuration
       config = {
           'ticketing_system': args.ticketing_system,
           'languages': args.languages,
           'openai_api_key': os.getenv('OPENAI_API_KEY'),
           'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
           
           # Ticketing system credentials
           'zendesk_subdomain': os.getenv('ZENDESK_SUBDOMAIN'),
           'zendesk_email': os.getenv('ZENDESK_EMAIL'),
           'zendesk_token': os.getenv('ZENDESK_TOKEN'),
           
           # Knowledge base
           'kb_url': os.getenv('KB_URL', 'https://kb.example.com'),
           'kb_api_key': os.getenv('KB_API_KEY'),
           
           # SLA policies
           'sla_policies': {
               'first_response': {
                   'urgent': timedelta(hours=1),
                   'high': timedelta(hours=4),
                   'normal': timedelta(hours=24),
                   'low': timedelta(days=2)
               },
               'resolution': {
                   'urgent': timedelta(hours=4),
                   'high': timedelta(days=1),
                   'normal': timedelta(days=3),
                   'low': timedelta(days=7)
               }
           },
           
           # Analytics
           'analytics_backend': 'postgres',
           'analytics_db_url': os.getenv('DATABASE_URL')
       }
       
       # Create automation system
       automation = CustomerSupportAutomation(config)
       
       if args.mode == 'process':
           if args.ticket_id:
               # Process single ticket
               result = await automation.process_ticket(
                   args.ticket_id,
                   auto_respond=args.auto_respond
               )
               
               print(f"\n✅ Ticket {args.ticket_id} processed!")
               print(f"Status: {result['automation_status']}")
               print(f"Actions: {', '.join(result['actions_taken'])}")
               
               if result.get('classification'):
                   print(f"Category: {result['classification']['category']}")
                   print(f"Priority: {result['classification']['priority']}")
               
           else:
               # Batch process tickets
               print(f"🔄 Starting batch processing (size: {args.batch_size})")
               
               while True:
                   # Get new tickets
                   new_tickets = await automation.ticketing_system.get_new_tickets(
                       limit=args.batch_size
                   )
                   
                   if not new_tickets:
                       print("No new tickets to process")
                       await asyncio.sleep(60)
                       continue
                   
                   # Process batch
                   batch_result = await automation.batch_process_tickets(
                       [t['id'] for t in new_tickets],
                       parallel=True
                   )
                   
                   print(f"\n📊 Batch Results:")
                   print(f"Processed: {batch_result['processed']}/{batch_result['total_tickets']}")
                   print(f"Automated: {batch_result['automated']}")
                   print(f"Escalated: {batch_result['escalated']}")
                   
                   await asyncio.sleep(30)  # Wait before next batch
       
       elif args.mode == 'monitor':
           # Real-time monitoring mode
           print("📊 Starting real-time monitoring dashboard")
           
           dashboard = SupportAnalyticsDashboard(automation.analytics)
           
           while True:
               data = await dashboard.get_dashboard_data(timedelta(hours=1))
               
               print("\n" + "="*50)
               print(f"Support Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
               print("="*50)
               
               print(f"\n📈 Overview:")
               print(f"Total Tickets: {data['overview']['total_tickets']}")
               print(f"Avg Response Time: {data['overview']['avg_response_time']:.2f} min")
               print(f"Customer Satisfaction: {data['overview']['csat_score']:.2f}/5")
               
               print(f"\n🤖 Automation:")
               print(f"Automation Rate: {data['automation_metrics']['automation_rate']:.1%}")
               print(f"Success Rate: {data['automation_metrics']['automation_success_rate']:.1%}")
               print(f"Time Saved: {data['automation_metrics']['time_saved']:.1f} hours")
               
               print(f"\n⏱️ SLA Compliance:")
               print(f"Overall: {data['sla_compliance']['overall_compliance']:.1%}")
               print(f"At Risk: {data['sla_compliance']['tickets_at_risk']}")
               
               await asyncio.sleep(60)  # Update every minute
       
       elif args.mode == 'report':
           # Generate report
           print("📄 Generating weekly report...")
           
           dashboard = SupportAnalyticsDashboard(automation.analytics)
           report = await dashboard.generate_weekly_report()
           
           # Save report
           report_file = f"support_report_{datetime.now().strftime('%Y%m%d')}.json"
           with open(report_file, 'w') as f:
               json.dump(report, f, indent=2, default=str)
           
           print(f"✅ Report saved to: {report_file}")
           
           # Print summary
           print("\n📊 Weekly Summary:")
           print(f"Total Tickets: {report['metrics']['overview']['total_tickets']}")
           print(f"Automation Rate: {report['metrics']['automation_metrics']['automation_rate']:.1%}")
           print(f"Cost Savings: ${report['metrics']['automation_metrics']['cost_savings']:,.2f}")
           
           print("\n💡 Top Insights:")
           for i, insight in enumerate(report['insights'][:5], 1):
               print(f"{i}. {insight}")
   
   if __name__ == "__main__":
       asyncio.run(main())

Integration Examples
--------------------

Zendesk Integration
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class ZendeskAutomation:
       """Zendesk-specific automation features."""
       
       async def setup_zendesk_triggers(self):
           """Setup automated triggers in Zendesk."""
           triggers = [
               {
                   'title': 'Auto-tag high priority tickets',
                   'conditions': {
                       'all': [
                           {'field': 'status', 'operator': 'is', 'value': 'new'},
                           {'field': 'description', 'operator': 'contains', 'value': 'urgent'}
                       ]
                   },
                   'actions': [
                       {'field': 'priority', 'value': 'high'},
                       {'field': 'tags', 'value': ['urgent', 'auto-tagged']}
                   ]
               },
               {
                   'title': 'Route payment issues to finance team',
                   'conditions': {
                       'any': [
                           {'field': 'subject', 'operator': 'contains', 'value': 'payment'},
                           {'field': 'subject', 'operator': 'contains', 'value': 'billing'}
                       ]
                   },
                   'actions': [
                       {'field': 'group_id', 'value': 'finance_team_id'},
                       {'field': 'tags', 'value': ['payment-issue']}
                   ]
               }
           ]
           
           for trigger in triggers:
               await self.create_trigger(trigger)

Best Practices
--------------

1. **Response Quality**: Always review and test automated responses
2. **Escalation Paths**: Define clear escalation criteria and paths
3. **Knowledge Base**: Keep KB articles updated and well-organized
4. **Customer Context**: Always consider customer history and preferences
5. **Monitoring**: Track automation metrics and customer satisfaction
6. **Compliance**: Ensure responses meet regulatory requirements
7. **Continuous Improvement**: Regular review and optimization of automation rules

Summary
-------

The Customer Support Automation system demonstrates:

- Comprehensive ticket processing automation
- Intelligent classification and routing
- Multi-language support capabilities
- Seamless integration with popular ticketing systems
- Advanced analytics and reporting
- SLA monitoring and compliance

This system provides a foundation for building efficient, scalable customer support operations that maintain high quality while reducing costs.