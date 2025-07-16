Interactive Chat Bot
====================

This example demonstrates how to build a sophisticated interactive chat bot with context awareness, memory management, multi-turn conversations, and integration with various tools and APIs. The bot can handle complex queries, maintain conversation history, and provide intelligent responses.

.. note::
   **Level:** Intermediate  
   **Duration:** 45-60 minutes  
   **Prerequisites:** Basic Python knowledge, understanding of conversational AI concepts, familiarity with async programming

Overview
--------

The Interactive Chat Bot provides:

1. **Context-Aware Conversations**: Maintains conversation context across multiple turns
2. **Memory Management**: Short-term and long-term memory systems
3. **Tool Integration**: Access to web search, calculations, and external APIs
4. **Personalization**: Adapts to user preferences and conversation style
5. **Multi-Modal Support**: Handle text, images, and file uploads
6. **Conversation Analytics**: Track and analyze conversation patterns
7. **Safety Controls**: Content filtering and response validation

**Key Features:**
- Natural language understanding and generation
- Conversation state management
- Dynamic tool selection and usage
- User profile and preference tracking
- Conversation branching and threading
- Real-time response streaming
- Multi-language support

Quick Start
-----------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-org/orchestrator.git
   cd orchestrator
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Set up environment variables
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   
   # Run the example
   python examples/interactive_chat_bot.py \
     --mode interactive \
     --persona helpful-assistant

Complete Implementation
-----------------------

Pipeline Configuration (YAML)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # interactive_chat_pipeline.yaml
   id: interactive_chat_bot
   name: Context-Aware Interactive Chat Bot
   version: "1.0"
   
   metadata:
     description: "Sophisticated conversational AI with memory and tool integration"
     author: "AI Team"
     tags: ["chatbot", "conversational-ai", "interactive", "tools"]
   
   models:
     chat_model:
       provider: "openai"
       model: "gpt-4"
       temperature: 0.7
     intent_classifier:
       provider: "anthropic"
       model: "claude-3-opus"
       temperature: 0.3
     safety_checker:
       provider: "openai"
       model: "gpt-3.5-turbo"
       temperature: 0.1
   
   context:
     persona: "{{ inputs.persona }}"
     conversation_style: "{{ inputs.style }}"
     available_tools: ["web_search", "calculator", "weather", "knowledge_base"]
     safety_level: "{{ inputs.safety_level }}"
   
   tasks:
     - id: process_input
       name: "Process User Input"
       action: "analyze_user_input"
       parameters:
         message: "{{ inputs.message }}"
         conversation_history: "{{ state.conversation_history }}"
         user_profile: "{{ state.user_profile }}"
         detect_language: true
       outputs:
         - processed_input
         - detected_language
         - input_metadata
     
     - id: safety_check
       name: "Safety and Content Check"
       action: "check_content_safety"
       model: "safety_checker"
       parameters:
         content: "{{ process_input.processed_input }}"
         safety_level: "{{ context.safety_level }}"
         check_types: ["toxicity", "pii", "inappropriate"]
       dependencies:
         - process_input
       outputs:
         - is_safe
         - safety_issues
         - filtered_content
     
     - id: retrieve_context
       name: "Retrieve Conversation Context"
       action: "get_conversation_context"
       parameters:
         conversation_id: "{{ inputs.conversation_id }}"
         history_depth: <AUTO>Determine based on conversation complexity</AUTO>
         include_user_preferences: true
       dependencies:
         - safety_check
       outputs:
         - conversation_context
         - relevant_history
         - user_preferences
     
     - id: classify_intent
       name: "Classify User Intent"
       action: "classify_intent"
       model: "intent_classifier"
       parameters:
         message: "{{ safety_check.filtered_content }}"
         context: "{{ retrieve_context.conversation_context }}"
         possible_intents: <AUTO>Determine from conversation pattern</AUTO>
       dependencies:
         - retrieve_context
       outputs:
         - primary_intent
         - secondary_intents
         - confidence_scores
     
     - id: determine_tools
       name: "Determine Required Tools"
       action: "select_tools"
       parameters:
         intent: "{{ classify_intent.primary_intent }}"
         available_tools: "{{ context.available_tools }}"
         user_permissions: "{{ retrieve_context.user_preferences.tool_permissions }}"
       dependencies:
         - classify_intent
       outputs:
         - required_tools
         - tool_parameters
     
     - id: execute_tools
       name: "Execute Selected Tools"
       action: "run_tools"
       parallel: true
       for_each: "{{ determine_tools.required_tools }}"
       condition: "determine_tools.required_tools | length > 0"
       parameters:
         tool: "{{ item }}"
         parameters: "{{ determine_tools.tool_parameters[item] }}"
         timeout: 10
       dependencies:
         - determine_tools
       outputs:
         - tool_results
         - execution_status
     
     - id: generate_response
       name: "Generate Conversational Response"
       action: "generate_chat_response"
       model: "chat_model"
       parameters:
         user_message: "{{ safety_check.filtered_content }}"
         conversation_context: "{{ retrieve_context.conversation_context }}"
         tool_results: "{{ execute_tools.tool_results }}"
         persona: "{{ context.persona }}"
         style: <AUTO>Adapt based on conversation history</AUTO>
         max_tokens: 500
       dependencies:
         - execute_tools
         - classify_intent
       outputs:
         - response_text
         - response_metadata
         - suggested_followups
     
     - id: enhance_response
       name: "Enhance Response Quality"
       action: "enhance_response"
       parameters:
         response: "{{ generate_response.response_text }}"
         user_language: "{{ process_input.detected_language }}"
         add_formatting: true
         check_factuality: true
       dependencies:
         - generate_response
       outputs:
         - enhanced_response
         - formatting_applied
         - fact_check_results
     
     - id: update_memory
       name: "Update Conversation Memory"
       action: "update_memory_systems"
       parameters:
         conversation_id: "{{ inputs.conversation_id }}"
         user_message: "{{ inputs.message }}"
         bot_response: "{{ enhance_response.enhanced_response }}"
         intent: "{{ classify_intent.primary_intent }}"
         important_entities: <AUTO>Extract from conversation</AUTO>
       dependencies:
         - enhance_response
       outputs:
         - memory_updated
         - extracted_facts
         - conversation_summary
     
     - id: prepare_output
       name: "Prepare Final Output"
       action: "format_chat_output"
       parameters:
         response: "{{ enhance_response.enhanced_response }}"
         metadata: "{{ generate_response.response_metadata }}"
         suggested_actions: "{{ generate_response.suggested_followups }}"
         streaming: "{{ inputs.enable_streaming }}"
       dependencies:
         - update_memory
       outputs:
         - final_response
         - conversation_state
         - next_turn_context

Python Implementation
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # interactive_chat_bot.py
   import asyncio
   import os
   from typing import Dict, List, Any, Optional, AsyncGenerator
   import json
   from datetime import datetime
   import uuid
   from dataclasses import dataclass, field
   
   from orchestrator import Orchestrator
   from orchestrator.tools.chat_tools import (
       ConversationManager,
       MemorySystem,
       IntentClassifier,
       ResponseGenerator
   )
   from orchestrator.tools.integration_tools import (
       WebSearchTool,
       CalculatorTool,
       WeatherTool,
       KnowledgeBaseTool
   )
   from orchestrator.streaming import StreamingResponse
   
   
   @dataclass
   class ConversationState:
       """Maintains conversation state."""
       conversation_id: str
       user_id: Optional[str] = None
       messages: List[Dict[str, Any]] = field(default_factory=list)
       context: Dict[str, Any] = field(default_factory=dict)
       user_profile: Dict[str, Any] = field(default_factory=dict)
       active_tools: List[str] = field(default_factory=list)
       created_at: datetime = field(default_factory=datetime.now)
       last_activity: datetime = field(default_factory=datetime.now)
   
   
   class InteractiveChatBot:
       """
       Sophisticated interactive chat bot with context awareness and tool integration.
       
       Features:
       - Multi-turn conversation handling
       - Context-aware responses
       - Dynamic tool selection and execution
       - Memory management
       - Personalization
       """
       
       def __init__(self, config: Dict[str, Any]):
           self.config = config
           self.orchestrator = None
           self.conversation_manager = None
           self.memory_system = None
           self.conversations = {}
           self._setup_bot()
       
       def _setup_bot(self):
           """Initialize chat bot components."""
           self.orchestrator = Orchestrator()
           
           # Register AI models
           self._register_models()
           
           # Initialize conversation manager
           self.conversation_manager = ConversationManager(
               max_history=self.config.get('max_history', 50)
           )
           
           # Initialize memory system
           self.memory_system = MemorySystem(
               short_term_capacity=100,
               long_term_storage=self.config.get('memory_storage', 'redis')
           )
           
           # Initialize tools
           self.tools = {
               'web_search': WebSearchTool(),
               'calculator': CalculatorTool(),
               'weather': WeatherTool(api_key=self.config.get('weather_api_key')),
               'knowledge_base': KnowledgeBaseTool()
           }
           
           # Initialize components
           self.intent_classifier = IntentClassifier()
           self.response_generator = ResponseGenerator(self.config)
       
       async def start_conversation(
           self,
           user_id: Optional[str] = None,
           persona: str = 'helpful-assistant',
           **kwargs
       ) -> str:
           """
           Start a new conversation.
           
           Args:
               user_id: Optional user identifier
               persona: Bot persona to use
               
           Returns:
               Conversation ID
           """
           conversation_id = str(uuid.uuid4())
           
           # Create conversation state
           state = ConversationState(
               conversation_id=conversation_id,
               user_id=user_id,
               context={'persona': persona, **kwargs}
           )
           
           # Load user profile if available
           if user_id:
               state.user_profile = await self._load_user_profile(user_id)
           
           self.conversations[conversation_id] = state
           
           # Initialize conversation in memory
           await self.memory_system.initialize_conversation(conversation_id)
           
           print(f"💬 New conversation started: {conversation_id}")
           return conversation_id
       
       async def chat(
           self,
           conversation_id: str,
           message: str,
           streaming: bool = False,
           **kwargs
       ) -> Union[Dict[str, Any], AsyncGenerator]:
           """
           Process a chat message.
           
           Args:
               conversation_id: Conversation identifier
               message: User message
               streaming: Enable response streaming
               
           Returns:
               Bot response or async generator for streaming
           """
           if conversation_id not in self.conversations:
               raise ValueError(f"Conversation {conversation_id} not found")
           
           state = self.conversations[conversation_id]
           state.last_activity = datetime.now()
           
           # Add user message to history
           state.messages.append({
               'role': 'user',
               'content': message,
               'timestamp': datetime.now().isoformat()
           })
           
           # Prepare context
           context = {
               'conversation_id': conversation_id,
               'message': message,
               'persona': state.context.get('persona', 'helpful-assistant'),
               'enable_streaming': streaming,
               **kwargs
           }
           
           # Execute pipeline
           try:
               if streaming:
                   return self._stream_response(context, state)
               else:
                   results = await self.orchestrator.execute_pipeline(
                       'interactive_chat_pipeline.yaml',
                       context=context,
                       state=state.__dict__
                   )
                   
                   # Process results
                   response = await self._process_chat_results(results, state)
                   
                   # Add to conversation history
                   state.messages.append({
                       'role': 'assistant',
                       'content': response['response'],
                       'timestamp': datetime.now().isoformat(),
                       'metadata': response.get('metadata', {})
                   })
                   
                   return response
                   
           except Exception as e:
               print(f"❌ Chat processing failed: {str(e)}")
               raise
       
       async def _stream_response(
           self,
           context: Dict[str, Any],
           state: ConversationState
       ) -> AsyncGenerator:
           """Stream response tokens as they're generated."""
           response_buffer = ""
           
           async for chunk in self.orchestrator.stream_pipeline(
               'interactive_chat_pipeline.yaml',
               context=context,
               state=state.__dict__
           ):
               if chunk.get('token'):
                   response_buffer += chunk['token']
                   yield {
                       'type': 'token',
                       'content': chunk['token']
                   }
               
               elif chunk.get('tool_execution'):
                   yield {
                       'type': 'tool',
                       'tool': chunk['tool_execution']['tool'],
                       'status': chunk['tool_execution']['status']
                   }
               
               elif chunk.get('complete'):
                   # Add complete response to history
                   state.messages.append({
                       'role': 'assistant',
                       'content': response_buffer,
                       'timestamp': datetime.now().isoformat()
                   })
                   
                   yield {
                       'type': 'complete',
                       'response': response_buffer,
                       'metadata': chunk.get('metadata', {})
                   }
       
       async def _process_chat_results(
           self,
           results: Dict[str, Any],
           state: ConversationState
       ) -> Dict[str, Any]:
           """Process pipeline results into chat response."""
           response = {
               'response': results.get('prepare_output', {}).get('final_response', ''),
               'conversation_id': state.conversation_id,
               'timestamp': datetime.now().isoformat()
           }
           
           # Add metadata
           if 'generate_response' in results:
               response['metadata'] = {
                   'intent': results.get('classify_intent', {}).get('primary_intent'),
                   'confidence': results.get('classify_intent', {}).get('confidence_scores', {}),
                   'tools_used': results.get('determine_tools', {}).get('required_tools', []),
                   'suggested_followups': results.get('generate_response', {}).get('suggested_followups', [])
               }
           
           # Update conversation state
           if 'update_memory' in results:
               memory_data = results['update_memory']
               state.context['extracted_facts'] = memory_data.get('extracted_facts', [])
               state.context['conversation_summary'] = memory_data.get('conversation_summary', '')
           
           return response
       
       async def _load_user_profile(self, user_id: str) -> Dict[str, Any]:
           """Load user profile and preferences."""
           # In real implementation, load from database
           return {
               'user_id': user_id,
               'preferences': {
                   'language': 'en',
                   'formality': 'casual',
                   'expertise_level': 'intermediate',
                   'tool_permissions': ['web_search', 'calculator', 'weather']
               },
               'history_summary': await self.memory_system.get_user_history_summary(user_id)
           }
       
       async def get_conversation_history(
           self,
           conversation_id: str,
           limit: Optional[int] = None
       ) -> List[Dict[str, Any]]:
           """Get conversation history."""
           if conversation_id not in self.conversations:
               raise ValueError(f"Conversation {conversation_id} not found")
           
           messages = self.conversations[conversation_id].messages
           
           if limit:
               return messages[-limit:]
           return messages
       
       async def end_conversation(self, conversation_id: str) -> Dict[str, Any]:
           """End a conversation and save to long-term memory."""
           if conversation_id not in self.conversations:
               raise ValueError(f"Conversation {conversation_id} not found")
           
           state = self.conversations[conversation_id]
           
           # Generate conversation summary
           summary = await self._generate_conversation_summary(state)
           
           # Save to long-term memory
           await self.memory_system.save_conversation(
               conversation_id=conversation_id,
               user_id=state.user_id,
               messages=state.messages,
               summary=summary,
               metadata=state.context
           )
           
           # Clean up
           del self.conversations[conversation_id]
           
           return {
               'conversation_id': conversation_id,
               'summary': summary,
               'message_count': len(state.messages),
               'duration': (datetime.now() - state.created_at).total_seconds()
           }

Advanced Features
^^^^^^^^^^^^^^^^^

.. code-block:: python

   class ContextManager:
       """Manage conversation context and state."""
       
       def __init__(self, window_size: int = 10):
           self.window_size = window_size
           self.entity_tracker = EntityTracker()
           self.topic_tracker = TopicTracker()
       
       async def build_context(
           self,
           messages: List[Dict[str, Any]],
           user_profile: Dict[str, Any]
       ) -> Dict[str, Any]:
           """Build comprehensive conversation context."""
           # Get recent messages
           recent_messages = messages[-self.window_size:]
           
           # Extract entities
           entities = await self.entity_tracker.extract_entities(recent_messages)
           
           # Track topics
           topics = await self.topic_tracker.track_topics(recent_messages)
           
           # Build context
           context = {
               'recent_history': recent_messages,
               'entities': entities,
               'current_topic': topics['current'],
               'topic_history': topics['history'],
               'user_preferences': user_profile.get('preferences', {}),
               'conversation_style': self._analyze_conversation_style(recent_messages)
           }
           
           return context
       
       def _analyze_conversation_style(
           self,
           messages: List[Dict[str, Any]]
       ) -> Dict[str, Any]:
           """Analyze conversation style and patterns."""
           user_messages = [m for m in messages if m['role'] == 'user']
           
           return {
               'avg_message_length': sum(len(m['content']) for m in user_messages) / len(user_messages) if user_messages else 0,
               'formality_level': self._detect_formality(user_messages),
               'technical_level': self._detect_technical_level(user_messages),
               'emotional_tone': self._detect_emotional_tone(user_messages)
           }
   
   
   class MemorySystem:
       """Advanced memory management for conversations."""
       
       def __init__(self, short_term_capacity: int, long_term_storage: str):
           self.short_term_capacity = short_term_capacity
           self.short_term_memory = {}
           self.long_term_storage = self._init_storage(long_term_storage)
           
       async def remember_fact(
           self,
           conversation_id: str,
           fact: Dict[str, Any]
       ):
           """Store a fact in memory."""
           if conversation_id not in self.short_term_memory:
               self.short_term_memory[conversation_id] = []
           
           # Add to short-term memory
           self.short_term_memory[conversation_id].append({
               'fact': fact,
               'timestamp': datetime.now(),
               'importance': await self._calculate_importance(fact)
           })
           
           # Manage capacity
           if len(self.short_term_memory[conversation_id]) > self.short_term_capacity:
               await self._consolidate_memory(conversation_id)
       
       async def recall_facts(
           self,
           conversation_id: str,
           query: str
       ) -> List[Dict[str, Any]]:
           """Recall relevant facts from memory."""
           # Search short-term memory
           short_term_facts = self._search_short_term(conversation_id, query)
           
           # Search long-term memory
           long_term_facts = await self._search_long_term(conversation_id, query)
           
           # Combine and rank by relevance
           all_facts = short_term_facts + long_term_facts
           return sorted(all_facts, key=lambda x: x['relevance'], reverse=True)[:10]
       
       async def _consolidate_memory(self, conversation_id: str):
           """Consolidate short-term memory to long-term storage."""
           memories = self.short_term_memory[conversation_id]
           
           # Identify important memories to keep
           important_memories = sorted(
               memories,
               key=lambda x: x['importance'],
               reverse=True
           )[:self.short_term_capacity // 2]
           
           # Move less important memories to long-term
           for memory in memories:
               if memory not in important_memories:
                   await self.long_term_storage.store(
                       conversation_id,
                       memory
                   )
           
           # Keep only important memories in short-term
           self.short_term_memory[conversation_id] = important_memories
   
   
   class ResponsePersonalizer:
       """Personalize responses based on user profile and preferences."""
       
       async def personalize_response(
           self,
           response: str,
           user_profile: Dict[str, Any],
           conversation_style: Dict[str, Any]
       ) -> str:
           """Personalize response for the user."""
           # Adjust formality
           if user_profile.get('preferences', {}).get('formality') == 'formal':
               response = await self._make_formal(response)
           elif user_profile.get('preferences', {}).get('formality') == 'casual':
               response = await self._make_casual(response)
           
           # Adjust technical level
           expertise = user_profile.get('preferences', {}).get('expertise_level', 'intermediate')
           response = await self._adjust_technical_level(response, expertise)
           
           # Add personality touches
           if conversation_style.get('emotional_tone') == 'friendly':
               response = await self._add_friendly_touches(response)
           
           return response
       
       async def _make_formal(self, text: str) -> str:
           """Make response more formal."""
           # Implementation for formal language adjustment
           return text
       
       async def _make_casual(self, text: str) -> str:
           """Make response more casual."""
           # Implementation for casual language adjustment
           return text

Tool Integration
^^^^^^^^^^^^^^^^

.. code-block:: python

   class ToolOrchestrator:
       """Orchestrate tool selection and execution."""
       
       def __init__(self, available_tools: Dict[str, Any]):
           self.tools = available_tools
           self.tool_selector = ToolSelector()
           
       async def select_and_execute_tools(
           self,
           intent: str,
           message: str,
           context: Dict[str, Any]
       ) -> Dict[str, Any]:
           """Select and execute appropriate tools."""
           # Select tools based on intent
           selected_tools = await self.tool_selector.select_tools(
               intent,
               message,
               list(self.tools.keys())
           )
           
           # Execute tools in parallel when possible
           results = {}
           
           if self._can_parallelize(selected_tools):
               tasks = []
               for tool_name in selected_tools:
                   tool = self.tools[tool_name]
                   params = await self._prepare_tool_params(tool_name, message, context)
                   tasks.append(self._execute_tool(tool, params))
               
               tool_results = await asyncio.gather(*tasks)
               
               for tool_name, result in zip(selected_tools, tool_results):
                   results[tool_name] = result
           else:
               # Execute sequentially
               for tool_name in selected_tools:
                   tool = self.tools[tool_name]
                   params = await self._prepare_tool_params(tool_name, message, context)
                   results[tool_name] = await self._execute_tool(tool, params)
           
           return results
       
       async def _execute_tool(self, tool: Any, params: Dict[str, Any]) -> Dict[str, Any]:
           """Execute a single tool."""
           try:
               result = await tool.execute(**params)
               return {
                   'status': 'success',
                   'result': result
               }
           except Exception as e:
               return {
                   'status': 'error',
                   'error': str(e)
               }
       
       def _can_parallelize(self, tools: List[str]) -> bool:
           """Check if tools can be executed in parallel."""
           # Some tools might depend on others
           dependencies = {
               'calculator': [],
               'web_search': [],
               'weather': [],
               'knowledge_base': ['web_search']  # Might need web results
           }
           
           for tool in tools:
               deps = dependencies.get(tool, [])
               if any(dep in tools for dep in deps):
                   return False
           
           return True

Interactive CLI Interface
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class InteractiveCLI:
       """Command-line interface for the chat bot."""
       
       def __init__(self, bot: InteractiveChatBot):
           self.bot = bot
           self.conversation_id = None
           self.commands = {
               '/help': self.show_help,
               '/new': self.new_conversation,
               '/history': self.show_history,
               '/clear': self.clear_screen,
               '/persona': self.change_persona,
               '/tools': self.list_tools,
               '/save': self.save_conversation,
               '/load': self.load_conversation,
               '/exit': self.exit_chat
           }
       
       async def start(self):
           """Start interactive chat session."""
           print("🤖 Interactive Chat Bot Started!")
           print("Type /help for commands or start chatting\n")
           
           # Start default conversation
           self.conversation_id = await self.bot.start_conversation()
           
           while True:
               try:
                   user_input = input("You: ").strip()
                   
                   if user_input.startswith('/'):
                       # Handle command
                       await self._handle_command(user_input)
                   elif user_input.lower() in ['exit', 'quit', 'bye']:
                       await self.exit_chat()
                       break
                   else:
                       # Process chat message
                       await self._process_message(user_input)
                       
               except KeyboardInterrupt:
                   print("\n\nGoodbye! 👋")
                   break
               except Exception as e:
                   print(f"Error: {e}")
       
       async def _process_message(self, message: str):
           """Process and display chat message."""
           print("Bot: ", end="", flush=True)
           
           # Stream response
           async for chunk in self.bot.chat(
               self.conversation_id,
               message,
               streaming=True
           ):
               if chunk['type'] == 'token':
                   print(chunk['content'], end="", flush=True)
               elif chunk['type'] == 'tool':
                   print(f"\n[Using {chunk['tool']}...]", end="", flush=True)
               elif chunk['type'] == 'complete':
                   print("\n")
                   
                   # Show suggested follow-ups if available
                   if chunk.get('metadata', {}).get('suggested_followups'):
                       print("\n💡 Suggested questions:")
                       for i, followup in enumerate(chunk['metadata']['suggested_followups'][:3], 1):
                           print(f"   {i}. {followup}")
       
       async def _handle_command(self, command: str):
           """Handle slash commands."""
           parts = command.split(maxsplit=1)
           cmd = parts[0]
           args = parts[1] if len(parts) > 1 else None
           
           if cmd in self.commands:
               await self.commands[cmd](args)
           else:
               print(f"Unknown command: {cmd}")
               await self.show_help()
       
       async def show_help(self, args=None):
           """Show available commands."""
           print("\n📚 Available Commands:")
           print("  /help     - Show this help message")
           print("  /new      - Start a new conversation")
           print("  /history  - Show conversation history")
           print("  /clear    - Clear the screen")
           print("  /persona  - Change bot persona")
           print("  /tools    - List available tools")
           print("  /save     - Save conversation")
           print("  /load     - Load conversation")
           print("  /exit     - Exit the chat\n")

Running the Chat Bot
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # main.py
   import asyncio
   import argparse
   from interactive_chat_bot import InteractiveChatBot
   from interactive_cli import InteractiveCLI
   
   async def main():
       parser = argparse.ArgumentParser(description='Interactive Chat Bot')
       parser.add_argument('--mode', default='interactive',
                          choices=['interactive', 'api', 'test'])
       parser.add_argument('--persona', default='helpful-assistant',
                          choices=['helpful-assistant', 'technical-expert', 
                                  'creative-writer', 'tutor', 'friend'])
       parser.add_argument('--safety-level', default='moderate',
                          choices=['strict', 'moderate', 'relaxed'])
       parser.add_argument('--enable-voice', action='store_true',
                          help='Enable voice input/output')
       parser.add_argument('--language', default='en',
                          help='Primary language')
       
       args = parser.parse_args()
       
       # Configuration
       config = {
           'openai_api_key': os.getenv('OPENAI_API_KEY'),
           'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
           'weather_api_key': os.getenv('WEATHER_API_KEY'),
           'max_history': 50,
           'memory_storage': 'redis',
           'safety_level': args.safety_level,
           'voice_enabled': args.enable_voice,
           'default_language': args.language
       }
       
       # Create bot
       bot = InteractiveChatBot(config)
       
       if args.mode == 'interactive':
           # Start CLI interface
           cli = InteractiveCLI(bot)
           await cli.start()
           
       elif args.mode == 'api':
           # Start API server
           from api_server import start_api_server
           await start_api_server(bot, port=8000)
           
       elif args.mode == 'test':
           # Run test conversation
           await run_test_conversation(bot, args.persona)
   
   async def run_test_conversation(bot: InteractiveChatBot, persona: str):
       """Run a test conversation."""
       print("🧪 Running test conversation...\n")
       
       # Start conversation
       conversation_id = await bot.start_conversation(persona=persona)
       
       # Test messages
       test_messages = [
           "Hello! What can you help me with?",
           "What's the weather like in San Francisco?",
           "Calculate the compound interest on $10,000 at 5% for 10 years",
           "Search for the latest news on AI advancements",
           "Can you help me write a Python function to sort a list?"
       ]
       
       for message in test_messages:
           print(f"User: {message}")
           
           response = await bot.chat(conversation_id, message)
           print(f"Bot: {response['response']}\n")
           
           # Show metadata
           if response.get('metadata'):
               print(f"Intent: {response['metadata'].get('intent')}")
               print(f"Tools used: {response['metadata'].get('tools_used')}")
               print("-" * 50 + "\n")
           
           await asyncio.sleep(1)
       
       # End conversation
       summary = await bot.end_conversation(conversation_id)
       print(f"\n📋 Conversation Summary:")
       print(f"Duration: {summary['duration']:.2f} seconds")
       print(f"Messages: {summary['message_count']}")
       print(f"Summary: {summary['summary']}")
   
   if __name__ == "__main__":
       asyncio.run(main())

Advanced Capabilities
---------------------

Multi-Modal Support
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class MultiModalHandler:
       """Handle multi-modal inputs and outputs."""
       
       async def process_image(
           self,
           image_path: str,
           query: str
       ) -> Dict[str, Any]:
           """Process image with query."""
           # Use vision model to analyze image
           analysis = await self.vision_model.analyze(image_path)
           
           # Generate response based on image and query
           response = await self.generate_image_response(
               analysis,
               query
           )
           
           return {
               'image_description': analysis['description'],
               'objects_detected': analysis['objects'],
               'response': response
           }
       
       async def generate_image(
           self,
           prompt: str,
           style: Optional[str] = None
       ) -> str:
           """Generate image from text prompt."""
           # Use image generation model
           image_url = await self.image_generator.generate(
               prompt,
               style=style
           )
           
           return image_url

Conversation Analytics
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class ConversationAnalytics:
       """Analyze conversation patterns and metrics."""
       
       async def analyze_conversation(
           self,
           conversation: ConversationState
       ) -> Dict[str, Any]:
           """Analyze a complete conversation."""
           analytics = {
               'basic_metrics': self._calculate_basic_metrics(conversation),
               'sentiment_analysis': await self._analyze_sentiment(conversation),
               'topic_distribution': await self._analyze_topics(conversation),
               'engagement_metrics': self._calculate_engagement(conversation),
               'tool_usage': self._analyze_tool_usage(conversation)
           }
           
           return analytics
       
       def _calculate_basic_metrics(
           self,
           conversation: ConversationState
       ) -> Dict[str, Any]:
           """Calculate basic conversation metrics."""
           messages = conversation.messages
           
           return {
               'total_messages': len(messages),
               'user_messages': len([m for m in messages if m['role'] == 'user']),
               'bot_messages': len([m for m in messages if m['role'] == 'assistant']),
               'avg_message_length': sum(len(m['content']) for m in messages) / len(messages) if messages else 0,
               'conversation_duration': (conversation.last_activity - conversation.created_at).total_seconds()
           }

Best Practices
--------------

1. **Context Management**: Keep conversation context relevant and focused
2. **Memory Efficiency**: Balance between context window and memory usage
3. **Response Quality**: Ensure responses are helpful, harmless, and honest
4. **Tool Usage**: Use tools judiciously to enhance responses
5. **Personalization**: Adapt to user preferences without being intrusive
6. **Safety First**: Always validate content for safety and appropriateness
7. **Performance**: Stream responses for better user experience

Summary
-------

The Interactive Chat Bot demonstrates:

- Sophisticated conversation management with context awareness
- Dynamic tool integration for enhanced capabilities
- Memory systems for personalization and continuity
- Real-time response streaming
- Multi-modal support for rich interactions
- Comprehensive safety and content controls

This bot provides a foundation for building advanced conversational AI systems for various applications.