Creative Writing Assistant
==========================

This example demonstrates how to build an AI-powered creative writing assistant that helps authors generate stories, develop characters, create plots, and maintain consistency across narrative elements. The system combines multiple AI models to provide comprehensive writing support.

.. note::
   **Level:** Intermediate  
   **Duration:** 60-75 minutes  
   **Prerequisites:** Basic Python knowledge, understanding of creative writing concepts, interest in AI-assisted content creation

Overview
--------

The Creative Writing Assistant provides:

1. **Story Generation**: Create original stories with consistent narrative
2. **Character Development**: Design complex characters with detailed backgrounds
3. **Plot Construction**: Build compelling plots with proper story arcs
4. **World Building**: Create rich, consistent fictional worlds
5. **Dialogue Generation**: Write natural, character-appropriate dialogue
6. **Style Adaptation**: Match specific writing styles and genres
7. **Consistency Checking**: Ensure narrative consistency throughout

**Key Features:**
- Multi-genre support (fantasy, sci-fi, mystery, romance, etc.)
- Character relationship mapping
- Plot twist generation
- Writing style analysis and emulation
- Collaborative writing mode
- Story bible maintenance
- Export to various formats

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
   python examples/creative_writing_assistant.py \
     --genre fantasy \
     --length novel \
     --style "tolkien-inspired"

Complete Implementation
-----------------------

Pipeline Configuration (YAML)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # creative_writing_pipeline.yaml
   id: creative_writing_assistant
   name: AI-Powered Creative Writing Pipeline
   version: "1.0"
   
   metadata:
     description: "Comprehensive creative writing assistance system"
     author: "Creative AI Team"
     tags: ["writing", "storytelling", "creative-ai", "narrative"]
   
   models:
     story_generator:
       provider: "openai"
       model: "gpt-4"
       temperature: 0.8
     character_developer:
       provider: "anthropic"
       model: "claude-3-opus"
       temperature: 0.7
     dialogue_writer:
       provider: "openai"
       model: "gpt-4"
       temperature: 0.9
     consistency_checker:
       provider: "anthropic"
       model: "claude-3-opus"
       temperature: 0.2
   
   context:
     genre: "{{ inputs.genre }}"
     target_length: "{{ inputs.length }}"
     writing_style: "{{ inputs.style }}"
     target_audience: "{{ inputs.audience }}"
   
   tasks:
     - id: analyze_genre
       name: "Analyze Genre Requirements"
       action: "analyze_genre_conventions"
       model: "story_generator"
       parameters:
         genre: "{{ context.genre }}"
         sub_genres: <AUTO>Identify relevant sub-genres</AUTO>
         tropes: true
         audience_expectations: true
       outputs:
         - genre_analysis
         - key_elements
         - common_tropes
     
     - id: generate_premise
       name: "Generate Story Premise"
       action: "create_story_premise"
       model: "story_generator"
       parameters:
         genre: "{{ context.genre }}"
         genre_elements: "{{ analyze_genre.key_elements }}"
         originality_score: <AUTO>Balance familiarity with innovation</AUTO>
         themes: <AUTO>Select appropriate themes for genre</AUTO>
       dependencies:
         - analyze_genre
       outputs:
         - premise
         - central_conflict
         - themes
         - setting_basics
     
     - id: develop_characters
       name: "Develop Main Characters"
       action: "create_characters"
       model: "character_developer"
       parameters:
         story_premise: "{{ generate_premise.premise }}"
         character_count: <AUTO>Determine optimal number of main characters</AUTO>
         depth_level: "comprehensive"
         include_backstories: true
         personality_profiles: true
       dependencies:
         - generate_premise
       outputs:
         - main_characters
         - character_profiles
         - character_arcs
         - relationships
     
     - id: create_world
       name: "Build Story World"
       action: "worldbuilding"
       model: "story_generator"
       condition: "context.genre in ['fantasy', 'sci-fi', 'dystopian']"
       parameters:
         premise: "{{ generate_premise.premise }}"
         setting: "{{ generate_premise.setting_basics }}"
         detail_level: <AUTO>Based on story length and genre</AUTO>
         consistency_rules: true
       dependencies:
         - generate_premise
       outputs:
         - world_details
         - locations
         - cultures
         - world_rules
     
     - id: plot_structure
       name: "Create Plot Structure"
       action: "design_plot"
       model: "story_generator"
       parameters:
         premise: "{{ generate_premise.premise }}"
         characters: "{{ develop_characters.main_characters }}"
         story_length: "{{ context.target_length }}"
         plot_structure: <AUTO>Three-act, five-act, or hero's journey</AUTO>
         include_subplots: true
       dependencies:
         - develop_characters
       outputs:
         - plot_outline
         - major_events
         - turning_points
         - chapter_breakdown
     
     - id: generate_scenes
       name: "Generate Key Scenes"
       action: "write_scenes"
       model: "story_generator"
       parallel: true
       for_each: "{{ plot_structure.major_events[:5] }}"
       parameters:
         scene_description: "{{ item }}"
         characters_involved: "{{ item.characters }}"
         scene_purpose: "{{ item.purpose }}"
         writing_style: "{{ context.writing_style }}"
         word_count: <AUTO>Based on scene importance</AUTO>
       dependencies:
         - plot_structure
       outputs:
         - scene_content
         - scene_metadata
     
     - id: write_dialogue
       name: "Generate Character Dialogue"
       action: "create_dialogue"
       model: "dialogue_writer"
       parallel: true
       for_each: "{{ generate_scenes.scene_content }}"
       parameters:
         scene: "{{ item }}"
         character_profiles: "{{ develop_characters.character_profiles }}"
         dialogue_style: <AUTO>Match character voice and personality</AUTO>
         subtext: true
       dependencies:
         - generate_scenes
       outputs:
         - dialogue_enhanced_scenes
         - character_voice_consistency
     
     - id: add_descriptions
       name: "Enhance Descriptive Elements"
       action: "enhance_descriptions"
       model: "story_generator"
       parameters:
         scenes: "{{ write_dialogue.dialogue_enhanced_scenes }}"
         world_details: "{{ create_world.world_details }}"
         sensory_details: true
         atmosphere: <AUTO>Match genre and scene mood</AUTO>
       dependencies:
         - write_dialogue
         - create_world
       outputs:
         - enhanced_scenes
         - descriptive_elements
     
     - id: check_consistency
       name: "Verify Narrative Consistency"
       action: "consistency_check"
       model: "consistency_checker"
       parameters:
         full_content: "{{ add_descriptions.enhanced_scenes }}"
         character_profiles: "{{ develop_characters.character_profiles }}"
         world_rules: "{{ create_world.world_rules }}"
         plot_outline: "{{ plot_structure.plot_outline }}"
       dependencies:
         - add_descriptions
       outputs:
         - consistency_report
         - inconsistencies
         - suggested_fixes
     
     - id: generate_chapters
       name: "Organize into Chapters"
       action: "create_chapters"
       model: "story_generator"
       parameters:
         scenes: "{{ add_descriptions.enhanced_scenes }}"
         chapter_structure: "{{ plot_structure.chapter_breakdown }}"
         transitions: <AUTO>Create smooth chapter transitions</AUTO>
         cliffhangers: true
       dependencies:
         - check_consistency
       outputs:
         - chapters
         - chapter_summaries
         - reading_time
     
     - id: create_story_bible
       name: "Generate Story Bible"
       action: "compile_story_bible"
       parameters:
         characters: "{{ develop_characters.character_profiles }}"
         world: "{{ create_world.world_details }}"
         plot: "{{ plot_structure.plot_outline }}"
         style_guide: "{{ context.writing_style }}"
       dependencies:
         - generate_chapters
       outputs:
         - story_bible
         - character_sheets
         - world_map
         - timeline

Python Implementation
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # creative_writing_assistant.py
   import asyncio
   import os
   from pathlib import Path
   from typing import Dict, List, Any, Optional
   import json
   from datetime import datetime
   import yaml
   
   from orchestrator import Orchestrator
   from orchestrator.tools.writing_tools import (
       StoryGeneratorTool,
       CharacterDeveloperTool,
       DialogueWriterTool,
       WorldBuilderTool,
       ConsistencyCheckerTool
   )
   from orchestrator.integrations.export import StoryExporter
   
   
   class CreativeWritingAssistant:
       """
       AI-powered creative writing assistant for story generation and development.
       
       Features:
       - Multi-genre story generation
       - Character development and arc planning
       - World building and consistency
       - Dialogue generation
       - Style adaptation
       """
       
       def __init__(self, config: Dict[str, Any]):
           self.config = config
           self.orchestrator = None
           self.story_data = {}
           self._setup_assistant()
       
       def _setup_assistant(self):
           """Initialize writing assistant components."""
           self.orchestrator = Orchestrator()
           
           # Register AI models
           self._register_models()
           
           # Initialize tools
           self.tools = {
               'story_generator': StoryGeneratorTool(self.config),
               'character_developer': CharacterDeveloperTool(),
               'dialogue_writer': DialogueWriterTool(),
               'world_builder': WorldBuilderTool(),
               'consistency_checker': ConsistencyCheckerTool(),
               'exporter': StoryExporter()
           }
       
       async def create_story(
           self,
           genre: str,
           length: str = 'short_story',
           style: Optional[str] = None,
           audience: str = 'general',
           initial_premise: Optional[str] = None,
           **kwargs
       ) -> Dict[str, Any]:
           """
           Create a complete story with AI assistance.
           
           Args:
               genre: Story genre (fantasy, sci-fi, mystery, etc.)
               length: Target length (flash, short_story, novella, novel)
               style: Writing style to emulate
               audience: Target audience
               initial_premise: Optional starting premise
               
           Returns:
               Complete story with metadata
           """
           print(f"✍️ Starting creative writing process for {genre} {length}")
           
           # Prepare context
           context = {
               'genre': genre,
               'length': length,
               'style': style or 'contemporary',
               'audience': audience,
               'initial_premise': initial_premise,
               'timestamp': datetime.now().isoformat(),
               **kwargs
           }
           
           # Execute pipeline
           try:
               results = await self.orchestrator.execute_pipeline(
                   'creative_writing_pipeline.yaml',
                   context=context,
                   progress_callback=self._progress_callback
               )
               
               # Process results
               story = await self._process_story_results(results)
               
               # Save story data
               self.story_data = story
               
               # Export story
               await self._export_story(story, context.get('export_format', 'markdown'))
               
               return story
               
           except Exception as e:
               print(f"❌ Story creation failed: {str(e)}")
               raise
       
       async def _progress_callback(self, task_id: str, progress: float, message: str):
           """Handle progress updates."""
           icons = {
               'analyze_genre': '📚',
               'generate_premise': '💡',
               'develop_characters': '👥',
               'create_world': '🌍',
               'plot_structure': '📊',
               'generate_scenes': '🎬',
               'write_dialogue': '💬',
               'add_descriptions': '🖋️',
               'check_consistency': '✓',
               'generate_chapters': '📖',
               'create_story_bible': '📔'
           }
           icon = icons.get(task_id, '•')
           print(f"{icon} {task_id}: {progress:.0%} - {message}")
       
       async def _process_story_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
           """Process and organize story results."""
           story = {
               'metadata': {
                   'title': self._generate_title(results),
                   'genre': results['context']['genre'],
                   'length': results['context']['length'],
                   'created_at': datetime.now().isoformat(),
                   'word_count': 0
               },
               'premise': results.get('generate_premise', {}).get('premise', ''),
               'characters': self._organize_characters(
                   results.get('develop_characters', {})
               ),
               'world': results.get('create_world', {}),
               'plot': results.get('plot_structure', {}),
               'chapters': results.get('generate_chapters', {}).get('chapters', []),
               'story_bible': results.get('create_story_bible', {}).get('story_bible', {}),
               'consistency_report': results.get('check_consistency', {})
           }
           
           # Calculate word count
           story['metadata']['word_count'] = sum(
               len(chapter.get('content', '').split())
               for chapter in story['chapters']
           )
           
           return story
       
       def _generate_title(self, results: Dict[str, Any]) -> str:
           """Generate story title from premise."""
           premise = results.get('generate_premise', {}).get('premise', '')
           # In real implementation, use AI to generate title
           return "The Untitled Story"
       
       def _organize_characters(self, character_data: Dict[str, Any]) -> List[Dict]:
           """Organize character information."""
           characters = []
           
           main_chars = character_data.get('main_characters', [])
           profiles = character_data.get('character_profiles', {})
           arcs = character_data.get('character_arcs', {})
           
           for char in main_chars:
               char_id = char['id']
               characters.append({
                   'name': char['name'],
                   'role': char['role'],
                   'profile': profiles.get(char_id, {}),
                   'arc': arcs.get(char_id, {}),
                   'relationships': char.get('relationships', [])
               })
           
           return characters
       
       async def _export_story(self, story: Dict[str, Any], format: str):
           """Export story to specified format."""
           timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
           filename = f"{story['metadata']['title'].replace(' ', '_')}_{timestamp}"
           
           if format == 'markdown':
               content = self._format_as_markdown(story)
               output_file = f"{filename}.md"
               Path(output_file).write_text(content)
               print(f"✅ Story exported to: {output_file}")
           
           elif format == 'json':
               output_file = f"{filename}.json"
               with open(output_file, 'w') as f:
                   json.dump(story, f, indent=2)
               print(f"✅ Story data exported to: {output_file}")
           
           # Also save story bible
           bible_file = f"{filename}_bible.yaml"
           with open(bible_file, 'w') as f:
               yaml.dump(story['story_bible'], f)
           print(f"📔 Story bible saved to: {bible_file}")
       
       def _format_as_markdown(self, story: Dict[str, Any]) -> str:
           """Format story as markdown."""
           md_content = f"# {story['metadata']['title']}\n\n"
           md_content += f"*Genre: {story['metadata']['genre']}*\n"
           md_content += f"*Word Count: {story['metadata']['word_count']:,}*\n\n"
           
           # Add chapters
           for i, chapter in enumerate(story['chapters'], 1):
               md_content += f"\n## Chapter {i}: {chapter.get('title', 'Untitled')}\n\n"
               md_content += chapter.get('content', '') + "\n"
           
           return md_content

Character Development
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class CharacterDeveloper:
       """Advanced character development system."""
       
       async def create_character(
           self,
           role: str,
           story_context: Dict[str, Any],
           depth: str = 'comprehensive'
       ) -> Dict[str, Any]:
           """Create a detailed character."""
           character = {
               'id': self._generate_character_id(),
               'role': role,
               'basic_info': await self._generate_basic_info(role, story_context),
               'personality': await self._generate_personality(depth),
               'backstory': await self._generate_backstory(role, story_context),
               'motivations': await self._generate_motivations(role, story_context),
               'arc': await self._plan_character_arc(role, story_context),
               'voice': await self._develop_character_voice()
           }
           
           if depth == 'comprehensive':
               character.update({
                   'relationships': [],
                   'internal_conflicts': await self._generate_internal_conflicts(),
                   'external_conflicts': await self._generate_external_conflicts(story_context),
                   'skills_abilities': await self._generate_skills(role, story_context),
                   'flaws': await self._generate_character_flaws(),
                   'growth_points': await self._identify_growth_opportunities()
               })
           
           return character
       
       async def _generate_personality(self, depth: str) -> Dict[str, Any]:
           """Generate character personality."""
           personality = {
               'traits': await self._select_personality_traits(),
               'myers_briggs': await self._determine_mbti_type(),
               'enneagram': await self._determine_enneagram_type(),
               'values': await self._identify_core_values(),
               'fears': await self._identify_fears(),
               'desires': await self._identify_desires()
           }
           
           if depth == 'comprehensive':
               personality['quirks'] = await self._generate_quirks()
               personality['habits'] = await self._generate_habits()
               personality['speech_patterns'] = await self._analyze_speech_patterns()
           
           return personality
       
       async def develop_character_relationships(
           self,
           characters: List[Dict[str, Any]]
       ) -> Dict[str, Any]:
           """Develop relationships between characters."""
           relationships = {}
           
           for i, char1 in enumerate(characters):
               for char2 in characters[i+1:]:
                   relationship = await self._generate_relationship(char1, char2)
                   key = f"{char1['id']}_{char2['id']}"
                   relationships[key] = relationship
           
           return relationships

World Building
^^^^^^^^^^^^^^

.. code-block:: python

   class WorldBuilder:
       """Create rich, consistent fictional worlds."""
       
       async def build_world(
           self,
           genre: str,
           premise: str,
           scope: str = 'medium'
       ) -> Dict[str, Any]:
           """Build a complete fictional world."""
           world = {
               'name': await self._generate_world_name(genre),
               'type': self._determine_world_type(genre),
               'geography': await self._create_geography(scope),
               'history': await self._generate_history(scope),
               'cultures': await self._develop_cultures(genre, scope),
               'technology_magic': await self._define_systems(genre),
               'rules': await self._establish_world_rules(genre),
               'languages': await self._create_languages(scope)
           }
           
           if genre in ['fantasy', 'sci-fi']:
               world['unique_elements'] = await self._add_unique_elements(genre)
           
           return world
       
       async def _create_geography(self, scope: str) -> Dict[str, Any]:
           """Create world geography."""
           geography = {
               'continents': [],
               'major_cities': [],
               'landmarks': [],
               'climate_zones': [],
               'resources': []
           }
           
           # Determine number of locations based on scope
           location_count = {
               'small': 3,
               'medium': 7,
               'large': 15,
               'epic': 30
           }.get(scope, 7)
           
           for i in range(location_count):
               location = await self._generate_location(i)
               if location['type'] == 'city':
                   geography['major_cities'].append(location)
               elif location['type'] == 'landmark':
                   geography['landmarks'].append(location)
           
           return geography
       
       async def create_magic_system(self) -> Dict[str, Any]:
           """Create a consistent magic system."""
           magic_system = {
               'name': await self._generate_magic_name(),
               'source': await self._determine_magic_source(),
               'rules': await self._create_magic_rules(),
               'limitations': await self._define_limitations(),
               'costs': await self._define_magic_costs(),
               'practitioners': await self._define_magic_users(),
               'spells_abilities': await self._create_spell_list()
           }
           
           return magic_system

Plot Development
^^^^^^^^^^^^^^^^

.. code-block:: python

   class PlotDeveloper:
       """Create compelling plot structures."""
       
       async def create_plot_structure(
           self,
           premise: str,
           characters: List[Dict],
           length: str,
           structure_type: str = 'three_act'
       ) -> Dict[str, Any]:
           """Create detailed plot structure."""
           if structure_type == 'three_act':
               plot = await self._create_three_act_structure(premise, characters, length)
           elif structure_type == 'five_act':
               plot = await self._create_five_act_structure(premise, characters, length)
           elif structure_type == 'heros_journey':
               plot = await self._create_heros_journey(premise, characters, length)
           else:
               plot = await self._create_custom_structure(premise, characters, length)
           
           # Add subplots
           plot['subplots'] = await self._generate_subplots(characters, plot)
           
           # Add pacing
           plot['pacing'] = await self._determine_pacing(plot, length)
           
           return plot
       
       async def _create_three_act_structure(
           self,
           premise: str,
           characters: List[Dict],
           length: str
       ) -> Dict[str, Any]:
           """Create three-act structure."""
           return {
               'act_1': {
                   'setup': await self._generate_setup(premise, characters),
                   'inciting_incident': await self._create_inciting_incident(premise),
                   'first_plot_point': await self._create_plot_point(1, premise, characters)
               },
               'act_2': {
                   'rising_action': await self._generate_rising_action(characters),
                   'midpoint': await self._create_midpoint(premise, characters),
                   'complications': await self._generate_complications(characters),
                   'second_plot_point': await self._create_plot_point(2, premise, characters)
               },
               'act_3': {
                   'climax': await self._create_climax(premise, characters),
                   'falling_action': await self._generate_falling_action(characters),
                   'resolution': await self._create_resolution(premise, characters)
               }
           }
       
       async def generate_plot_twists(
           self,
           plot: Dict[str, Any],
           count: int = 2
       ) -> List[Dict[str, Any]]:
           """Generate plot twists."""
           twists = []
           
           for i in range(count):
               twist = {
                   'type': await self._select_twist_type(),
                   'timing': await self._determine_twist_timing(plot, i),
                   'setup': await self._create_twist_setup(),
                   'reveal': await self._create_twist_reveal(),
                   'impact': await self._analyze_twist_impact(plot)
               }
               twists.append(twist)
           
           return twists

Style Analysis and Adaptation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class StyleAdapter:
       """Analyze and adapt writing styles."""
       
       async def analyze_writing_style(
           self,
           sample_text: str
       ) -> Dict[str, Any]:
           """Analyze writing style characteristics."""
           analysis = {
               'sentence_structure': await self._analyze_sentences(sample_text),
               'vocabulary': await self._analyze_vocabulary(sample_text),
               'tone': await self._determine_tone(sample_text),
               'pacing': await self._analyze_pacing(sample_text),
               'literary_devices': await self._identify_devices(sample_text),
               'point_of_view': await self._identify_pov(sample_text)
           }
           
           return analysis
       
       async def adapt_to_style(
           self,
           content: str,
           target_style: Dict[str, Any]
       ) -> str:
           """Adapt content to match target style."""
           # Apply style transformations
           adapted = content
           
           # Adjust sentence structure
           adapted = await self._adjust_sentences(adapted, target_style['sentence_structure'])
           
           # Adjust vocabulary
           adapted = await self._adjust_vocabulary(adapted, target_style['vocabulary'])
           
           # Adjust tone
           adapted = await self._adjust_tone(adapted, target_style['tone'])
           
           return adapted

Interactive Writing Mode
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class InteractiveWriter:
       """Support interactive writing sessions."""
       
       def __init__(self, assistant):
           self.assistant = assistant
           self.session_data = {}
       
       async def start_session(self):
           """Start interactive writing session."""
           print("🖊️ Interactive Writing Mode Started")
           print("Commands: /character, /plot, /describe, /dialogue, /help")
           
           while True:
               user_input = input("\n> ")
               
               if user_input.startswith('/'):
                   await self._handle_command(user_input)
               elif user_input.lower() == 'exit':
                   break
               else:
                   await self._continue_story(user_input)
       
       async def _handle_command(self, command: str):
           """Handle interactive commands."""
           parts = command.split()
           cmd = parts[0]
           
           if cmd == '/character':
               character = await self._create_character_interactive()
               print(f"Created character: {character['name']}")
               
           elif cmd == '/plot':
               suggestion = await self._suggest_plot_development()
               print(f"Plot suggestion: {suggestion}")
               
           elif cmd == '/dialogue':
               dialogue = await self._generate_dialogue_interactive()
               print(f"Generated dialogue:\n{dialogue}")
               
           elif cmd == '/describe':
               description = await self._generate_description(parts[1:])
               print(f"Description: {description}")

Running the Assistant
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # main.py
   import asyncio
   import argparse
   from creative_writing_assistant import CreativeWritingAssistant
   
   async def main():
       parser = argparse.ArgumentParser(description='Creative Writing Assistant')
       parser.add_argument('--genre', required=True, 
                          choices=['fantasy', 'sci-fi', 'mystery', 'romance', 
                                  'thriller', 'horror', 'literary'])
       parser.add_argument('--length', default='short_story',
                          choices=['flash', 'short_story', 'novella', 'novel'])
       parser.add_argument('--style', help='Writing style to emulate')
       parser.add_argument('--interactive', action='store_true',
                          help='Start interactive writing mode')
       parser.add_argument('--premise', help='Initial story premise')
       parser.add_argument('--export-format', default='markdown',
                          choices=['markdown', 'docx', 'pdf', 'json'])
       
       args = parser.parse_args()
       
       # Configuration
       config = {
           'openai_api_key': os.getenv('OPENAI_API_KEY'),
           'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
           'model_preferences': {
               'creativity': 'high',
               'consistency': 'strict'
           }
       }
       
       # Create assistant
       assistant = CreativeWritingAssistant(config)
       
       if args.interactive:
           # Start interactive mode
           from interactive_writer import InteractiveWriter
           interactive = InteractiveWriter(assistant)
           await interactive.start_session()
       else:
           # Generate complete story
           story = await assistant.create_story(
               genre=args.genre,
               length=args.length,
               style=args.style,
               initial_premise=args.premise,
               export_format=args.export_format
           )
           
           # Display results
           print("\n📖 Story Created Successfully!")
           print(f"Title: {story['metadata']['title']}")
           print(f"Genre: {story['metadata']['genre']}")
           print(f"Word Count: {story['metadata']['word_count']:,}")
           
           print("\n👥 Main Characters:")
           for char in story['characters'][:3]:
               print(f"  - {char['name']} ({char['role']})")
           
           print(f"\n📊 Plot Structure: {len(story['chapters'])} chapters")
           
           if story.get('consistency_report', {}).get('inconsistencies'):
               print("\n⚠️ Consistency Issues Found:")
               for issue in story['consistency_report']['inconsistencies'][:3]:
                   print(f"  - {issue}")
   
   if __name__ == "__main__":
       asyncio.run(main())

Best Practices
--------------

1. **Character Consistency**: Maintain character voice throughout the story
2. **Plot Coherence**: Ensure all plot threads are resolved
3. **World Building**: Keep world rules consistent
4. **Pacing**: Balance action, dialogue, and description
5. **Show Don't Tell**: Use scenes to reveal character and plot
6. **Revision**: Always review AI-generated content
7. **Style Guide**: Maintain consistent style throughout

Summary
-------

The Creative Writing Assistant demonstrates:

- AI-powered story generation with narrative consistency
- Comprehensive character development and world building
- Multi-genre support with style adaptation
- Interactive writing mode for collaboration
- Plot structure generation with proper story arcs
- Consistency checking and story bible maintenance

This assistant provides a powerful foundation for AI-assisted creative writing across various genres and formats.