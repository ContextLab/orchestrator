Content Creation Pipeline
=========================

This example demonstrates how to build a multi-format content creation pipeline using the Orchestrator's declarative YAML framework. The pipeline generates, optimizes, and distributes content across various platforms - all defined in pure YAML with no custom Python code required.

.. note::
   **Level:** Intermediate  
   **Duration:** 45-60 minutes  
   **Prerequisites:** Orchestrator framework installed, API keys configured

Overview
--------

The Content Creation Pipeline automates:

1. **Topic Research**: Keyword research and competitor analysis
2. **Content Generation**: Create blog posts, social media, and newsletters
3. **SEO Optimization**: Optimize content for search engines
4. **Visual Creation**: Generate images and infographics
5. **Multi-Format Export**: Adapt content for different platforms
6. **Quality Assurance**: Review and refine content
7. **A/B Testing**: Create variations for optimization
8. **Distribution**: Publish to multiple platforms automatically

**Key Features Demonstrated:**
- Declarative YAML pipeline definition
- AUTO tag resolution for natural language task descriptions
- Multi-format content generation (blog, social, email)
- SEO optimization with scoring
- Parallel processing for platform-specific content
- A/B testing setup
- Automated publishing and scheduling
- No Python code required

Quick Start
-----------

.. code-block:: bash

   # Set up environment variables
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   
   # Run the content pipeline
   orchestrator run examples/content_creation_pipeline.yaml \
     --input topic="AI in Healthcare" \
     --input formats='["blog", "social", "email"]' \
     --input auto_publish=true

Complete YAML Pipeline
----------------------

The complete pipeline is defined in ``examples/content_creation_pipeline.yaml``. Here are the key sections:

**Pipeline Structure:**

.. code-block:: yaml

   name: "Content Creation Pipeline"
   description: "Multi-format content creation with optimization and distribution"

   inputs:
     topic:
       type: string
       description: "Content topic to write about"
       required: true
     
     formats:
       type: list
       description: "Content formats to generate"
       default: ["blog", "social"]
     
     audience:
       type: string
       description: "Target audience description"
       default: "general audience interested in technology"
     
     auto_publish:
       type: boolean
       description: "Automatically publish content"
       default: false

**Key Pipeline Steps:**

1. **Research and Planning:**

.. code-block:: yaml

   - id: research_topic
     action: <AUTO>research the topic "{{topic}}" to identify:
       1. Primary and secondary keywords for SEO
       2. Current trends and angles
       3. Related topics and subtopics
       4. Competitor content analysis
       5. Target audience interests</AUTO>

2. **Content Generation:**

.. code-block:: yaml

   - id: create_blog_content
     action: <AUTO>write a comprehensive blog post about "{{topic}}" using:
       - Outline: {{generate_outline.result}}
       - Keywords: {{research_topic.result.keywords}}
       - Brand voice: {{brand_voice}}
       - Target length: {{target_length}} words
       
       Include compelling introduction, examples, data, and CTAs</AUTO>

3. **Multi-Platform Optimization:**

.. code-block:: yaml

   - id: create_social_content
     action: <AUTO>create social media posts for platforms:
       - Twitter: 3 variations (280 chars max)
       - LinkedIn: 2 variations (3000 chars max)
       - Instagram: 2 variations with caption
       - Facebook: 2 variations</AUTO>
     loop:
       foreach: ["twitter", "linkedin", "instagram", "facebook"]
       parallel: true

How It Works
------------

**1. Intelligent Content Generation**

The framework automatically:
- Researches topics using web search and analysis tools
- Generates content optimized for each platform
- Maintains consistent brand voice across formats
- Incorporates SEO best practices

**2. Platform-Specific Optimization**

Each platform gets tailored content:
- **Blog**: Long-form, SEO-optimized articles
- **Twitter**: Concise posts with hashtags
- **LinkedIn**: Professional tone with insights
- **Instagram**: Visual-first with engaging captions
- **Email**: Personalized newsletters with CTAs

**3. Quality Assurance**

Automated checks ensure:
- Grammar and spelling accuracy
- Brand voice consistency
- Fact verification
- Legal compliance
- Audience appropriateness

Running the Pipeline
--------------------

**Using the CLI:**

.. code-block:: bash

   # Basic content generation
   orchestrator run content_creation_pipeline.yaml \
     --input topic="Sustainable Technology"

   # Full campaign with all formats
   orchestrator run content_creation_pipeline.yaml \
     --input topic="Remote Work Best Practices" \
     --input formats='["blog", "social", "email"]' \
     --input target_length=2000

   # Auto-publish to platforms
   orchestrator run content_creation_pipeline.yaml \
     --input topic="AI Ethics" \
     --input auto_publish=true \
     --input goals='["educate", "thought_leadership"]'

**Using Python SDK:**

.. code-block:: python

   from orchestrator import Orchestrator
   
   # Initialize orchestrator
   orchestrator = Orchestrator()
   
   # Run content pipeline
   result = await orchestrator.run_pipeline(
       "content_creation_pipeline.yaml",
       inputs={
           "topic": "Future of Renewable Energy",
           "formats": ["blog", "social", "email"],
           "audience": "environmentally conscious professionals",
           "brand_voice": "innovative and optimistic",
           "auto_publish": True
       }
   )
   
   # Access results
   print(f"SEO Score: {result['outputs']['seo_score']}")
   print(f"Blog URL: {result['outputs']['published_urls']['blog']}")
   print(f"Campaign ID: {result['outputs']['campaign_id']}")

Example Output
--------------

**Console Output:**

.. code-block:: text

   ✍️ Content Creation Pipeline
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ✓ research_topic: Found 15 keywords, 5 trending angles (8.2s)
   ✓ generate_outline: Created 7-section outline with CTAs (4.1s)
   ✓ create_blog_content: Generated 1,542 word article (12.3s)
   ✓ optimize_seo: SEO score: 92/100 (3.4s)
   ⟳ create_social_content: Creating platform-specific posts...
     ✓ twitter: 3 tweets with hashtags (2.1s)
     ✓ linkedin: 2 professional posts (2.8s)
     ✓ instagram: 2 visual posts with captions (3.2s)
     ✓ facebook: 2 engaging posts (2.5s)
   ✓ generate_visuals: Created 6 images with alt text (15.7s)
   ✓ create_email_content: Newsletter with 3 subject lines (5.2s)
   ✓ quality_review: All content passed quality checks (4.8s)
   ✓ create_ab_tests: Set up 4 A/B test variations (2.3s)
   ✓ schedule_content: Optimal schedule created (1.8s)
   ✓ publish_content: Published to 5 platforms (6.4s)
   ✓ setup_monitoring: Analytics configured (2.1s)
   
   ✅ Pipeline completed successfully in 71.3s
   📝 Blog published: https://example.com/ai-healthcare-revolution
   📊 SEO Score: 92/100
   📱 Social posts scheduled: 12 total
   📧 Email campaign ready: 5,000 subscribers

**Generated Content Examples:**

**Blog Post Excerpt:**

.. code-block:: markdown

   # The AI Healthcare Revolution: Transforming Patient Care in 2024
   
   ## Introduction
   
   Artificial intelligence is revolutionizing healthcare delivery, from diagnostic 
   accuracy to personalized treatment plans. This comprehensive guide explores 
   the latest AI applications transforming patient care...
   
   ## Key Takeaways
   - AI improves diagnostic accuracy by up to 40%
   - Personalized treatment plans based on genetic data
   - Reduced healthcare costs through predictive analytics
   - Enhanced patient engagement via AI assistants

**Social Media Examples:**

.. code-block:: text

   Twitter:
   "🏥 AI is transforming healthcare! From 40% better diagnostics to 
   personalized treatments, the future of medicine is here. 
   
   Read our latest insights → [link]
   
   #HealthTech #AIinHealthcare #DigitalHealth #Innovation"
   
   LinkedIn:
   "The integration of AI in healthcare is yielding remarkable results. 
   Our latest analysis reveals:
   
   ✅ 40% improvement in diagnostic accuracy
   ✅ 30% reduction in treatment costs
   ✅ 50% faster drug discovery
   
   Healthcare professionals are embracing these tools to deliver better 
   patient outcomes. What's your experience with AI in healthcare?
   
   Full article: [link]"

Advanced Features
-----------------

**1. Dynamic Content Personalization:**

.. code-block:: yaml

   - id: personalize_content
     action: <AUTO>personalize content for different segments:
       - Industry professionals: Technical depth
       - General audience: Simplified explanations
       - Decision makers: ROI focus
       - Practitioners: Implementation details</AUTO>
     condition: "{{enable_personalization}} == true"

**2. Multi-Language Support:**

.. code-block:: yaml

   - id: translate_content
     action: <AUTO>translate content to target languages:
       - Spanish: Localize for Latin American audience
       - French: Adapt for European market
       - Japanese: Cultural adaptation included
       Maintain SEO optimization per language</AUTO>
     loop:
       foreach: "{{target_languages}}"

**3. Content Repurposing:**

.. code-block:: yaml

   - id: repurpose_content
     action: <AUTO>repurpose blog content into:
       - Video script for YouTube
       - Podcast talking points
       - Slide deck for presentations
       - Downloadable PDF guide
       - Interactive web story</AUTO>

Performance Optimization
------------------------

The pipeline includes several optimizations:

**1. Parallel Content Generation**
- Social media posts created simultaneously
- Visual generation runs alongside text creation
- Platform-specific optimizations in parallel

**2. Smart Caching**
- Research results cached for reuse
- Outlines saved for content variations
- SEO keywords stored for consistency

**3. Batch Processing**
- Multiple images generated together
- Social posts scheduled in batches
- Analytics configured in single call

Error Handling
--------------

The system handles various failure scenarios:

**1. Platform API Failures:**

.. code-block:: yaml

   on_error:
     action: <AUTO>save content locally with publishing 
       instructions for manual upload</AUTO>
     continue_on_error: true

**2. Content Generation Issues:**

.. code-block:: yaml

   on_error:
     action: <AUTO>retry with adjusted parameters or 
       use simpler content structure</AUTO>
     retry_count: 2

**3. SEO Optimization Failures:**

.. code-block:: yaml

   on_error:
     action: <AUTO>proceed with basic optimization 
       and flag for manual review</AUTO>
     fallback_value: "manual_seo_required"

Customization Examples
----------------------

**1. Industry-Specific Content:**

.. code-block:: yaml

   - id: industry_customize
     action: <AUTO>adapt content for {{industry}}:
       - Healthcare: Include HIPAA compliance notes
       - Finance: Add regulatory disclaimers
       - Education: Include learning objectives
       - Technology: Add technical specifications</AUTO>

**2. Campaign Themes:**

.. code-block:: yaml

   - id: apply_campaign_theme
     action: <AUTO>apply campaign theme "{{campaign_theme}}":
       - Seasonal: Holiday-specific messaging
       - Product launch: Feature highlights
       - Thought leadership: Industry insights
       - Event promotion: Registration CTAs</AUTO>

**3. Content Series:**

.. code-block:: yaml

   - id: create_series
     action: <AUTO>create content series with {{num_parts}} parts:
       - Maintain narrative continuity
       - Build on previous concepts
       - Include series navigation
       - Create anticipation for next part</AUTO>

Analytics and Reporting
-----------------------

Track content performance:

- **Engagement Metrics**: Views, clicks, shares, comments
- **SEO Performance**: Rankings, organic traffic, backlinks
- **Conversion Tracking**: Sign-ups, downloads, purchases
- **A/B Test Results**: Winner identification and insights

Key Takeaways
-------------

This example demonstrates the power of Orchestrator's declarative framework:

1. **Zero Code Required**: Complete content pipeline in pure YAML
2. **Multi-Format Support**: One source, multiple outputs
3. **Intelligent Optimization**: Platform-specific adaptations
4. **Automated Distribution**: Publish everywhere from one place
5. **Quality Assurance**: Built-in review and optimization
6. **Performance Tracking**: Analytics and testing included

The declarative approach makes sophisticated content operations accessible without programming.

Next Steps
----------

- Try the :doc:`code_analysis_suite` for technical content
- Explore :doc:`customer_support_automation` for support content
- Read the :doc:`../../advanced/content_optimization` guide
- Check the :doc:`../../user_guide/publishing_integrations` guide