Customer Support Automation
===========================

This example demonstrates how to build a comprehensive customer support automation system using the Orchestrator's declarative YAML framework. The system handles ticket management, automated responses, knowledge base integration, and intelligent routing - all defined in pure YAML with no custom Python code required.

.. note::
   **Level:** Advanced  
   **Duration:** 75-90 minutes  
   **Prerequisites:** Orchestrator framework installed, ticketing system credentials

Overview
--------

The Customer Support Automation system provides:

1. **Ticket Reception**: Fetch and process support tickets
2. **Language Detection**: Multi-language support capabilities
3. **Sentiment Analysis**: Understand customer emotions and urgency
4. **Smart Classification**: Categorize and prioritize tickets
5. **Knowledge Base Search**: Find relevant solutions automatically
6. **Automated Responses**: Generate personalized responses
7. **Intelligent Routing**: Assign to best available agents
8. **SLA Monitoring**: Track and ensure compliance
9. **Analytics Tracking**: Comprehensive metrics and insights

**Key Features Demonstrated:**
- Declarative YAML pipeline definition
- AUTO tag resolution for natural language task descriptions
- Integration with ticketing systems (Zendesk, Freshdesk, Jira)
- Multi-language support with translation
- Sentiment-based escalation logic
- Knowledge base integration
- SLA compliance monitoring
- No Python code required

Quick Start
-----------

.. code-block:: bash

   # Set up environment variables
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export ZENDESK_SUBDOMAIN="your-subdomain"
   export ZENDESK_TOKEN="your-token"
   
   # Run customer support automation
   orchestrator run examples/customer_support_automation.yaml \
     --input ticket_id="TICKET-12345" \
     --input ticketing_system="zendesk" \
     --input auto_respond=true

Complete YAML Pipeline
----------------------

The complete pipeline is defined in ``examples/customer_support_automation.yaml``. Here are the key sections:

**Pipeline Structure:**

.. code-block:: yaml

   name: "Customer Support Automation"
   description: "AI-powered customer support with intelligent routing"

   inputs:
     ticket_id:
       type: string
       description: "Support ticket ID to process"
       required: true
     
     auto_respond:
       type: boolean
       description: "Enable automated responses"
       default: true
     
     languages:
       type: list
       description: "Supported languages"
       default: ["en", "es", "fr"]

**Key Pipeline Steps:**

1. **Sentiment Analysis:**

.. code-block:: yaml

   - id: analyze_sentiment
     action: <AUTO>analyze customer sentiment and emotion:
       Current message: {{receive_ticket.result.description}}
       
       Detect:
       1. Overall sentiment score (-1 to 1)
       2. Specific emotions (anger, frustration, satisfaction)
       3. Urgency level
       4. Customer frustration indicators
       5. Potential churn risk</AUTO>

2. **Intelligent Classification:**

.. code-block:: yaml

   - id: classify_ticket
     action: <AUTO>classify the support ticket:
       Determine:
       1. Primary category (billing, technical, account, etc.)
       2. Secondary categories
       3. Relevant tags
       4. Priority score (0-10)
       5. Complexity level</AUTO>

3. **Automated Response Generation:**

.. code-block:: yaml

   - id: generate_response
     action: <AUTO>generate personalized support response:
       Create response that:
       1. Acknowledges the specific issue
       2. Shows empathy matching customer sentiment
       3. Provides clear solution steps
       4. Includes relevant KB article links
       5. Sets expectations for resolution
       6. Offers additional help if needed</AUTO>

How It Works
------------

**1. Intelligent Ticket Processing**

The framework automatically:
- Detects customer language and translates if needed
- Analyzes sentiment to understand emotional state
- Extracts key information (order numbers, products, etc.)
- Classifies tickets into appropriate categories

**2. Smart Automation Decisions**

The system decides whether to automate based on:
- Customer sentiment (escalate if too negative)
- Knowledge base confidence
- Customer tier and history
- Ticket complexity
- Regulatory requirements

**3. Personalized Responses**

When automating, the system:
- Generates context-aware responses
- Matches customer tone and formality
- Includes relevant solutions
- Provides clear next steps
- Maintains brand voice

Running the Pipeline
--------------------

**Using the CLI:**

.. code-block:: bash

   # Process single ticket
   orchestrator run customer_support_automation.yaml \
     --input ticket_id="TICKET-12345"

   # Process with specific settings
   orchestrator run customer_support_automation.yaml \
     --input ticket_id="TICKET-12345" \
     --input auto_respond=true \
     --input escalation_threshold=-0.3

   # Batch process tickets
   orchestrator run customer_support_automation.yaml \
     --input ticket_id="BATCH" \
     --input batch_size=10

**Using Python SDK:**

.. code-block:: python

   from orchestrator import Orchestrator
   
   # Initialize orchestrator
   orchestrator = Orchestrator()
   
   # Process support ticket
   result = await orchestrator.run_pipeline(
       "customer_support_automation.yaml",
       inputs={
           "ticket_id": "TICKET-12345",
           "ticketing_system": "zendesk",
           "auto_respond": True,
           "languages": ["en", "es", "fr", "de"]
       }
   )
   
   # Access results
   print(f"Status: {result['outputs']['automation_status']}")
   print(f"Assigned to: {result['outputs']['assigned_agent']}")
   print(f"SLA Status: {result['outputs']['sla_status']}")

Example Output
--------------

**Console Output:**

.. code-block:: text

   🎫 Customer Support Automation
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ✓ receive_ticket: Fetched ticket TICKET-12345 (1.2s)
   ✓ detect_language: Detected English (confidence: 0.99) (0.8s)
   ✓ analyze_sentiment: Sentiment: -0.7 (frustrated) (2.1s)
   ✓ extract_entities: Found order #12345, product SKU-789 (1.5s)
   ✓ classify_ticket: Category: account_access, Priority: 9/10 (2.3s)
   ✓ search_knowledge_base: Found 3 relevant articles (3.2s)
   ✓ check_automation_eligibility: Escalating - high frustration (0.5s)
   ✓ update_ticket: Updated category, priority, tags (1.1s)
   ✓ assign_to_agent: Assigned to Sarah Johnson (1.8s)
   ✓ monitor_sla: First response due in 15 minutes (0.4s)
   ✓ create_followup: Scheduled satisfaction survey (0.9s)
   ✓ log_analytics: Metrics logged (0.3s)
   
   ✅ Pipeline completed in 15.9s
   
   📊 RESULTS
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   🎫 Ticket: TICKET-12345
   📂 Category: Account Access
   🔥 Priority: 9/10 (High)
   😤 Sentiment: -0.7 (Frustrated)
   🤖 Status: Escalated to Human
   👤 Agent: Sarah Johnson
   ⏱️ SLA: At Risk (15 min remaining)

**Escalation Example:**

When a ticket requires human intervention, the system provides a comprehensive handoff package:

.. code-block:: text

   👤 Agent Handoff Package
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   
   Customer: John Doe (Premium Tier)
   Issue: Account access - password reset not working
   Sentiment: Frustrated (-0.7)
   Previous Tickets: 3 (1 escalation)
   
   🔍 Key Information:
   - Email: john.doe@example.com
   - Customer ID: CUST-98765
   - Error: "Invalid credentials"
   - Urgency: Meeting in 2 hours
   
   💡 Suggested Approach:
   1. Apologetic tone - customer is frustrated
   2. Manual password reset immediately
   3. Check email server for delivery issues
   4. Consider service credit ($50 suggested)
   5. Follow up within 24 hours
   
   📚 Relevant KB Articles:
   - Account Access Troubleshooting (92% match)
   - Password Reset Issues (88% match)

Advanced Features
-----------------

**1. Multi-Language Support:**

.. code-block:: yaml

   - id: translate_response
     action: <AUTO>translate response to customer's language:
       Target language: {{detect_language.result.language}}
       
       Maintain:
       1. Professional tone
       2. Technical accuracy
       3. Cultural appropriateness</AUTO>
     condition: "{{detect_language.result.language}} != 'en'"

**2. Dynamic Priority Calculation:**

.. code-block:: yaml

   - id: calculate_priority
     action: <AUTO>calculate ticket priority based on:
       - Customer tier (premium = +3)
       - Sentiment score (negative = +2)
       - Urgency keywords (urgent, asap = +1)
       - Category severity
       - Business impact</AUTO>

**3. Proactive Support:**

.. code-block:: yaml

   - id: proactive_check
     action: <AUTO>identify proactive support opportunities:
       - Similar issues trending
       - Preventable problems
       - Feature education needs
       - Upsell opportunities</AUTO>

Performance Optimization
------------------------

The pipeline optimizes performance through:

**1. Intelligent Caching**
- Cache customer information
- Store KB search results
- Reuse classification models

**2. Parallel Processing**
- Sentiment analysis and entity extraction in parallel
- Simultaneous KB search and classification
- Batch ticket processing

**3. Smart Routing**
- Skills-based agent matching
- Workload balancing
- Language capabilities
- Historical performance

Error Handling
--------------

The system handles various scenarios gracefully:

**1. System Unavailability:**

.. code-block:: yaml

   on_error:
     action: <AUTO>create manual task for agent with all 
       extracted information and priority flag</AUTO>
     continue_on_error: true

**2. Translation Failures:**

.. code-block:: yaml

   on_error:
     action: <AUTO>proceed with original language and 
       flag for bilingual agent</AUTO>
     fallback_value: "original_text"

**3. Knowledge Base Issues:**

.. code-block:: yaml

   on_error:
     action: <AUTO>skip KB search and route directly 
       to specialist agent</AUTO>
     continue_on_error: true

Real-World Integration
----------------------

**1. Zendesk Integration:**

.. code-block:: bash

   orchestrator run customer_support_automation.yaml \
     --input ticketing_system="zendesk" \
     --input zendesk_subdomain="mycompany" \
     --input auto_create_macros=true

**2. Freshdesk Integration:**

.. code-block:: bash

   orchestrator run customer_support_automation.yaml \
     --input ticketing_system="freshdesk" \
     --input freshdesk_domain="mycompany.freshdesk.com"

**3. Jira Service Desk:**

.. code-block:: bash

   orchestrator run customer_support_automation.yaml \
     --input ticketing_system="jira" \
     --input jira_project="SUPPORT"

Analytics and Insights
----------------------

Track key metrics:

- **Automation Rate**: Percentage of tickets handled automatically
- **Resolution Time**: Average time to resolve by category
- **Customer Satisfaction**: CSAT scores by automation status
- **Escalation Reasons**: Why tickets required human intervention
- **Agent Performance**: Efficiency gains from automation

Key Takeaways
-------------

This example demonstrates the power of Orchestrator's declarative framework:

1. **Zero Code Required**: Complete support system in pure YAML
2. **Intelligent Automation**: Smart decisions on when to automate
3. **Seamless Integration**: Works with existing ticketing systems
4. **Multi-Language Ready**: Global support capabilities
5. **Data-Driven**: Analytics for continuous improvement
6. **Human-Centric**: Escalates appropriately to maintain quality

The declarative approach makes sophisticated support automation accessible to all teams.

Next Steps
----------

- Try the :doc:`automated_testing_system` for QA automation
- Explore :doc:`interactive_chat_bot` for conversational AI
- Read the :doc:`../../advanced/ticketing_integration` guide
- Check the :doc:`../../user_guide/sentiment_analysis` guide