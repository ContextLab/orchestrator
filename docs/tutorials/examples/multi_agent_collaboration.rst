Multi-Agent Collaboration
=========================

This example demonstrates how to build a sophisticated multi-agent system using the Orchestrator's declarative YAML framework. Multiple specialized AI agents collaborate to solve complex problems through coordination, communication, and emergent problem-solving - all defined in pure YAML with no custom Python code required.

.. note::
   **Level:** Advanced  
   **Duration:** 60-90 minutes  
   **Prerequisites:** Orchestrator framework installed, multiple API keys configured

Overview
--------

The Multi-Agent Collaboration system orchestrates:

1. **Agent Network Creation**: Initialize specialized agents with different roles
2. **Problem Decomposition**: Break complex problems into manageable sub-tasks
3. **Task Assignment**: Match agent capabilities to sub-problems
4. **Collaborative Solving**: Agents work together through multiple rounds
5. **Consensus Building**: Reach agreement through negotiation
6. **Conflict Resolution**: Handle disagreements constructively
7. **Solution Integration**: Combine partial solutions into unified whole
8. **Emergent Analysis**: Study patterns that emerge from collaboration

**Key Features Demonstrated:**
- Declarative YAML pipeline definition
- AUTO tag resolution for natural language task descriptions
- Dynamic agent creation and role assignment
- Multi-round collaboration with convergence checking
- Peer review and conflict resolution
- Emergent behavior analysis
- No Python code required

Quick Start
-----------

.. code-block:: bash

   # Set up environment variables
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   
   # Run the multi-agent system
   orchestrator run examples/multi_agent_collaboration.yaml \
     --input problem="Design a sustainable city for 1 million people" \
     --input num_agents=7 \
     --input max_rounds=15

Complete YAML Pipeline
----------------------

The complete pipeline is defined in ``examples/multi_agent_collaboration.yaml``. Here are the key sections:

**Pipeline Structure:**

.. code-block:: yaml

   name: "Multi-Agent Collaboration"
   description: "Collaborative multi-agent system for complex problem solving"

   inputs:
     problem:
       type: string
       description: "Complex problem to solve collaboratively"
       required: true
     
     num_agents:
       type: integer
       description: "Number of agents to create"
       default: 5
     
     max_rounds:
       type: integer
       description: "Maximum collaboration rounds"
       default: 10
     
     consensus_threshold:
       type: float
       description: "Threshold for solution consensus (0-1)"
       default: 0.85

**Key Pipeline Steps:**

1. **Agent Network Initialization:**

.. code-block:: yaml

   - id: initialize_agents
     action: <AUTO>create a network of {{num_agents}} specialized AI agents with roles:
       1. Researcher agents - gather information and conduct analysis
       2. Analyst agents - process data and identify patterns
       3. Creative agents - generate innovative solutions
       4. Critic agents - evaluate and identify issues
       5. Synthesizer agent - integrate and coordinate
       
       Create communication channels between agents.
       Return agent profiles and network topology</AUTO>

2. **Collaborative Problem Solving:**

.. code-block:: yaml

   - id: collaboration_round
     action: <AUTO>agents work on assigned tasks collaboratively:
       1. Each agent analyzes their assigned sub-problem
       2. Agents share insights through message passing
       3. Request help from peers when needed
       4. Build on each other's solutions
       5. Update shared knowledge base</AUTO>
     loop:
       max_iterations: "{{max_rounds}}"
       break_condition: "{{check_convergence.result.score}} >= {{consensus_threshold}}"

3. **Emergent Behavior Analysis:**

.. code-block:: yaml

   - id: analyze_emergence
     action: <AUTO>analyze emergent behaviors from agent collaboration:
       1. Communication patterns and clusters
       2. Information flow dynamics
       3. Decision-making patterns
       4. Self-organization indicators
       5. Collective intelligence metrics</AUTO>

How It Works
------------

**1. Agent Specialization**

The framework automatically creates agents with different cognitive styles:

- **Researchers**: High exploration, broad information gathering
- **Analysts**: Precise, data-driven, pattern recognition
- **Creatives**: High temperature, lateral thinking
- **Critics**: Low temperature, risk assessment
- **Synthesizers**: Balanced, integration focused

**2. Communication Protocols**

Agents communicate through:
- Direct messaging for specific requests
- Broadcast messages for announcements
- Shared knowledge base for persistent information
- Voting mechanisms for decisions

**3. Convergence Dynamics**

The system monitors convergence through:
- Solution similarity metrics
- Consensus scores
- Iteration efficiency
- Quality improvements

Running the Pipeline
--------------------

**Using the CLI:**

.. code-block:: bash

   # Basic multi-agent problem solving
   orchestrator run multi_agent_collaboration.yaml \
     --input problem="Develop a climate change mitigation strategy"

   # With more agents and rounds
   orchestrator run multi_agent_collaboration.yaml \
     --input problem="Design an AI governance framework" \
     --input num_agents=10 \
     --input max_rounds=20

   # Custom consensus threshold
   orchestrator run multi_agent_collaboration.yaml \
     --input problem="Create a universal healthcare system" \
     --input consensus_threshold=0.9

**Using Python SDK:**

.. code-block:: python

   from orchestrator import Orchestrator
   
   # Initialize orchestrator
   orchestrator = Orchestrator()
   
   # Run multi-agent collaboration
   result = await orchestrator.run_pipeline(
       "multi_agent_collaboration.yaml",
       inputs={
           "problem": "Design a mars colony infrastructure",
           "num_agents": 8,
           "max_rounds": 15,
           "agent_roles": "balanced"
       }
   )
   
   # Access results
   print(f"Solution quality: {result['outputs']['quality_score']}")
   print(f"Rounds needed: {result['outputs']['rounds_executed']}")
   print(f"Convergence achieved: {result['outputs']['convergence_achieved']}")

Example Output
--------------

**Console Output:**

.. code-block:: text

   🤝 Multi-Agent Collaboration
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ✓ initialize_agents: Created 7 agents (2 researchers, 2 analysts, 1 creative, 1 critic, 1 synthesizer) (3.2s)
   ✓ decompose_problem: Identified 5 sub-problems with dependencies (4.1s)
   ✓ assign_tasks: Distributed tasks based on agent capabilities (1.8s)
   ⟳ collaboration_round: Round 1/15...
     → Agents working on assigned tasks...
     → Inter-agent messages: 23
     → Knowledge base updates: 12
   ✓ check_convergence: Convergence score: 0.42 (2.3s)
   ✓ peer_review: 8 improvement suggestions generated (5.2s)
   ⟳ collaboration_round: Round 2/15...
     → Incorporating feedback...
     → Inter-agent messages: 31
     → Knowledge base updates: 18
   ✓ check_convergence: Convergence score: 0.67 (2.1s)
   ...
   ⟳ collaboration_round: Round 5/15...
   ✓ check_convergence: Convergence score: 0.86 - Threshold met! (1.9s)
   ✓ integrate_solutions: Unified solution created (6.8s)
   ✓ final_review: Quality score: 0.91/1.0 (3.4s)
   ✓ generate_report: Comprehensive report generated (4.2s)
   ✓ analyze_emergence: Identified 3 communication clusters (2.7s)
   
   ✅ Pipeline completed successfully in 89.3s
   📊 Convergence achieved in 5 rounds
   🎯 Solution quality: 0.91/1.0
   🤖 Agent efficiency: 0.87

**Solution Report Example:**

.. code-block:: markdown

   # Multi-Agent Solution: Sustainable City Design
   
   ## Executive Summary
   
   Through collaborative analysis, our agent network has designed a sustainable city 
   framework for 1 million residents, balancing environmental, social, and economic factors.
   
   ## Solution Architecture
   
   ### 1. Urban Planning (Researcher_1 + Analyst_1)
   - Mixed-use neighborhoods reducing commute times
   - Green corridors connecting all districts
   - Distributed energy generation nodes
   
   ### 2. Transportation (Analyst_2 + Creative_1)
   - Integrated public transit with 95% coverage
   - Bike-sharing and pedestrian priority zones
   - Electric vehicle infrastructure
   
   ### 3. Resource Management (Researcher_2 + Critic_1)
   - Closed-loop water recycling systems
   - Zero-waste initiatives with 90% diversion rate
   - Local food production via vertical farms
   
   ## Implementation Roadmap
   
   Phase 1 (Years 1-3): Infrastructure foundation
   Phase 2 (Years 4-7): System integration
   Phase 3 (Years 8-10): Optimization and scaling
   
   ## Emergent Insights
   
   - Agents spontaneously formed working groups
   - Creative-Critic pairs produced highest quality solutions
   - Information flow followed small-world network pattern

Advanced Features
-----------------

**1. Dynamic Role Assignment:**

.. code-block:: yaml

   - id: dynamic_roles
     action: <AUTO>adjust agent roles based on problem type:
       - Technical problems: More analysts and researchers
       - Creative challenges: More creative agents
       - Risk assessment: Additional critics
       - Complex integration: Multiple synthesizers</AUTO>
     condition: "{{agent_roles}} == 'auto'"

**2. Adaptive Consensus Building:**

.. code-block:: yaml

   - id: adaptive_consensus
     action: <AUTO>adjust consensus strategy based on convergence rate:
       - Slow convergence: Introduce mediator agents
       - Divergence: Break into smaller working groups
       - Deadlock: Use ranked voting system
       - Fast agreement: Increase quality thresholds</AUTO>

**3. Knowledge Persistence:**

.. code-block:: yaml

   - id: persist_knowledge
     action: <AUTO>save successful collaboration patterns:
       - Agent pairing effectiveness
       - Communication strategies
       - Problem decomposition approaches
       - Conflict resolution methods
       Store for future problem solving</AUTO>

Performance Optimization
------------------------

The pipeline includes several optimizations:

**1. Parallel Agent Execution**
- Agents work simultaneously on independent sub-problems
- Message passing is asynchronous
- Shared resources are lock-free

**2. Early Convergence Detection**
- Convergence checked after each round
- Pipeline terminates when consensus reached
- Avoids unnecessary iterations

**3. Intelligent Caching**
- Problem decompositions are cached
- Successful patterns are remembered
- Agent trust scores persist across runs

Error Handling
--------------

The system handles various failure scenarios:

**1. Agent Failures:**

.. code-block:: yaml

   on_error:
     action: <AUTO>redistribute failed agent's tasks to 
       available agents with similar capabilities</AUTO>
     continue_on_error: true

**2. Communication Breakdowns:**

.. code-block:: yaml

   on_error:
     action: <AUTO>switch to broadcast communication mode 
       and rebuild agent network connections</AUTO>
     retry_count: 3

**3. Convergence Failures:**

.. code-block:: yaml

   on_error:
     action: <AUTO>present best partial solutions with 
       confidence scores and unresolved issues</AUTO>
     fallback_value: "partial_solutions"

Real-World Applications
-----------------------

**1. Strategic Planning:**

.. code-block:: bash

   orchestrator run multi_agent_collaboration.yaml \
     --input problem="Develop 5-year digital transformation strategy" \
     --input num_agents=12

**2. Research Projects:**

.. code-block:: bash

   orchestrator run multi_agent_collaboration.yaml \
     --input problem="Design novel cancer treatment approach" \
     --input agent_roles="research-heavy"

**3. Policy Development:**

.. code-block:: bash

   orchestrator run multi_agent_collaboration.yaml \
     --input problem="Create comprehensive education reform policy" \
     --input consensus_threshold=0.95

Customization Examples
----------------------

**1. Domain-Specific Agents:**

.. code-block:: yaml

   - id: create_domain_agents
     action: <AUTO>create specialized agents for {{domain}}:
       - Medical: clinicians, researchers, ethicists
       - Finance: analysts, risk assessors, strategists
       - Engineering: designers, testers, integrators</AUTO>

**2. Hierarchical Organization:**

.. code-block:: yaml

   - id: hierarchical_setup
     action: <AUTO>organize agents hierarchically:
       - Team leads coordinate sub-groups
       - Specialists report to leads
       - Synthesizer acts as overall coordinator
       Enable both vertical and horizontal communication</AUTO>

**3. Competitive Collaboration:**

.. code-block:: yaml

   - id: competitive_mode
     action: <AUTO>split agents into competing teams:
       - Each team develops independent solution
       - Teams present and defend approaches
       - Best elements combined in final solution
       Foster innovation through competition</AUTO>

Monitoring and Analysis
-----------------------

Track collaboration metrics:

- **Communication Density**: Messages per agent per round
- **Convergence Velocity**: Rate of consensus building  
- **Knowledge Growth**: Unique insights generated
- **Efficiency Score**: Solution quality vs. resources used

Key Takeaways
-------------

This example demonstrates the power of Orchestrator's declarative framework:

1. **Zero Code Required**: Complete multi-agent system in pure YAML
2. **Emergent Intelligence**: Complex behaviors from simple rules
3. **Automatic Coordination**: Framework handles agent communication
4. **Flexible Architecture**: Easily adjust agent counts and roles
5. **Production Ready**: Robust error handling and monitoring
6. **Scalable Design**: Works with 3 to 100+ agents

The declarative approach makes sophisticated AI systems accessible without programming expertise.

Next Steps
----------

- Try the :doc:`content_creation_pipeline` for creative workflows
- Explore :doc:`code_analysis_suite` for software development
- Read the :doc:`../../advanced/agent_systems` guide
- Check the :doc:`../../user_guide/collaboration_patterns` guide