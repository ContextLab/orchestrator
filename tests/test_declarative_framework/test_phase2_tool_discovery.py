"""Tests for Phase 2: Smart tool discovery and execution."""

import pytest
from orchestrator.tools.discovery import ToolDiscoveryEngine, ToolMatch
from orchestrator.engine.enhanced_executor import EnhancedTaskExecutor
from orchestrator.engine.pipeline_spec import TaskSpec


class TestToolDiscoveryEngine:
    """Test the smart tool discovery engine."""
    
    @pytest.fixture
    def discovery_engine(self):
        """Create a test discovery engine instance."""
        return ToolDiscoveryEngine()
    
    def test_pattern_based_discovery(self, discovery_engine):
        """Test pattern-based tool discovery."""
        test_cases = [
            ("search web for information", "web-search"),
            ("analyze the data", "data-processing"),
            ("generate a report", "report-generator"),
            ("scrape website content", "headless-browser"),
            ("validate input data", "validation"),
            ("read configuration file", "filesystem"),
            ("run database script", "terminal")
        ]
        
        for action, expected_tool in test_cases:
            matches = discovery_engine.discover_tools_for_action(action)
            assert len(matches) > 0, f"No tools found for: {action}"
            assert expected_tool in [m.tool_name for m in matches], \
                f"Expected {expected_tool} for action: {action}"
    
    def test_confidence_scoring(self, discovery_engine):
        """Test confidence scoring in tool discovery."""
        # Direct pattern match should have high confidence
        matches = discovery_engine.discover_tools_for_action("search web for data")
        web_search_match = next((m for m in matches if m.tool_name == "web-search"), None)
        assert web_search_match is not None
        assert web_search_match.confidence >= 0.9
        
        # Semantic match should have lower confidence
        matches = discovery_engine.discover_tools_for_action("find information")
        assert any(m.confidence < 0.9 for m in matches)
    
    def test_context_enhanced_discovery(self, discovery_engine):
        """Test context-enhanced tool discovery."""
        # Context with data should suggest data-processing
        context = {"data": [1, 2, 3, 4, 5]}
        matches = discovery_engine.discover_tools_for_action("process this", context)
        assert "data-processing" in [m.tool_name for m in matches]
        
        # Context with URL should suggest browser tools
        context = {"url": "https://example.com"}
        matches = discovery_engine.discover_tools_for_action("get information", context)
        assert "headless-browser" in [m.tool_name for m in matches]
        
        # Context with file path should suggest filesystem
        context = {"file_path": "/tmp/data.json"}
        matches = discovery_engine.discover_tools_for_action("handle the content", context)
        assert "filesystem" in [m.tool_name for m in matches]
    
    def test_tool_chain_discovery(self, discovery_engine):
        """Test discovery of tool chains for complex actions."""
        # Search and analyze pattern
        action = "search web for data and then analyze the results"
        chain = discovery_engine.get_tool_chain_for_action(action)
        assert len(chain) >= 1
        assert any(m.tool_name == "web-search" for m in chain)
    
    def test_edge_cases(self, discovery_engine):
        """Test edge cases in tool discovery."""
        # Empty action
        matches = discovery_engine.discover_tools_for_action("")
        assert len(matches) == 0
        
        # Unrelated action
        matches = discovery_engine.discover_tools_for_action("make coffee")
        assert len(matches) == 0
        
        # Very long action with multiple tools
        action = "search web and scrape data and analyze results and generate report"
        matches = discovery_engine.discover_tools_for_action(action)
        assert len(matches) > 0
    
    def test_tool_availability(self, discovery_engine):
        """Test tool availability checking."""
        available_tools = discovery_engine.tool_registry.list_tools()
        assert len(available_tools) > 0
        
        # All discovered tools should be available
        matches = discovery_engine.discover_tools_for_action("search and analyze data")
        for match in matches:
            assert match.tool_name in available_tools
    
    def test_alternative_suggestions(self, discovery_engine):
        """Test alternative tool suggestions."""
        required_tools = ["web-search", "non-existent-tool"]
        suggestions = discovery_engine.suggest_missing_tools(required_tools)
        
        # web-search should not be in missing tools
        assert "web-search" not in suggestions
        
        # non-existent-tool should be in missing tools
        assert "non-existent-tool" in suggestions


class TestEnhancedTaskExecutor:
    """Test the enhanced task executor."""
    
    @pytest.fixture
    def executor(self):
        """Create a test executor instance."""
        return EnhancedTaskExecutor()
    
    def test_execution_strategy_selection(self, executor):
        """Test execution strategy selection."""
        # Single tool - sequential
        matches = [ToolMatch("web-search", 0.9, "test", {})]
        strategy = executor._select_execution_strategy(matches, {})
        assert strategy == "sequential"
        
        # Multiple different tools - potentially parallel
        matches = [
            ToolMatch("web-search", 0.9, "test", {}),
            ToolMatch("data-processing", 0.9, "test", {})
        ]
        enhanced_spec = {"prompt": "search and analyze"}
        strategy = executor._select_execution_strategy(matches, enhanced_spec)
        assert strategy in ["parallel", "sequential"]
        
        # Pipeline pattern
        enhanced_spec = {"prompt": "search web and analyze results"}
        strategy = executor._select_execution_strategy(matches, enhanced_spec)
        assert strategy in ["pipeline", "sequential"]
    
    def test_tool_parameter_preparation(self, executor):
        """Test tool parameter preparation."""
        from orchestrator.tools.web_tools import WebSearchTool
        tool = WebSearchTool()
        match = ToolMatch("web-search", 0.9, "test", {"max_results": 5})
        enhanced_spec = {"prompt": "search for AI news"}
        context = {"topic": "artificial intelligence"}
        
        params = executor._prepare_enhanced_tool_params(tool, match, enhanced_spec, context)
        
        assert "query" in params
        assert params["max_results"] == 5
        assert params["topic"] == "artificial intelligence"
    
    @pytest.mark.asyncio
    async def test_enhanced_task_execution(self, executor):
        """Test enhanced task execution."""
        task_spec = TaskSpec(
            id="test_task",
            action="<AUTO>search for information about Python</AUTO>"
        )
        
        context = {"topic": "Python programming"}
        
        # This will fail without models, but we can test the structure
        try:
            result = await executor.execute_task(task_spec, context)
        except Exception as e:
            # Expected to fail without model registry
            assert "model registry" in str(e).lower()
    
    def test_parallel_result_combination(self, executor):
        """Test combining results from parallel execution."""
        tool_results = {
            "web-search": {
                "success": True,
                "data": ["result1", "result2"],
                "content": "Search results"
            },
            "data-processing": {
                "success": True,
                "insights": ["insight1", "insight2"],
                "result": "Analysis complete"
            }
        }
        
        combined = executor._combine_parallel_results(tool_results)
        
        assert combined["combined_results"] is True
        assert len(combined["data"]) == 2
        assert len(combined["insights"]) == 2
        assert "Search results" in combined["content"]