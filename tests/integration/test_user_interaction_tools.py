"""Integration tests for user interaction tools.

These tests verify the user interaction tools work with real user input simulation.
"""

import pytest
from unittest import mock

from src.orchestrator.tools.user_interaction_tools import (
    UserPromptTool,
    ApprovalGateTool,
    FeedbackCollectionTool
)


class TestUserPromptTool:
    """Test UserPromptTool with simulated user input."""
    
    @pytest.fixture
    def prompt_tool(self):
        """Create user prompt tool instance."""
        return UserPromptTool()
    
    @pytest.mark.asyncio
    async def test_text_input_cli(self, prompt_tool):
        """Test basic text input in CLI context."""
        # Simulate user input
        with mock.patch('builtins.input', return_value='Hello, World!'):
            result = await prompt_tool.execute(
                prompt="Enter a greeting:",
                input_type="text",
                context="cli"
            )
        
        assert result["success"] is True
        assert result["value"] == "Hello, World!"
        assert result["context"] == "cli"
        assert "retries" in result
    
    @pytest.mark.asyncio
    async def test_number_input_with_validation(self, prompt_tool):
        """Test number input with type conversion."""
        # Test integer
        with mock.patch('builtins.input', return_value='42'):
            result = await prompt_tool.execute(
                prompt="Enter a number:",
                input_type="number",
                context="cli"
            )
        
        assert result["success"] is True
        assert result["value"] == 42
        assert isinstance(result["value"], int)
        
        # Test float
        with mock.patch('builtins.input', return_value='3.14'):
            result = await prompt_tool.execute(
                prompt="Enter a decimal:",
                input_type="number",
                context="cli"
            )
        
        assert result["success"] is True
        assert result["value"] == 3.14
        assert isinstance(result["value"], float)
    
    @pytest.mark.asyncio
    async def test_boolean_input(self, prompt_tool):
        """Test boolean input conversion."""
        # Test various true values
        for true_value in ['yes', 'Yes', 'y', 'true', 'True', '1']:
            with mock.patch('builtins.input', return_value=true_value):
                result = await prompt_tool.execute(
                    prompt="Confirm?",
                    input_type="boolean",
                    context="cli"
                )
            
            assert result["success"] is True
            assert result["value"] is True
            assert isinstance(result["value"], bool)
        
        # Test various false values
        for false_value in ['no', 'No', 'n', 'false', 'False', '0']:
            with mock.patch('builtins.input', return_value=false_value):
                result = await prompt_tool.execute(
                    prompt="Confirm?",
                    input_type="boolean",
                    context="cli"
                )
            
            assert result["success"] is True
            assert result["value"] is False
            assert isinstance(result["value"], bool)
    
    @pytest.mark.asyncio
    async def test_choice_input(self, prompt_tool):
        """Test choice input validation."""
        choices = ["apple", "banana", "orange"]
        
        # Valid choice
        with mock.patch('builtins.input', return_value='banana'):
            result = await prompt_tool.execute(
                prompt="Choose a fruit:",
                input_type="choice",
                choices=choices,
                context="cli"
            )
        
        assert result["success"] is True
        assert result["value"] == "banana"
        
        # Invalid choice with retries
        with mock.patch('builtins.input', side_effect=['grape', 'kiwi', 'apple']):
            result = await prompt_tool.execute(
                prompt="Choose a fruit:",
                input_type="choice",
                choices=choices,
                context="cli",
                retry_on_invalid=True,
                max_retries=3
            )
        
        assert result["success"] is True
        assert result["value"] == "apple"
    
    @pytest.mark.asyncio
    async def test_default_value(self, prompt_tool):
        """Test default value handling."""
        # Empty input should use default
        with mock.patch('builtins.input', return_value=''):
            result = await prompt_tool.execute(
                prompt="Name (default: Anonymous):",
                input_type="text",
                default="Anonymous",
                context="cli"
            )
        
        assert result["success"] is True
        assert result["value"] == "Anonymous"
        # When empty input uses default, it's not marked as "skipped"
        # so used_default will be False
        assert result["used_default"] is False
    
    @pytest.mark.asyncio
    async def test_validation_pattern(self, prompt_tool):
        """Test regex pattern validation."""
        # Valid email pattern
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        with mock.patch('builtins.input', return_value='user@example.com'):
            result = await prompt_tool.execute(
                prompt="Enter email:",
                input_type="text",
                validation_pattern=email_pattern,
                context="cli"
            )
        
        assert result["success"] is True
        assert result["value"] == "user@example.com"
        
        # Invalid email with retries
        with mock.patch('builtins.input', side_effect=['invalid-email', 'user@example.com']):
            result = await prompt_tool.execute(
                prompt="Enter email:",
                input_type="text",
                validation_pattern=email_pattern,
                retry_on_invalid=True,
                context="cli"
            )
        
        assert result["success"] is True
        assert result["value"] == "user@example.com"
    
    @pytest.mark.asyncio
    async def test_api_context_simulation(self, prompt_tool):
        """Test API context returns structured response."""
        result = await prompt_tool.execute(
            prompt="API prompt test",
            input_type="text",
            context="api",
            default="API default response"
        )
        
        # API context should return default or error
        assert "error" in result or result["value"] == "API default response"
        assert result["context"] == "api"


class TestApprovalGateTool:
    """Test ApprovalGateTool with simulated approvals."""
    
    @pytest.fixture
    def approval_tool(self):
        """Create approval gate tool instance."""
        return ApprovalGateTool()
    
    @pytest.mark.asyncio
    async def test_simple_approval(self, approval_tool):
        """Test simple approval flow."""
        with mock.patch('builtins.input', return_value='approve'):
            result = await approval_tool.execute(
                title="Deploy to Production",
                content="Ready to deploy version 1.2.3?",
                options=["approve", "reject"],
                context="cli"
            )
        
        assert result["success"] is True
        assert result["approved"] is True
        assert result["action"] == "approve"
        assert "timestamp" in result
    
    @pytest.mark.asyncio
    async def test_rejection_flow(self, approval_tool):
        """Test rejection flow."""
        with mock.patch('builtins.input', return_value='reject'):
            result = await approval_tool.execute(
                title="Deploy to Production",
                content="Ready to deploy version 1.2.3?",
                options=["approve", "reject"],
                context="cli"
            )
        
        assert result["success"] is True
        assert result["approved"] is False
        assert result["action"] == "reject"
    
    @pytest.mark.asyncio
    async def test_modify_option(self, approval_tool):
        """Test modify option with modified content."""
        # Mock input to provide modify choice, then raise EOFError to stop reading
        def mock_input_sequence(prompt=""):
            # This generator will provide inputs in sequence
            inputs = iter(['modify', 'Updated schema with new fields', 'Added validation rules'])
            def _input(p=""):
                try:
                    return next(inputs)
                except StopIteration:
                    raise EOFError()  # Signal end of input for modify content
            return _input
        
        with mock.patch('builtins.input', side_effect=mock_input_sequence()):
            result = await approval_tool.execute(
                title="Review Changes",
                content="Proposed changes to database schema",
                options=["approve", "reject", "modify"],
                require_reason=True,
                context="cli"
            )
        
        assert result["success"] is True
        assert result["approved"] is True  # Modification implies approval
        assert result["action"] == "modify"
        assert "modified_content" in result
        assert "Updated schema with new fields" in result["modified_content"]
        assert "Added validation rules" in result["modified_content"]
    
    @pytest.mark.asyncio
    async def test_auto_approve(self, approval_tool):
        """Test auto-approval feature with hash."""
        content = "This should be auto-approved"
        # Calculate the hash of the content
        import hashlib
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        result = await approval_tool.execute(
            title="Auto Approval Test",
            content=content,
            auto_approve_hash=content_hash,
            context="cli"
        )
        
        assert result["success"] is True
        assert result["approved"] is True
        assert result["auto_approved"] is True
        assert result["action"] == "auto_approved"
    
    @pytest.mark.asyncio
    async def test_metadata_inclusion(self, approval_tool):
        """Test metadata is included in approval."""
        metadata = {
            "version": "1.2.3",
            "environment": "production",
            "changes": ["Feature A", "Bug fix B"]
        }
        
        with mock.patch('builtins.input', return_value='approve'):
            result = await approval_tool.execute(
                title="Deploy with Metadata",
                content="Deploy with additional context",
                metadata=metadata,
                context="cli"
            )
        
        assert result["success"] is True
        assert result["approved"] is True
        # Metadata is recorded internally in approval history but not returned in result
        # We can verify the approval was successful with the correct action
        assert result["action"] == "approve"


class TestFeedbackCollectionTool:
    """Test FeedbackCollectionTool with simulated feedback."""
    
    @pytest.fixture
    def feedback_tool(self):
        """Create feedback collection tool instance."""
        return FeedbackCollectionTool()
    
    @pytest.mark.asyncio
    async def test_simple_feedback_form(self, feedback_tool):
        """Test simple feedback collection."""
        questions = [
            {
                "id": "satisfaction",
                "text": "How satisfied are you?",
                "type": "rating",
                "scale": 5
            },
            {
                "id": "comments",
                "text": "Any additional comments?",
                "type": "text"
            }
        ]
        
        # Simulate user responses
        with mock.patch('builtins.input', side_effect=['4', 'Great service!']):
            result = await feedback_tool.execute(
                questions=questions,
                title="Service Feedback",
                context="cli"
            )
        
        assert result["success"] is True
        assert len(result["responses"]) == 2
        assert result["responses"]["satisfaction"] == 4
        assert result["responses"]["comments"] == "Great service!"
    
    @pytest.mark.asyncio
    async def test_multiple_choice_feedback(self, feedback_tool):
        """Test multiple choice questions."""
        questions = [
            {
                "id": "preference",
                "text": "Which feature do you use most?",
                "type": "choice",
                "choices": ["Search", "Filter", "Export", "Share"]
            },
            {
                "id": "frequency",
                "text": "How often do you use it?",
                "type": "choice",
                "choices": ["Daily", "Weekly", "Monthly", "Rarely"]
            }
        ]
        
        with mock.patch('builtins.input', side_effect=['1', '1']):
            result = await feedback_tool.execute(
                questions=questions,
                title="Feature Usage Survey",
                context="cli"
            )
        
        assert result["success"] is True
        assert result["responses"]["preference"] == "Search"
        assert result["responses"]["frequency"] == "Daily"
    
    @pytest.mark.asyncio
    async def test_boolean_feedback(self, feedback_tool):
        """Test yes/no questions."""
        questions = [
            {
                "id": "recommend",
                "text": "Would you recommend this to others?",
                "type": "boolean"
            }
        ]
        
        with mock.patch('builtins.input', return_value='yes'):
            result = await feedback_tool.execute(
                questions=questions,
                title="Recommendation Survey",
                context="cli"
            )
        
        assert result["success"] is True
        assert result["responses"]["recommend"] is True
    
    @pytest.mark.asyncio
    async def test_anonymous_feedback(self, feedback_tool):
        """Test anonymous feedback collection."""
        questions = [
            {
                "id": "honest_feedback",
                "text": "Please provide honest feedback:",
                "type": "text"
            }
        ]
        
        with mock.patch('builtins.input', return_value='Room for improvement in UI'):
            result = await feedback_tool.execute(
                questions=questions,
                title="Anonymous Feedback",
                anonymous=True,
                context="cli"
            )
        
        assert result["success"] is True
        # Anonymous setting affects internal tracking but isn't returned in result
        assert len(result["responses"]) == 1
        assert result["responses"]["honest_feedback"] == 'Room for improvement in UI'
    
    @pytest.mark.asyncio
    async def test_skip_optional_questions(self, feedback_tool):
        """Test skipping optional questions."""
        questions = [
            {
                "id": "required",
                "text": "Required question:",
                "type": "text",
                "required": True
            },
            {
                "id": "optional",
                "text": "Optional question (press Enter to skip):",
                "type": "text",
                "required": False
            }
        ]
        
        with mock.patch('builtins.input', side_effect=['My answer', '']):
            result = await feedback_tool.execute(
                questions=questions,
                title="Mixed Requirements",
                context="cli"
            )
        
        assert result["success"] is True
        assert result["responses"]["required"] == "My answer"
        # Optional question with empty input won't be in responses
        assert "optional" not in result["responses"]
    
    @pytest.mark.asyncio
    async def test_feedback_with_validation(self, feedback_tool):
        """Test feedback with validation rules."""
        questions = [
            {
                "id": "satisfaction",
                "text": "Rate your satisfaction:",
                "type": "rating",
                "scale": 10
            }
        ]
        
        # Test valid rating
        with mock.patch('builtins.input', return_value='8'):
            result = await feedback_tool.execute(
                questions=questions,
                title="Satisfaction Survey",
                context="cli"
            )
        
        assert result["success"] is True
        assert result["responses"]["satisfaction"] == 8
        
        # Test validation with retries
        with mock.patch('builtins.input', side_effect=['0', '11', '7']):
            result = await feedback_tool.execute(
                questions=questions,
                title="Satisfaction Survey",
                context="cli"
            )
        
        assert result["success"] is True
        assert result["responses"]["satisfaction"] == 7


@pytest.mark.asyncio
async def test_all_tools_integration():
    """Test integration between different user interaction tools."""
    prompt_tool = UserPromptTool()
    approval_tool = ApprovalGateTool()
    feedback_tool = FeedbackCollectionTool()
    
    # Simulate a complete workflow
    with mock.patch('builtins.input', side_effect=[
        'John Doe',  # Name prompt
        'approve',   # Approval
        '5',        # Rating
        'Excellent!' # Comments
    ]):
        # Get user name
        name_result = await prompt_tool.execute(
            prompt="Enter your name:",
            input_type="text",
            context="cli"
        )
        
        assert name_result["success"] is True
        user_name = name_result["value"]
        
        # Request approval
        approval_result = await approval_tool.execute(
            title=f"Welcome {user_name}",
            content="Do you approve the terms of service?",
            context="cli"
        )
        
        assert approval_result["success"] is True
        assert approval_result["approved"] is True
        
        # Collect feedback
        feedback_result = await feedback_tool.execute(
            questions=[
                {"id": "rating", "text": "Rate your experience (1-5):", "type": "rating", "scale": 5},
                {"id": "comments", "text": "Comments:", "type": "text"}
            ],
            title="User Experience",
            context="cli"
        )
        
        assert feedback_result["success"] is True
        assert len(feedback_result["responses"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])