"""User interaction tools for human-in-the-loop workflows."""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import hashlib

from .base import Tool


@dataclass
class UserResponse:
    """Container for user response data."""

    value: Any
    timestamp: float = field(default_factory=time.time)
    response_time: float = 0.0
    skipped: bool = False
    approved: bool = True


@dataclass
class ApprovalRequest:
    """Request for user approval."""

    title: str
    content: str
    options: List[str] = field(default_factory=lambda: ["approve", "reject", "modify"])
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackForm:
    """Structured feedback form."""

    questions: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    anonymous: bool = False


class UserPromptTool(Tool):
    """Get input from users with support for different contexts."""

    def __init__(self):
        super().__init__(
            name="user-prompt",
            description="Prompt users for input with support for defaults and timeouts",
        )
        self.add_parameter("prompt", "string", "The prompt to display to the user")
        self.add_parameter(
            "input_type",
            "string",
            "Type of input: text, number, boolean, choice",
            default="text",
        )
        self.add_parameter(
            "default", "any", "Default value if no input provided", required=False
        )
        self.add_parameter(
            "choices",
            "array",
            "Available choices for 'choice' input type",
            required=False,
        )
        self.add_parameter(
            "timeout", "number", "Timeout in seconds (0 for no timeout)", default=0
        )
        self.add_parameter(
            "context", "string", "Execution context: cli, gui, api", default="cli"
        )
        self.add_parameter(
            "validation_pattern",
            "string",
            "Regex pattern for input validation",
            required=False,
        )
        self.add_parameter(
            "retry_on_invalid", "boolean", "Retry if validation fails", default=True
        )
        self.add_parameter(
            "max_retries", "integer", "Maximum validation retries", default=3
        )

        self.logger = logging.getLogger(__name__)

    def _validate_input(self, value: str, pattern: str) -> bool:
        """Validate input against pattern."""
        import re

        try:
            return bool(re.match(pattern, value))
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False

    def _convert_input(self, value: str, input_type: str) -> Any:
        """Convert input string to appropriate type."""
        if input_type == "number":
            try:
                return float(value) if "." in value else int(value)
            except ValueError:
                raise ValueError(f"Invalid number: {value}")

        elif input_type == "boolean":
            lower_value = value.lower()
            if lower_value in ("true", "yes", "y", "1"):
                return True
            elif lower_value in ("false", "no", "n", "0"):
                return False
            else:
                raise ValueError(f"Invalid boolean: {value}")

        elif input_type == "choice":
            return value  # Validation happens separately

        else:  # text
            return value

    async def _get_cli_input(
        self,
        prompt: str,
        input_type: str,
        default: Optional[Any],
        choices: Optional[List[str]],
        timeout: float,
    ) -> UserResponse:
        """Get input from CLI."""
        start_time = time.time()

        # Build prompt message
        prompt_msg = prompt
        if choices:
            prompt_msg += f"\nChoices: {', '.join(choices)}"
        if default is not None:
            prompt_msg += f" [{default}]"
        prompt_msg += ": "

        if timeout > 0:
            # Use asyncio with timeout
            try:
                # Create a future for the input
                loop = asyncio.get_event_loop()
                future = loop.create_future()

                def get_input():
                    try:
                        result = input(prompt_msg)
                        loop.call_soon_threadsafe(future.set_result, result)
                    except Exception as e:
                        loop.call_soon_threadsafe(future.set_exception, e)

                # Run input in thread
                import threading

                thread = threading.Thread(target=get_input)
                thread.daemon = True
                thread.start()

                # Wait with timeout
                value = await asyncio.wait_for(future, timeout=timeout)

            except asyncio.TimeoutError:
                print(f"\nTimeout! Using default value: {default}")
                return UserResponse(value=default, response_time=timeout, skipped=True)
        else:
            # No timeout - direct input
            value = input(prompt_msg)

        # Use default if empty
        if not value and default is not None:
            value = default

        response_time = time.time() - start_time
        return UserResponse(value=value, response_time=response_time)

    async def _get_gui_input(
        self,
        prompt: str,
        input_type: str,
        default: Optional[Any],
        choices: Optional[List[str]],
        timeout: float,
    ) -> UserResponse:
        """Get input from GUI (simulated for now)."""
        # In a real implementation, this would integrate with a GUI framework
        self.logger.info("GUI input requested - falling back to CLI")
        return await self._get_cli_input(prompt, input_type, default, choices, timeout)

    async def _get_api_input(
        self,
        prompt: str,
        input_type: str,
        default: Optional[Any],
        choices: Optional[List[str]],
        timeout: float,
    ) -> UserResponse:
        """Get input from API context."""
        # In API context, we typically use the default or return a pending status
        self.logger.info(f"API context prompt: {prompt}")

        if default is not None:
            return UserResponse(value=default, skipped=True)

        # Return a special response indicating manual intervention needed
        return UserResponse(
            value={
                "status": "pending_user_input",
                "prompt": prompt,
                "input_type": input_type,
                "choices": choices,
            },
            skipped=True,
        )

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute user prompt."""
        prompt = kwargs["prompt"]
        input_type = kwargs.get("input_type", "text")
        default = kwargs.get("default")
        choices = kwargs.get("choices")
        timeout = kwargs.get("timeout", 0)
        context = kwargs.get("context", "cli")
        validation_pattern = kwargs.get("validation_pattern")
        retry_on_invalid = kwargs.get("retry_on_invalid", True)
        max_retries = kwargs.get("max_retries", 3)

        # Validate input_type
        valid_types = ["text", "number", "boolean", "choice"]
        if input_type not in valid_types:
            return {
                "success": False,
                "error": f"Invalid input_type: {input_type}. Must be one of {valid_types}",
            }

        # Validate choices for choice type
        if input_type == "choice" and not choices:
            return {
                "success": False,
                "error": "Choices must be provided for 'choice' input type",
            }

        retries = 0
        while retries <= max_retries:
            try:
                # Get input based on context
                if context == "cli":
                    response = await self._get_cli_input(
                        prompt, input_type, default, choices, timeout
                    )
                elif context == "gui":
                    response = await self._get_gui_input(
                        prompt, input_type, default, choices, timeout
                    )
                elif context == "api":
                    response = await self._get_api_input(
                        prompt, input_type, default, choices, timeout
                    )
                else:
                    return {
                        "success": False,
                        "error": f"Invalid context: {context}. Must be cli, gui, or api",
                    }

                # Convert and validate
                if not response.skipped:
                    # Convert type
                    try:
                        converted_value = self._convert_input(
                            str(response.value), input_type
                        )
                        response.value = converted_value
                    except ValueError as e:
                        if retry_on_invalid and retries < max_retries:
                            print(f"Invalid input: {e}. Please try again.")
                            retries += 1
                            continue
                        else:
                            return {
                                "success": False,
                                "error": str(e),
                                "retries": retries,
                            }

                    # Validate pattern
                    if validation_pattern and not self._validate_input(
                        str(response.value), validation_pattern
                    ):
                        if retry_on_invalid and retries < max_retries:
                            print(
                                "Input does not match required pattern. Please try again."
                            )
                            retries += 1
                            continue
                        else:
                            return {
                                "success": False,
                                "error": "Input validation failed",
                                "pattern": validation_pattern,
                                "retries": retries,
                            }

                    # Validate choice
                    if input_type == "choice" and response.value not in choices:
                        if retry_on_invalid and retries < max_retries:
                            print(
                                f"Invalid choice. Please select from: {', '.join(choices)}"
                            )
                            retries += 1
                            continue
                        else:
                            return {
                                "success": False,
                                "error": f"Invalid choice: {response.value}",
                                "valid_choices": choices,
                                "retries": retries,
                            }

                # Success!
                return {
                    "success": True,
                    "value": response.value,
                    "response_time": response.response_time,
                    "skipped": response.skipped,
                    "used_default": response.skipped and default is not None,
                    "context": context,
                    "retries": retries,
                }

            except Exception as e:
                self.logger.error(f"Error getting user input: {e}")
                return {"success": False, "error": str(e), "context": context}


class ApprovalGateTool(Tool):
    """Require user approval before proceeding."""

    def __init__(self):
        super().__init__(
            name="approval-gate",
            description="Present results for user review and approval",
        )
        self.add_parameter("title", "string", "Title of the approval request")
        self.add_parameter("content", "string", "Content to review")
        self.add_parameter(
            "format", "string", "Content format: text, json, markdown", default="text"
        )
        self.add_parameter(
            "allow_modifications",
            "boolean",
            "Allow user to modify content",
            default=True,
        )
        self.add_parameter(
            "require_reason", "boolean", "Require reason for rejection", default=True
        )
        self.add_parameter(
            "context", "string", "Execution context: cli, gui, api", default="cli"
        )
        self.add_parameter(
            "auto_approve_hash",
            "string",
            "Auto-approve if content hash matches",
            required=False,
        )
        self.add_parameter(
            "metadata", "object", "Additional metadata for the approval", required=False
        )

        self.logger = logging.getLogger(__name__)

        # Store approval history
        self.approval_history: List[Dict[str, Any]] = []

    def _hash_content(self, content: str) -> str:
        """Generate hash of content for comparison."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _format_content(self, content: str, format: str) -> str:
        """Format content for display."""
        if format == "json":
            try:
                parsed = json.loads(content)
                return json.dumps(parsed, indent=2)
            except Exception:
                return content
        elif format == "markdown":
            # Basic markdown rendering for CLI
            lines = content.split("\n")
            formatted = []
            for line in lines:
                if line.startswith("# "):
                    formatted.append("\n" + "=" * 40)
                    formatted.append(line[2:].upper())
                    formatted.append("=" * 40)
                elif line.startswith("## "):
                    formatted.append("\n" + line[3:])
                    formatted.append("-" * len(line[3:]))
                else:
                    formatted.append(line)
            return "\n".join(formatted)
        else:
            return content

    async def _get_cli_approval(
        self, request: ApprovalRequest, allow_modifications: bool, require_reason: bool
    ) -> Dict[str, Any]:
        """Get approval from CLI."""
        print("\n" + "=" * 60)
        print(f"APPROVAL REQUEST: {request.title}")
        print("=" * 60)
        print(request.content)
        print("=" * 60)

        # Get user choice
        options_str = "/".join(request.options)
        choice = input(f"\nAction ({options_str}): ").lower().strip()

        if choice not in request.options:
            choice = "reject"  # Default to reject on invalid input

        result = {
            "approved": choice == "approve",
            "action": choice,
            "modified_content": None,
            "reason": None,
        }

        # Handle modifications
        if choice == "modify" and allow_modifications:
            print("\nEnter modified content (press Ctrl+D or Ctrl+Z when done):")
            modified_lines = []
            try:
                while True:
                    modified_lines.append(input())
            except EOFError:
                pass
            result["modified_content"] = "\n".join(modified_lines)
            result["approved"] = True  # Modification implies approval

        # Get rejection reason
        if choice == "reject" and require_reason:
            result["reason"] = input("Reason for rejection: ")

        return result

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute approval gate."""
        title = kwargs["title"]
        content = kwargs["content"]
        format = kwargs.get("format", "text")
        allow_modifications = kwargs.get("allow_modifications", True)
        require_reason = kwargs.get("require_reason", True)
        context = kwargs.get("context", "cli")
        auto_approve_hash = kwargs.get("auto_approve_hash")
        metadata = kwargs.get("metadata", {})

        # Check auto-approval
        content_hash = self._hash_content(content)
        if auto_approve_hash and content_hash == auto_approve_hash:
            self.logger.info("Auto-approving content with matching hash")
            return {
                "success": True,
                "approved": True,
                "action": "auto_approved",
                "content_hash": content_hash,
                "auto_approved": True,
            }

        # Format content
        formatted_content = self._format_content(content, format)

        # Create approval request
        request = ApprovalRequest(
            title=title, content=formatted_content, metadata=metadata
        )

        if allow_modifications and "modify" not in request.options:
            request.options = ["approve", "reject", "modify"]

        try:
            # Get approval based on context
            if context == "cli":
                result = await self._get_cli_approval(
                    request, allow_modifications, require_reason
                )
            elif context == "gui":
                # Fallback to CLI for now
                self.logger.info("GUI approval requested - falling back to CLI")
                result = await self._get_cli_approval(
                    request, allow_modifications, require_reason
                )
            elif context == "api":
                # In API context, return pending status
                return {
                    "success": True,
                    "approved": False,
                    "status": "pending_approval",
                    "approval_id": content_hash,
                    "content_hash": content_hash,
                    "request": {
                        "title": title,
                        "content": content,
                        "format": format,
                        "allow_modifications": allow_modifications,
                    },
                }
            else:
                return {"success": False, "error": f"Invalid context: {context}"}

            # Record in history
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "title": title,
                "content_hash": content_hash,
                "approved": result["approved"],
                "action": result["action"],
                "metadata": metadata,
            }
            if result.get("reason"):
                history_entry["reason"] = result["reason"]

            self.approval_history.append(history_entry)

            # Return result
            response = {
                "success": True,
                "approved": result["approved"],
                "action": result["action"],
                "content_hash": content_hash,
                "timestamp": history_entry["timestamp"],
            }

            if result.get("modified_content"):
                response["modified_content"] = result["modified_content"]
                response["modified_hash"] = self._hash_content(
                    result["modified_content"]
                )

            if result.get("reason"):
                response["rejection_reason"] = result["reason"]

            return response

        except Exception as e:
            self.logger.error(f"Error in approval gate: {e}")
            return {"success": False, "error": str(e), "approved": False}

    def get_approval_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent approval history."""
        return self.approval_history[-limit:]


class FeedbackCollectionTool(Tool):
    """Gather structured feedback from users."""

    def __init__(self):
        super().__init__(
            name="feedback-collection",
            description="Collect structured feedback with various question types",
        )
        self.add_parameter("title", "string", "Feedback form title")
        self.add_parameter("questions", "array", "List of feedback questions")
        self.add_parameter(
            "anonymous", "boolean", "Allow anonymous feedback", default=False
        )
        self.add_parameter(
            "required_questions", "array", "IDs of required questions", default=[]
        )
        self.add_parameter(
            "context", "string", "Execution context: cli, gui, api", default="cli"
        )
        self.add_parameter(
            "save_to_file", "string", "File path to save feedback", required=False
        )

        self.logger = logging.getLogger(__name__)
        self.feedback_store: List[Dict[str, Any]] = []

    def _validate_question(self, question: Dict[str, Any]) -> bool:
        """Validate question structure."""
        required_fields = ["id", "text", "type"]
        return all(field in question for field in required_fields)

    async def _collect_cli_feedback(
        self, form: FeedbackForm, required_questions: List[str]
    ) -> Dict[str, Any]:
        """Collect feedback from CLI."""
        print("\n" + "=" * 60)
        print(f"FEEDBACK FORM: {form.questions[0].get('_title', 'Feedback')}")
        print("=" * 60)

        if form.anonymous:
            print("This feedback will be collected anonymously.\n")

        responses = {}

        for question in form.questions:
            if not self._validate_question(question):
                continue

            q_id = question["id"]
            q_text = question["text"]
            q_type = question["type"]
            required = q_id in required_questions

            # Add required indicator
            if required:
                q_text += " *"

            # Get response based on type
            if q_type == "rating":
                scale = question.get("scale", 5)
                while True:
                    try:
                        rating = input(f"{q_text} (1-{scale}): ")
                        if not rating and not required:
                            break
                        rating_int = int(rating)
                        if 1 <= rating_int <= scale:
                            responses[q_id] = rating_int
                            break
                        else:
                            print(f"Please enter a number between 1 and {scale}")
                    except ValueError:
                        print("Please enter a valid number")

            elif q_type == "choice":
                choices = question.get("choices", [])
                print(f"{q_text}")
                for i, choice in enumerate(choices, 1):
                    print(f"  {i}. {choice}")

                while True:
                    choice_input = input("Choice (number): ")
                    if not choice_input and not required:
                        break
                    try:
                        choice_idx = int(choice_input) - 1
                        if 0 <= choice_idx < len(choices):
                            responses[q_id] = choices[choice_idx]
                            break
                        else:
                            print(f"Please enter a number between 1 and {len(choices)}")
                    except ValueError:
                        print("Please enter a valid number")

            elif q_type == "boolean":
                while True:
                    bool_input = input(f"{q_text} (yes/no): ").lower()
                    if not bool_input and not required:
                        break
                    if bool_input in ("yes", "y", "true", "1"):
                        responses[q_id] = True
                        break
                    elif bool_input in ("no", "n", "false", "0"):
                        responses[q_id] = False
                        break
                    else:
                        print("Please enter yes or no")

            else:  # text
                text_input = input(f"{q_text}: ")
                if text_input or required:
                    responses[q_id] = text_input

        print("\nThank you for your feedback!")
        return responses

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute feedback collection."""
        title = kwargs["title"]
        questions = kwargs["questions"]
        anonymous = kwargs.get("anonymous", False)
        required_questions = kwargs.get("required_questions", [])
        context = kwargs.get("context", "cli")
        save_to_file = kwargs.get("save_to_file")

        # Validate questions
        if not questions:
            return {"success": False, "error": "No questions provided"}

        # Add title to first question for display
        questions[0]["_title"] = title

        # Create feedback form
        form = FeedbackForm(
            questions=questions, anonymous=anonymous, metadata={"title": title}
        )

        try:
            # Collect feedback based on context
            if context == "cli":
                responses = await self._collect_cli_feedback(form, required_questions)
            elif context == "gui":
                # Fallback to CLI
                self.logger.info("GUI feedback requested - falling back to CLI")
                responses = await self._collect_cli_feedback(form, required_questions)
            elif context == "api":
                # Return form structure for API
                return {
                    "success": True,
                    "status": "form_created",
                    "form_id": hashlib.md5(title.encode()).hexdigest()[:8],
                    "form": {
                        "title": title,
                        "questions": questions,
                        "anonymous": anonymous,
                        "required_questions": required_questions,
                    },
                }
            else:
                return {"success": False, "error": f"Invalid context: {context}"}

            # Create feedback entry
            feedback_entry = {
                "timestamp": datetime.now().isoformat(),
                "title": title,
                "anonymous": anonymous,
                "responses": responses,
                "response_count": len(responses),
                "completion_rate": len(responses) / len(questions) * 100,
            }

            if not anonymous:
                feedback_entry["user"] = os.environ.get("USER", "unknown")

            # Store feedback
            self.feedback_store.append(feedback_entry)

            # Save to file if requested
            if save_to_file:
                try:
                    with open(save_to_file, "a") as f:
                        f.write(json.dumps(feedback_entry) + "\n")
                    self.logger.info(f"Feedback saved to {save_to_file}")
                except Exception as e:
                    self.logger.error(f"Failed to save feedback: {e}")

            # Calculate summary statistics
            summary = {
                "rating_average": None,
                "boolean_summary": {},
                "choice_summary": {},
            }

            # Process responses for summary
            ratings = []
            for q in questions:
                q_id = q["id"]
                if q_id in responses:
                    if q["type"] == "rating":
                        ratings.append(responses[q_id])
                    elif q["type"] == "boolean":
                        summary["boolean_summary"][q_id] = responses[q_id]
                    elif q["type"] == "choice":
                        summary["choice_summary"][q_id] = responses[q_id]

            if ratings:
                summary["rating_average"] = sum(ratings) / len(ratings)

            return {
                "success": True,
                "feedback_id": feedback_entry["timestamp"],
                "responses": responses,
                "response_count": len(responses),
                "completion_rate": feedback_entry["completion_rate"],
                "summary": summary,
                "saved_to_file": save_to_file is not None,
            }

        except Exception as e:
            self.logger.error(f"Error collecting feedback: {e}")
            return {"success": False, "error": str(e)}

    def get_feedback_summary(self, limit: int = 10) -> Dict[str, Any]:
        """Get summary of collected feedback."""
        recent_feedback = self.feedback_store[-limit:]

        if not recent_feedback:
            return {"total_responses": 0, "average_completion_rate": 0}

        total_responses = len(recent_feedback)
        avg_completion = (
            sum(f["completion_rate"] for f in recent_feedback) / total_responses
        )

        return {
            "total_responses": total_responses,
            "average_completion_rate": avg_completion,
            "recent_feedback": recent_feedback,
        }
