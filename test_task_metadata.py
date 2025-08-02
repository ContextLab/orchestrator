from src.orchestrator.core.task import Task

# Create a while loop task manually
loop_task = Task(
    id="loop",
    name="loop",
    action="control_flow",
    parameters={},
    metadata={
        "while": "true",
        "max_iterations": 1,
        "steps": [
            {
                "id": "task1",
                "action": "echo Hello"
            }
        ],
        "is_while_loop": True,
        "while_condition": "true"
    }
)

print(f"Task ID: {loop_task.id}")
print(f"Task action: {loop_task.action}")
print(f"Task metadata: {loop_task.metadata}")
print(f"Has 'steps' in metadata: {'steps' in loop_task.metadata}")
print(f"Number of steps: {len(loop_task.metadata.get('steps', []))}")
print(f"Is while loop: {loop_task.metadata.get('is_while_loop')}")