from src.orchestrator.compiler.yaml_compiler import YAMLCompiler

yaml_content = '''
name: test_simple_while
description: Simple test

steps:
  - id: init
    action: echo Starting
    
  - id: loop
    while: "true"
    max_iterations: 1
    steps:
      - id: task1
        action: echo Hello
        
  - id: done
    action: echo Done
    dependencies: [loop]
'''

compiler = YAMLCompiler(model_registry=None)
pipeline = compiler.compile(yaml_content, {})

print("Pipeline tasks:")
for task_id, task in pipeline.tasks.items():
    print(f"  {task_id}: action={task.action}, metadata={task.metadata}")
    
# Check loop task specifically
loop_task = pipeline.get_task("loop")
if loop_task:
    print(f"\nLoop task metadata: {loop_task.metadata}")
    print(f"Has 'steps' in metadata: {'steps' in loop_task.metadata}")
    if 'steps' in loop_task.metadata:
        print(f"Number of steps: {len(loop_task.metadata['steps'])}")
        print(f"Steps: {loop_task.metadata['steps']}")