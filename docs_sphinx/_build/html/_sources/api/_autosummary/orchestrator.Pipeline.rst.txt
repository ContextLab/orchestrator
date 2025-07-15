orchestrator.Pipeline
=====================

.. currentmodule:: orchestrator

.. autoclass:: Pipeline

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~Pipeline.__init__
      ~Pipeline.add_task
      ~Pipeline.clear_tasks
      ~Pipeline.from_dict
      ~Pipeline.get_completed_tasks
      ~Pipeline.get_critical_path
      ~Pipeline.get_dependencies
      ~Pipeline.get_dependents
      ~Pipeline.get_execution_levels
      ~Pipeline.get_execution_order
      ~Pipeline.get_execution_order_flat
      ~Pipeline.get_failed_tasks
      ~Pipeline.get_progress
      ~Pipeline.get_ready_task_ids
      ~Pipeline.get_ready_tasks
      ~Pipeline.get_running_tasks
      ~Pipeline.get_status
      ~Pipeline.get_task
      ~Pipeline.get_task_safe
      ~Pipeline.get_task_strict
      ~Pipeline.has_task
      ~Pipeline.is_complete
      ~Pipeline.is_failed
      ~Pipeline.is_valid
      ~Pipeline.remove_task
      ~Pipeline.remove_task_strict
      ~Pipeline.reset
      ~Pipeline.to_dict
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~Pipeline.description
      ~Pipeline.task_count
      ~Pipeline.version
      ~Pipeline.id
      ~Pipeline.name
      ~Pipeline.tasks
      ~Pipeline.context
      ~Pipeline.metadata
      ~Pipeline.created_at
   
   