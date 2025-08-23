Utilities API
=============

The orchestrator framework provides several utility modules to enhance pipeline functionality and output quality.

OutputSanitizer
--------------

.. automodule:: orchestrator.utils.output_sanitizer
   :members:
   :undoc-members:
   :show-inheritance:

The OutputSanitizer removes conversational markers and unnecessary content from AI model outputs.

Usage Examples
~~~~~~~~~~~~~~

Basic usage::

    from orchestrator.utils.output_sanitizer import OutputSanitizer, sanitize_output
    
    # Using the class directly
    sanitizer = OutputSanitizer(enabled=True)
    clean_text = sanitizer.sanitize("Certainly! Here is the content you requested: The actual content.")
    # Result: "The actual content."
    
    # Using the convenience function
    clean_text = sanitize_output("Sure! I'd be happy to help. The answer is 42.")
    # Result: "The answer is 42."

Features
~~~~~~~~

The OutputSanitizer provides comprehensive cleaning of AI outputs by removing:

* **Conversational starters**: "Certainly!", "Sure!", "Of course!", etc.
* **Conversational endings**: "Let me know if you need help", "Hope this helps", etc.
* **Meta-commentary**: "I'll create...", "I'm going to...", etc.
* **Conversational fillers**: "Well,", "So,", "Actually,", etc.

Configuration
~~~~~~~~~~~~~

The sanitizer can be configured globally or per-instance::

    from orchestrator.utils.output_sanitizer import configure_sanitizer, OutputSanitizer
    
    # Global configuration
    configure_sanitizer(
        enabled=True,
        custom_patterns={
            "starter": [r"^My response:\s*"],
            "ending": [r"\s*End of response\.\s*$"]
        }
    )
    
    # Per-instance configuration
    sanitizer = OutputSanitizer(enabled=True)
    sanitizer.add_custom_pattern(r"^Custom pattern:\s*", "starter")

API Reference
~~~~~~~~~~~~

.. class:: OutputSanitizer(enabled=True)

   Main sanitizer class for removing conversational content from text.
   
   :param enabled: Whether sanitization is active
   :type enabled: bool

   .. method:: sanitize(text: Union[str, Dict, List]) -> Union[str, Dict, List]
   
      Sanitize input text by removing conversational markers.
      
      :param text: Input to sanitize (only strings are processed)
      :type text: Union[str, Dict, List]
      :returns: Sanitized output with conversational content removed
      :rtype: Union[str, Dict, List]

   .. method:: sanitize_batch(texts: List[str]) -> List[str]
   
      Sanitize multiple texts in batch.
      
      :param texts: List of texts to sanitize
      :type texts: List[str]
      :returns: List of sanitized texts
      :rtype: List[str]

   .. method:: add_custom_pattern(pattern: str, pattern_type: str = "starter")
   
      Add custom pattern to remove from outputs.
      
      :param pattern: Regex pattern to remove
      :type pattern: str
      :param pattern_type: Pattern category ("starter", "ending", "filler", "meta")
      :type pattern_type: str

   .. method:: set_enabled(enabled: bool)
   
      Enable or disable the sanitizer.
      
      :param enabled: Whether to enable sanitization
      :type enabled: bool

   .. method:: is_enabled() -> bool
   
      Check if sanitizer is enabled.
      
      :returns: True if enabled
      :rtype: bool

.. function:: sanitize_output(text: Union[str, Dict, List], enabled: bool = True) -> Union[str, Dict, List]

   Convenience function using the default sanitizer instance.
   
   :param text: Text to sanitize
   :type text: Union[str, Dict, List]
   :param enabled: Whether to apply sanitization
   :type enabled: bool
   :returns: Sanitized text
   :rtype: Union[str, Dict, List]

.. function:: configure_sanitizer(enabled: bool = True, custom_patterns: Optional[Dict[str, List[str]]] = None)

   Configure the global default sanitizer.
   
   :param enabled: Enable/disable sanitization
   :type enabled: bool
   :param custom_patterns: Dictionary of pattern types to pattern lists
   :type custom_patterns: Optional[Dict[str, List[str]]]

Service Manager
--------------

.. automodule:: orchestrator.utils.service_manager
   :members:
   :undoc-members:
   :show-inheritance:

Model Configuration Loader
--------------------------

.. automodule:: orchestrator.utils.model_config_loader
   :members:
   :undoc-members:
   :show-inheritance:

API Keys Management
------------------

.. automodule:: orchestrator.utils.api_keys
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: orchestrator.utils.api_keys_flexible
   :members:
   :undoc-members:
   :show-inheritance:

Auto Installation
----------------

.. automodule:: orchestrator.utils.auto_install
   :members:
   :undoc-members:
   :show-inheritance: