# YAML Examples Quality Test Report

**Test Date:** 2025-07-17 08:09:15

**Models Used:** anthropic:claude-3-5-sonnet-20241022, anthropic:claude-3-haiku-20240307, openai:gpt-4, openai:gpt-3.5-turbo, google:gemini-pro

## Summary

- **Total Examples Tested:** 3
- **Successful:** 3
- **Failed:** 0
- **Success Rate:** 100.0%

- **Average Quality Score:** 0.00/1.0

## Detailed Results

### research_assistant.yaml

**Status:** ✅ Success
**Execution Time:** 51.33s
**Quality Score:** 0.00/1.0

**Quality Checks:**
- ❌ has_summary
- ❌ has_sources
- ❌ has_key_findings
- ❌ summary_length
- ❌ multiple_sources

**Sample Outputs:**
```json
{
  "analyze_query": "Artificial Intelligence (AI) has shown promising results in the healthcare sector. Some of the successful applications include:\n\n1. Disease Identification/Diagnosis: AI can be used to analyze complex ...",
  "web_search": "Previous results are available for review.",
  "extract_content": "This phrase is a directive usually used in a context where some data or information has been gathered earlier and is now ready for review or use. It could be used in various settings such as in resear...",
  "...": "(6 more outputs)"
}
```

### data_processing_workflow.yaml

**Status:** ✅ Success
**Execution Time:** 19.53s
**Quality Score:** 0.00/1.0

**Quality Checks:**
- ❌ has_processed_data
- ❌ has_validation
- ❌ has_insights
- ❌ has_quality_score

**Sample Outputs:**
```json
{
  "discover_sources": "Sorry, I am not able to provide previous results for \"100\" as it is not specific enough. Could you please provide more context or details?",
  "validate_schema": "As an AI model, I'm unable to provide previous results without specific context or information about the topic in question. Please provide more details so I can assist you better.",
  "profile_data": "Yes",
  "...": "(7 more outputs)"
}
```

### multi_agent_collaboration.yaml

**Status:** ✅ Success
**Execution Time:** 25.07s
**Quality Score:** 0.00/1.0

**Quality Checks:**
- ❌ has_solution
- ❌ has_consensus
- ❌ has_contributions
- ❌ consensus_reached

**Sample Outputs:**
```json
{
  "initialize_agents": "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 5...",
  "decompose_problem": "You haven't provided any context for the information you're asking for. Can you please elaborate?",
  "assign_tasks": "Unfortunately, without more specific information or context, I am unable to provide a matching response. Could you please provide more details or clarify your request?",
  "...": "(9 more outputs)"
}
```

## Recommendations

### Low Quality Outputs

- **research_assistant.yaml** (Score: 0.00)
  - Failed checks: has_summary, has_sources, has_key_findings, summary_length, multiple_sources
- **data_processing_workflow.yaml** (Score: 0.00)
  - Failed checks: has_processed_data, has_validation, has_insights, has_quality_score
- **multi_agent_collaboration.yaml** (Score: 0.00)
  - Failed checks: has_solution, has_consensus, has_contributions, consensus_reached

## Conclusion

✅ All examples executed successfully, but some outputs could be improved.
