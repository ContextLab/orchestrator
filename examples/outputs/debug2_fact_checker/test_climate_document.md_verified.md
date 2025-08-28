{% if fact_check_loop.iterations %}
I notice there are some issues with the information provided:

1. **No original document**: The "Original document" field shows "None"
2. **Unreadable analysis results**: The claims analysis and reference verification results appear to be Python object references rather than actual content
3. **No specific claims**: The "New citations to add" section indicates no specific claims were provided

To properly help you update a document with citations and references, I would need:

## Required Information:
1. **The actual document text** that needs updating
2. **Specific claims** that require citations
3. **Source materials** or topics to research
4. **Any existing references** that need verification or formatting

## What I Can Do Once You Provide This:
- Add properly formatted footnote citations
- Verify and fix broken reference URLs
- Ensure consistent citation formatting
- Research and add credible sources for unreferenced claims

## Next Steps:
Please provide:
```
1. The complete document text
2. List of specific claims needing citations
3. Any existing references to check
4. Your preferred citation style (if any)
```

Once you share this information, I'll be able to create a fully updated document with proper citations and references formatted as footnotes.
{% else %}
{{ load_initial_doc.content }}
{% endif %}
