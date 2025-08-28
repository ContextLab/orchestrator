{% if fact_check_loop.iterations %}
I notice there are some issues with the information provided that prevent me from completing the document update:

1. **No original document**: The "Original document" field shows "None"
2. **Inaccessible analysis results**: The claims analysis and reference verification results appear to be Python object references that I cannot read
3. **No specific claims listed**: The new citations section indicates no specific claims were provided

To properly update your document with citations and fix broken references, I would need:

## Required Information:

1. **The actual document text** you want me to update
2. **Specific claims** that need citations
3. **Current reference list** so I can identify broken URLs
4. **Any existing sources** you'd like me to verify

## What I Can Do Once You Provide This:

- Research and add credible citations for unreferenced claims
- Test and replace any broken reference URLs
- Format all references consistently as footnotes
- Ensure proper academic citation style

**Next Steps**: Please share the actual document content and specific claims that need citations, and I'll provide a fully updated version with proper references and footnotes.

Would you like to paste the document content so I can help you improve it?
{% else %}
{{ load_initial_doc.content }}
{% endif %}
