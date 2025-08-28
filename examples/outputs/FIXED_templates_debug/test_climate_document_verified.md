{% if fact_check_loop.iterations %}
I notice there are some issues with the information provided:

1. The "Original document" field shows "None" - there's no actual document content to update
2. The "Claims analysis" and "Reference verification results" appear to be Python object references rather than readable content
3. The "New citations to add" section indicates that no specific claims were provided for citation

To properly help you update a document with citations and references, I would need:

**Required Information:**
- The actual document text that needs updating
- A list of specific claims that need citations
- Any existing references that need to be checked or reformatted
- The preferred citation style (APA, MLA, Chicago, etc.)

**What I can help with once you provide the content:**
- Adding properly formatted citations for unreferenced claims
- Verifying and fixing broken reference URLs
- Standardizing reference formatting
- Creating consistent footnotes
- Ensuring all sources are credible and relevant

Could you please provide:
1. The actual document text
2. The specific claims that need citations
3. Any existing references that need verification
4. Your preferred citation format

Once I have this information, I'll be able to provide you with a complete, properly cited document with consistent formatting and working references.
{% else %}
{{ load_initial_doc.content }}
{% endif %}
