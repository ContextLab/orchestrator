{% if fact_check_loop.iterations %}
I notice there are some technical issues with the information provided. The original document shows "None", and the claims analysis and reference verification results appear to be Python object references rather than actual content.

To properly help you update a document with citations and references, I would need:

1. **The actual document text** that needs to be updated
2. **The specific claims** that require citations
3. **The reference verification results** in readable format (not Python object references)
4. **The list of sources** that were found for citation

Could you please provide:
- The original document content
- A list of the unreferenced claims that need citations
- Any broken reference URLs that need fixing
- The sources/citations that should be added

Once I have this information, I'll be able to:
- Add proper citations for all unreferenced claims
- Fix any broken reference URLs
- Format all references consistently as footnotes
- Return the complete updated document

Please share the actual document content and claims, and I'll provide a fully updated version with proper citations and formatting.
{% else %}
{{ load_initial_doc.content }}
{% endif %}
