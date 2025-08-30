{% if fact_check_loop.iterations %}
I notice there's an issue with the information provided. The document content shows as "None" and the claims analysis and reference verification results appear to be object references rather than actual content. Without the actual document text and specific claims to verify, I cannot provide the requested updates.

To properly help you update your document with citations and fix references, I would need:

1. **The actual document text** (not "None")
2. **The specific claims** that need citations (not an object reference)
3. **The reference verification results** showing which URLs are broken (not an object reference)

Could you please provide:
- The full text of the document you want updated
- A list of the specific claims that need citations
- Details about which references are broken or need fixing

Once I have this information, I'll be able to:
- Add proper citations for unreferenced claims
- Fix any broken reference URLs
- Format all references consistently as footnotes
- Return the complete updated document

Please share the actual document content and claims, and I'll provide a comprehensive update with proper citations and formatting.
{% else %}
{{ load_initial_doc.content }}
{% endif %}
