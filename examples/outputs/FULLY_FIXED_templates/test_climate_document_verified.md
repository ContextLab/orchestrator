{% if fact_check_loop.iterations %}
I notice there are some issues with the information provided:

1. **Original document**: Listed as "None" - I need the actual document text to work with
2. **Claims analysis**: Shows a Python object reference instead of the actual claims that need citations
3. **Reference verification results**: Also shows a Python object reference instead of verification results
4. **New citations**: Contains a generic response rather than specific citations

To properly help you update the document, I would need:

1. **The actual document text** that needs to be updated
2. **A list of specific claims** that require citations
3. **The reference verification results** showing which URLs are broken
4. **The new citations** with proper source information

Could you please provide:
- The original document text
- The specific unreferenced claims that need citations
- Any broken reference URLs that need fixing
- The new citation sources (URLs, titles, publication details)

Once I have this information, I'll be able to:
- Add proper footnote citations for all unreferenced claims
- Fix any broken reference URLs
- Format all references consistently as footnotes
- Return the complete updated document

Please share the actual content, and I'll get started on the improvements right away.
{% else %}
{{ load_initial_doc.content }}
{% endif %}
