{% if fact_check_loop.iterations %}
I notice there are some issues with the information provided:

1. **Original document**: Listed as "None" - I need the actual document text to update
2. **Claims analysis**: Shows as an object reference rather than readable content
3. **Reference verification results**: Also shows as an object reference rather than readable content
4. **New citations to add**: Contains a response asking for more information rather than specific citations

To properly help you update the document, I would need:

1. **The actual document text** that needs updating
2. **A list of specific claims** that need citations
3. **The reference verification results** in readable format showing which URLs are broken
4. **The sources found** for adding new citations

Could you please provide:
- The complete original document text
- The specific unreferenced claims that need citations
- Any broken reference URLs that need fixing
- The new sources/citations to be added

Once I have this information, I'll be able to:
- Add proper citations for all unreferenced claims
- Fix any broken references
- Format all references consistently as footnotes
- Return the complete updated document

Please share the actual document content and citation details, and I'll get started on the updates right away.
{% else %}
{{ load_initial_doc.content }}
{% endif %}
