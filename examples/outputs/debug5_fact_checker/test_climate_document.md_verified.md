{% if fact_check_loop.iterations %}
I notice that the information you've provided appears to contain placeholder objects rather than the actual document content and analysis results. The "Original document" shows "None" and the other sections show Python object references rather than readable content.

To properly help you update a document with citations and references, I would need:

1. **The actual document text** that needs to be updated
2. **The specific claims** that need citations
3. **The reference verification results** showing which URLs are broken
4. **The new citations** that should be added

Could you please provide:
- The original document text
- A list of the unreferenced claims that need citations
- The results of the reference verification (which URLs are broken)
- Any new sources or citations you'd like me to incorporate

Once you share this information, I'll be able to:
1. Add proper citations for all unreferenced claims
2. Fix any broken reference URLs
3. Format all references consistently as footnotes
4. Return the complete updated document

Please paste the actual content, and I'll help you create a properly cited and formatted document.
{% else %}
{{ load_initial_doc.content }}
{% endif %}
