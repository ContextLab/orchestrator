{% if fact_check_loop.iterations %}
I notice there are some technical issues with the information provided:

1. **No original document**: The "Original document" field shows "None"
2. **Unreadable analysis results**: The claims analysis and reference verification results appear to be Python object references rather than actual content
3. **No specific claims listed**: The new citations section indicates no specific claims were provided to research

**To properly help you update your document, I would need:**

1. **The actual document text** that needs updating
2. **A list of specific claims** that require citations
3. **The current reference list** (if any) that needs fixing
4. **Any broken URLs** that need replacement

**Once you provide this information, I can:**
- Research and add proper citations for unreferenced claims
- Fix any broken reference links
- Format all references consistently as footnotes
- Return a complete, properly cited document

Could you please share:
- The original document text
- The specific claims needing citations
- Any existing references that need fixing

Then I'll be able to provide you with a fully updated and properly referenced document.
{% else %}
{{ load_initial_doc.content }}
{% endif %}
