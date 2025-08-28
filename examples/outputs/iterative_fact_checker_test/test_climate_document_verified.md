{% if fact_check_loop.iterations %}
{{ fact_check_loop.iterations[-1].update_document.result }}
{% else %}
{{ load_initial_doc.content }}
{% endif %}
