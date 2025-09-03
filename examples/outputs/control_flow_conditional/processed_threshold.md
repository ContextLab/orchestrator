# Processed File

Original size: {{ read_file.size }} bytes
Processing type: {% if read_file.size == 0 %}Empty file{% elif read_file.size > size_threshold %}Compressed{% else %}Expanded{% endif %}


## Result

{% if handle_empty.status is not defined or handle_empty.status != 'skipped' %}{{ handle_empty }}{% elif compress_large.status is not defined or compress_large.status != 'skipped' %}{{ compress_large }}{% elif expand_small.status is not defined or expand_small.status != 'skipped' %}{{ expand_small }}{% else %}No content processed.{% endif %}
