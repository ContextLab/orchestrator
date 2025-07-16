"""Configuration file for the Sphinx documentation builder."""

import os
import sys

# Add the source directory to the path
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'Orchestrator'
copyright = '2025, Orchestrator Framework'
author = 'Orchestrator Framework Team'
release = '1.0.0'
version = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx_copybutton',
    'myst_parser',
    'sphinx_tabs.tabs',
    'sphinx_design',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
html_sidebars = {
    '**': [
        'searchbox.html',
        'globaltoc.html',
        'localtoc.html',
        'relations.html',
        'sourcelink.html',
    ]
}

# -- Options for autodoc ----------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# -- Options for autosummary ------------------------------------------------
autosummary_generate = True
autosummary_imported_members = True

# -- Options for napoleon ---------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for intersphinx ------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
}

# -- Options for todo extension ---------------------------------------------
todo_include_todos = True

# -- Options for coverage extension -----------------------------------------
coverage_show_missing_items = True

# -- Options for MyST parser ------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# -- Options for sphinx-copybutton ------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Options for sphinx-tabs ------------------------------------------------
sphinx_tabs_valid_builders = ['html', 'linkcheck']
sphinx_tabs_disable_tab_closing = True

# -- Custom configuration ---------------------------------------------------
def setup(app):
    """Custom setup function."""
    app.add_css_file('custom.css')
    
    # Add custom directives or roles here if needed
    pass

# Suppress warnings for missing references
suppress_warnings = ['ref.citation']

# -- LaTeX output configuration ---------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

latex_documents = [
    (master_doc, 'orchestrator.tex', 'Orchestrator Documentation',
     'Orchestrator Framework Team', 'manual'),
]

# -- Manual page output -----------------------------------------------------
man_pages = [
    (master_doc, 'orchestrator', 'Orchestrator Documentation',
     [author], 1)
]

# -- Texinfo output ---------------------------------------------------------
texinfo_documents = [
    (master_doc, 'orchestrator', 'Orchestrator Documentation',
     author, 'orchestrator', 'AI workflow orchestration framework.',
     'Miscellaneous'),
]

# -- Epub output ------------------------------------------------------------
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright