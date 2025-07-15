"""
Configuration file for the Sphinx documentation builder.
"""

import os
import sys
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'py-orc'
copyright = f'{datetime.now().year}, Contextual Dynamics Lab'
author = 'Contextual Dynamics Lab'  
release = '0.1.0'
version = '0.1.0'

# HTML context variables to customize theme
html_context = {
    'display_github': True,
    'github_user': 'ContextLab',
    'github_repo': 'orchestrator',
    'github_version': 'main',
    'conf_py_path': '/docs_sphinx/',
}

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'sphinx_tabs.tabs',
    'myst_parser',
    'nbsphinx',
]

# MyST parser configuration
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    "linkify",
    "strikethrough",
    "substitution",
    "tasklist",
    "attrs_inline",
    "attrs_block",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'monokai'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'xanadu'

# Theme options are theme-specific and customize the look and feel of a theme
# further.
html_theme_options = {
    "navbar_name": "Orchestrator",
    "navbar_logo_colour": "#FFF",
    "navbar_home_link": "https://github.com/contextualdynamics/orchestrator",
    "github_repo": "https://github.com/contextualdynamics/orchestrator",
    "navbar_left_links": [
        {
            "name": "Getting Started",
            "href": "getting_started",
            "active": True,
        },
        {
            "name": "Tutorials",
            "href": "tutorials/index",
            "active": True,
        },
        {
            "name": "API",
            "href": "api/index",
            "active": True,
        },
        {
            "name": "Examples",
            "href": "examples/index",
            "active": True,
        },
    ],
    "navbar_right_links": [
        {
            "name": "GitHub",
            "href": "https://github.com/contextualdynamics/orchestrator",
            "icon": "fab fa-github",
        },
    ],
    "extra_copyrights": [],
    "google_analytics_tracking_id": "",  # Add if needed
    "prev_next_button": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Include custom CSS
html_css_files = [
    'custom.css',
]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
html_sidebars = {
    "**": [
        "searchbox.html",
        "globaltoc.html",
        "relations.html",
        "sourcelink.html",
    ]
}

# -- Extension configuration -------------------------------------------------

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jinja2': ('https://jinja.palletsprojects.com/en/3.0.x/', None),
}

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autosummary settings
autosummary_generate = True

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# nbsphinx settings
nbsphinx_execute = 'never'  # Don't execute notebooks during build