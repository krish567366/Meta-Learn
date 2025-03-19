import os
import sys
sys.path.insert(0, os.path.abspath('../../metalearn'))

project = 'Krishna Bajpai Meta-Learn'
copyright = '2023, Krishna Bajpai'
author = 'QAI Team'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinxcontrib.bibtex'
]

autodoc_mock_imports = [
    'torch',
    'gymnasium',
    'quantumlib',
    'spikingjelly'
]

bibtex_bibfiles = ['refs.bib']
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['quantum.js']