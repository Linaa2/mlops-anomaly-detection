import os
import sys

PROJECT_ROOT = os.path.abspath('../..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

project = 'MLOps Anomaly Detection'
copyright = '2025, Groupe MLOps'
author = 'Groupe MLOps'
release = '1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

templates_path = ['_templates']
exclude_patterns = ['_build']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}
