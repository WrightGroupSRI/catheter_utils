# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path Setup -----------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'catheter_utils'
copyright = '2025, Sunnybrook Research Institute'
author = 'Sunnybrook Research Institute'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_click',
    'myst_parser'
    ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Options for autodoc
#autodoc_mock_imports = ['numpy', 'numba', 'quaternion', 'scipy','pandas', 'ruamel.yaml', 'schema', 'matplotlib', \
#'click', 'click_log', 'enlighten', 'scipy', 'opencv-python', 'pydicom', 'get_gt']
autodoc_mock_imports = ['get_gt']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
