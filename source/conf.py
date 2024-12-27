# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Football_injury_prediction'
copyright = '2024, Baji Fatima Ezahra'
author = 'Baji Fatima Ezahra'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
      'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
]

html_theme = 'sphinx_rtd_theme'
templates_path = ['_templates']


exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Chemin vers les fichiers statiques
html_static_path = ['_static']

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

# Assurez-vous que votre dossier `images` est correctement référencé
html_extra_path = ['images']
