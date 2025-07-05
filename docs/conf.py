# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Add the src directory to sys.path
sys.path.insert(0, os.path.abspath("../src"))

project = "TAMLEP"
copyright = "2025, Rakesh Sukumar"
author = "Rakesh Sukumar"
release = "0.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # for parsing MyST style markdown files
    "myst_parser",
    # prints time to build docs
    "sphinx.ext.duration",
    # generate link for all section headers
    "sphinx.ext.autosectionlabel",
    # generate docymentation from python docstrings
    "sphinx.ext.autodoc",
    # create a link that point to the real function/class
    "sphinx.ext.viewcode",
    # enhance the visual and usability of our documentation
    "sphinx_copybutton",
    "sphinx_design",
    # generate auto summary
    "sphinx.ext.autosummary",
    #
    # "sphinx_autodoc_typehints",
    # generate automatic summary
    # "autoapi.extension",
    #  build doc from Jupyter notebooks
    # "nbsphinx",
    # for supporting numpy style python docstrings
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# # default theme
# html_theme = "alabaster"
# # a good choice
# html_theme = "furo"
# theme used by numpy, pandas etc.
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# sphinx-autoapi configuration
# autoapi_dirs = ["../src"]
