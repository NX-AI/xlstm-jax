# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "xlstm-jax"
copyright = "2024, NXAI"
author = "NXAI"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.githubpages",  # create a .nojekyll file in the HTML docs to publish the document on GitHub Pages
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    # "sphinx.ext.viewcode",  # throws an error
    "sphinx_rtd_theme",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# AutoAPI extension settings
# https://github.com/readthedocs/sphinx-autoapi
extensions.append("autoapi.extension")
autoapi_dirs = ["../xlstm_jax"]
autoapi_file_patterns = ["*.py"]
autoapi_keep_files = True
autoapi_options = [
    "members",
    "undoc-members",
    # "imported-members",  # omit imported members (e.g. imported in __init__.py)
    "inherited-members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
    # "special-members",  # omit special members (e.g. __len__)
]
autoapi_type = "python"
autodoc_typehints = "description"

# Napoleon settings
extensions.append("sphinx.ext.napoleon")
napoleon_google_docstring = True
napoleon_include_init_with_doc = True

# intersphinx extension settings
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
intersphinx_mapping = {
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
    "grain": ("https://grain.readthedocs.io/en/latest/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "optax": ("https://optax.readthedocs.io/en/latest/", None),
    "optree": ("https://optree.readthedocs.io/en/latest/", None),
    "orbax": ("https://orbax.readthedocs.io/en/latest/", None),
    "python": ("https://docs.python.org/3/", None),
    "tabulate": ("https://tabulate.readthedocs.io/en/latest/", None),
    "tensorflow": (
        "https://www.tensorflow.org/api_docs/python",
        "https://github.com/GPflow/tensorflow-intersphinx/raw/master/tf2_py_objects.inv",
    ),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "logo": {
        "image_light": "_static/nxai_logo_light.svg",
        "image_dark": "_static/nxai_logo_dark.svg",
    }
}