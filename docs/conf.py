# -*- coding: utf-8 -*-
import inspect
import os
import os.path
import sys

from sphinx.ext import apidoc

from kartothek import __version__ as version

package = "kartothek"

nitpicky = True

__location__ = os.path.join(
    os.getcwd(), os.path.dirname(inspect.getfile(inspect.currentframe()))
)

# Generate module references
output_dir = os.path.abspath(os.path.join(__location__, "../docs/_rst"))
module_dir = os.path.abspath(os.path.join(__location__, "..", package))

apidoc_parameters = ["-f", "-e", "-o", output_dir, module_dir]
apidoc.main(apidoc_parameters)

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "sphinxext")))

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "reference_aliases",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "sphinx_click.ext",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "kartothek"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

release = version

html_static_path = ["_static"]

# Output file base name for HTML help builder.
htmlhelp_basename = "kartothek-doc"


# -- External mapping ------------------------------------------------------------
python_version = ".".join(map(str, sys.version_info[0:2]))

nitpick_ignore = [
    ("py:class", "kartothek.core._mixins.CopyMixin"),
    ("py:class", "dask.delayed.Delayed"),
    # Literals
    ("py:class", "exact"),
    ("py:class", "all"),
    ("py:class", "prefix"),
    # typing.TypeVars not working
    ("py:class", "LiteralValue"),
    ("py:class", "T"),
    ("py:class", ".."),
    ("py:class", "ValueType"),
    # Caused by SchemaWrapper copying/overloading doc string of pyarrow
    ("py:class", "Field"),
    ("py:class", "Schema"),
    ("py:class", "default False"),
    ("py:class", "False"),
    ("py:class", "DataFrame"),
]

intersphinx_mapping = {
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "simplekv": ("https://simplekv.readthedocs.io/en/stable/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
    # Storefact isn't exposing any sphinx refs
    # "storefact": ("https://storefact.readthedocs.io/en/stable", None),
}

# In particular type annotations are rendered as its full path to the class but
# the sphinx docs usually expose the high level API path. This renames the
# sphinx targets appropriately. The replacments are all applied, therefore mind
# overlapping replacements.
reftarget_replace = {
    "pyarrow.lib.": "pyarrow.",
    "pd.": "pandas.",
    "pandas.core.series.Series": "pandas.Series",
    "pandas.core.frame.DataFrame": "pandas.DataFrame",
    "dask.dataframe.core.DataFrame": "dask.dataframe.DataFrame",
    "dask.dataframe.core.Series": "dask.dataframe.Series",
    "dask.bag.core.Bag": "dask.bag.Bag",
    "kartothek.serialization._generic": "kartothek.serialization",
    "kartothek.serialization._parquet": "kartothek.serialization",
}

# In particular the deprecation warning in DatasetMetadata.table_schema is
# raising too many warning to handle sensibly using ipython directive pseudo
# decorators. Remove this with 4.X again
ipython_warning_is_error = False
