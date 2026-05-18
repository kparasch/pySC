from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS_BUILD = ROOT / "docs" / "build"
DOCS_CACHE = DOCS_BUILD / ".cache"
DOCS_CACHE.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(DOCS_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(DOCS_CACHE / "xdg"))

sys.path.insert(0, str(ROOT))

import pySC

project = "pySC"
author = "Konstantinos Paraschou"
copyright = f"{datetime.now().year}, {author}"
release = pySC.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = False
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["generated/*"]

html_theme = "sphinx_rtd_theme"
html_title = "pySC"
html_baseurl = os.environ.get(
    "READTHEDOCS_CANONICAL_URL",
    "https://accelerator-commissioning.readthedocs.io/",
)
html_context = {
    "display_github": True,
    "github_user": "kparasch",
    "github_repo": "pySC",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}

intersphinx_mapping = {}
