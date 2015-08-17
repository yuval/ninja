The directory holds documentation for Ninja.

We are using Sphinx, http://sphinx-doc.org/, which is used to generate
documentation for the Python project and many other projects, e.g. see
https://readthedocs.org/.  You write docs in rst (reStructuredText).
Github supports rst, so files can be viewed easily.  rst can be
transformed to html, pdf, and other formats.

rst files can be viewed directly on github, with decent rendering.
But building and viewing locally is an easy way to spot
syntax/rendering issues *before* pushing to github.  The rendered doc
is also a bit nicer here.  pdf requires either a latex install.

To install sphinx:

  $ virtualenv ~/venv
  $ . ~/venv/bin/activate
  # Do NOT put your venv in this dir (doc) because you'll end up
  # producing doc for from python's own rst files under venv!
  (venv)
  $ pip install sphinx     # http://sphinx-doc.org/

To generate the Makefile, we ran:

  $ sphinx-quickstart

To generate and view simple singlehtml format:

  $ make singlehtml
  $ open _build/singlehtml/index.html

Other targets are available:

  $ make help
  Please use `make <target>' where <target> is one of
    html       to make standalone HTML files
    dirhtml    to make HTML files named index.html in directories
    singlehtml to make a single large HTML file
    pickle     to make pickle files
    json       to make JSON files
    htmlhelp   to make HTML files and a HTML help project
    qthelp     to make HTML files and a qthelp project
    devhelp    to make HTML files and a Devhelp project
    epub       to make an epub
    latex      to make LaTeX files, you can set PAPER=a4 or PAPER=letter
    latexpdf   to make LaTeX files and run them through pdflatex
    latexpdfja to make LaTeX files and run them through platex/dvipdfmx
    text       to make text files
    man        to make manual pages
    texinfo    to make Texinfo files
    info       to make Texinfo files and run them through makeinfo
    gettext    to make PO message catalogs
    changes    to make an overview of all changed/added/deprecated items
    xml        to make Docutils-native XML files
    pseudoxml  to make pseudoxml-XML files for display purposes
    linkcheck  to check all external links for integrity
    doctest    to run all doctests embedded in the documentation (if enabled)
