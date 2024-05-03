(intro)=

# Jupyter Book on Read the Docs

This example shows a Jupyter Book project built and published on Read the Docs.
You're encouraged to use it to get inspiration and copy & paste from the files in [the source code repository][github]. In the source repository, you will also find the relevant configuration and instructions for building Jupyter Book projects on Read the Docs.

If you are using Read the Docs for the first time, have a look at the official [Read the Docs Tutorial][tutorial].
If you are using Jupyter Book for the first time, have a look at the [official Jupyter Book documentation][jb-docs].

## Why run Jupyter Book with Read the Docs?

[Read the Docs](https://readthedocs.org/) simplifies developing Jupyter Book projects by automating building, versioning, and hosting of your project for you.
You might be familiar with Read the Docs for software documentation projects, but these features are just as relevant for science.

With Read the Docs, you can improve collaboration on your Jupyter Book project with Git (GitHub, GitLab, BitBucket etc.) and then connect the Git repository to Read the Docs.
Once Read the Docs and the git repository are connected, your project will be built and published automatically every time you commit and push changes with git.
Furthermore, if you open Pull Requests, you can preview the result as rendered by Jupyter Book.

## What is in this example?

Jupyter Book has a number of built-in features.
This is a small example book to give you a feel for how book content is structured.
It shows off a few of the major file types, as well as some sample content.
It does not go in-depth into any particular topic - check out [the Jupyter Book documentation][jb-docs] for more information.

* [Examples of Markdown](/markdown)
* [Rendering a notebook Jupyter Notebook](/notebooks)
* [A notebook written in MyST Markdown](/markdown-notebooks)

We have also added some popular features for Jupyter Book that really you shouldn't miss when building your own project with Jupyter Book and Read the Docs:

* [intersphinx to link to other documentation and Jupyter Book projects](/intersphinx)
* [sphinx-examples to show examples and results side-by-side](/sphinx-examples)
* [sphinx-hoverxref to preview cross-references](/sphinx-hoverxref)
* [sphinx-proof for logic and math, to write proofs, theorems, lemmas etc.](/sphinx-proof)


## Table of Contents

Here is an automatically generated Tabel of Contents:

```{tableofcontents}
```

[github]: https://github.com/readthedocs-examples/example-jupyter-book/ "GitHub source code repository for the example project"
[tutorial]: https://docs.readthedocs.io/en/stable/tutorial/index.html "Official Read the Docs Tutorial"
[jb-docs]: https://jupyterbook.org/en/stable/ "Official Jupyter Book documentation"
