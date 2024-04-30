# Intersphinx: Cross-reference other projects

Behind the built-in Sphinx extension `intersphinx` is a powerful tool to reference sections in other Sphinx and Jupyter Book documentation projects.

You can configure mappings to external Sphinx projects in your Jupyter Book configuration, the `_config.yml` file. In this example project, we have configured `ebp` to reference `https://executablebooks.org/en/latest/`. In the following code examples, we refer to the configured `ebp` mapping and link directly to a section called `tools`.

```{tab} MyST (Markdown)

```{example}
We can link to pages in other documentation projects.
This is a link to the
[Executable Book project's list of tools they build](ebp:tools)
```


```{tab} reStructuredText

```{example}

```{eval-rst}
We can link to pages in other documentation projects.
This is a link to the
:doc:`Executable Book project's list of tools they build <ebp:tools>`
```

```{note}
In the above `reStructuredText` example, we use `{eval-rst}` to write reST inside a `.md` file (i.e. the one you are reading now). You only need to use this directive if you are writing reST code in a `.md` file.
```
