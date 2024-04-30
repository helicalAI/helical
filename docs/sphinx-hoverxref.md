# sphinx-hoverxref: Preview tooltips for cross-references

Using `sphinx-hoverxref`, we can preview cross-references in the documentation.

For instance, try placing your mouse over {hoverxref}`this reference to the front page <intro>`.
We have use a named "target" for the reference. Here are some examples of how to name your targets:

```{tab} MyST (Markdown)

```markdown

 (foobar)=
 
 # My Headline
 
 Here is a paragraph, we link to this headline {hoverxref}`like this <foobar>`.
```

```{tab} reStructuredText

```rest
.. _foobar:

My Headline
===========

Here is a paragraph, we link to this headline :hoverxref:`like this <foobar>`
```

`sphinx-hoverxref` works especially well on Read the Docs, since the platform runs the necessary API backend for generating preview data.
Aside from enabling the extention in your project's `_config.yml`, you don't have to do anyhing, it Just Works™️. 
