"""Building a memeval dataset: a deterministic plan, prose from a large model.

Everything that is graded - which turn carries which fact, the dates, the gold
answers - is decided by code in :mod:`.plan` from a seed.  The large model in
:mod:`.writer` only turns that plan into natural Turkish chat, and must report
which planned beat each turn carries; a missing beat fails the session rather
than silently producing a question with no evidence.  That split is what makes
the labels trustworthy without reading every generated turn.
"""
