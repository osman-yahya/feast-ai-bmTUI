"""Python reference implementation of the Mustachio memory engine (step 4).

One module per unit the Dart port will have, same names:

``timeparse``    Turkish relative time -> absolute dates, by rule (§3.3)
``store``        facts with history + raw episodes, per persona
``consolidate``  session-end write path: SLM extracts, code canonicalizes and
                 applies ADD / UPDATE / DELETE / NOOP; KVKK erasure
``retrieve``     Turkish BM25 + time / recency boosts
``gate``         when memory goes into the prompt at all
``assemble``     the MEMORY block with f/t handles, within a token budget
"""
