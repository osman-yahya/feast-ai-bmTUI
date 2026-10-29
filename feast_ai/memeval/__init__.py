"""memeval: a LongMemEval-style Turkish eval set for Mustachio's memory engine.

The set is the acceptance test for every stage of the on-device assistant
(feast-cur ``docs/mobile/mustachio-memory-architecture.md`` §5.1).  It lives
here because this is where a model can be run under a phone-class budget;
the same JSON files are copied into feast-cur as Dart test fixtures, so the
format in :mod:`feast_ai.memeval.schema` is a contract between two languages
and must stay plain JSON.
"""
