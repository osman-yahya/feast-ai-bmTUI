"""mustachio: the model workspace behind feast mobile's on-device assistant.

feast AI (the sibling package) measures *which pipeline* works on a phone.
This package changes *the model itself*: a text-only Gemma 4 E2B, fine-tuned
on the 15 feast tools in the model's native call format, robust to how
people actually type Turkish - no diacritics, slang, and English words mixed
into a Turkish sentence.

It reuses feast AI's definitions instead of copying them: the tools
(:mod:`feast_ai.agent.tools`), the held-out suites and their scoring
(:mod:`feast_ai.agent.suite`).  Weights, datasets and runs live under
``~/.feast-ai/mustachio/`` - never in the repository.
"""

__version__ = "0.1.0"
