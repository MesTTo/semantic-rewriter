"""Purpose: register the Hypothesis profiles the tests run under. `thorough`, the
default, gives each property test 400 cases unless the test fixes its own count;
`ci` gives 100, which CI's PyMeTTa job selects with `--hypothesis-profile=ci` so
that the engine properties on PyMeTTa finish in reasonable time.
"""

from hypothesis import settings

settings.register_profile("thorough", max_examples=400)
settings.register_profile("ci", max_examples=100)
settings.load_profile("thorough")
