"""Purpose: register the Hypothesis profiles the tests run under. `thorough`, the
default, gives each property test 400 cases unless the test fixes its own count;
`ci` gives 100, which CI's PyMeTTa job selects with `--hypothesis-profile=ci` so
that the engine properties on PyMeTTa finish in reasonable time. Neither puts a
deadline on an example.
"""

from hypothesis import settings

# Every profile turns Hypothesis's 200 ms per-example deadline off. On a shared, loaded machine an example that takes
# 20 ms stalls past it (the fine-grid trust property took 419 ms and 301 ms on two machines, then 22 and 80 ms on
# replay), and a timing failure says nothing about a property; the CI job's own time bounds the suite's cost.
settings.register_profile("thorough", max_examples=400, deadline=None)
settings.register_profile("ci", max_examples=100, deadline=None)
settings.load_profile("thorough")
