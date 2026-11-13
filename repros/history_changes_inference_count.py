"""Purpose: reproduce pymetta 0.9.4 counting upkeep that depends on what the process ran before inside an
evaluation's inferences, so the same evaluation in a fresh context of the same kind counts differently.

Run: .venv/bin/python repros/history_changes_inference_count.py
Each measurement opens a context, imports lib_tabling first in a tabling context, defines a ten-step countdown,
parses the call before counting and counts only m.eval of it, with Python's collector off. Expected: the same
count for every context of a kind, since each context is closed before the next opens and MeTTa.stats() calls its
count the block's own work. Observed: plain contexts count 5045, then 4922, then 4921 each; the first tabling
context counts 4973 and later ones 4969; after that a plain context counts 4907 right after a tabling one and 4911
after another plain one. The first evaluations pay for resolving the builtins they call and filling pymetta's
dispatch cache, inside their count. The 4907 and 4911 differ because pymetta keeps its translation cache's
invalidation hooks only while some translation is cached: a tabling context's import leaves them installed, a
plain context's close removes them, and the next plain context installs them inside its count.
"""
import gc

from metta import MeTTa, parse

COUNTDOWN = "(= (countdown $n) (if (> $n 0) (countdown (- $n 1)) done))"


def measured(*, tabling: bool) -> int:
    """Inferences one countdown from 10 counts in a fresh context, which imports lib_tabling first when asked."""
    with MeTTa() as m:
        if tabling:
            m.self.run("!(import! (context-space) (library lib_tabling))")
        m.self.run(COUNTDOWN)
        goal = parse("(countdown 10)")
        gc.disable()
        try:
            with m.stats() as stats:
                m.eval(goal)
        finally:
            gc.enable()
        return stats.inferences


if __name__ == "__main__":
    for kind in ("plain", "plain", "plain", "tabling", "tabling", "plain", "plain", "tabling", "plain"):
        print(f"{kind:8} context: {measured(tabling=kind == 'tabling')} inferences")
