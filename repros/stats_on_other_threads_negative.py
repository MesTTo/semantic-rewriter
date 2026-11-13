"""Purpose: reproduce pymetta 0.9.4's stats() reporting a meaningless inference count on every thread but the one that
started its runtime.

Run: .venv/bin/python repros/stats_on_other_threads_negative.py [main|thread ...]
Each step counts one ten-step countdown in a fresh context, on the main thread or on a new thread, in the order given
(by default main, main, thread, thread, main). Expected: about 4,900 inferences on every thread, since stats() reads
the calling thread's counters and the countdown runs on the thread that counts it. Observed: the thread that first
used pymetta counts 5,045 and then 4,922; every other thread, the main thread included when a new thread went first,
reports -40 the first time it counts and -5 after that, with the right answer each time.
"""
import sys
import threading

from metta import MeTTa, parse

COUNTDOWN = "(= (countdown $n) (if (> $n 0) (countdown (- $n 1)) done))"


def counted() -> tuple[int, str]:
    """Inferences one countdown from 10 counts in a fresh context on the calling thread, and its answer."""
    with MeTTa() as m:
        m.self.run(COUNTDOWN)
        goal = parse("(countdown 10)")
        with m.stats() as stats:
            answers = m.eval(goal)
        return stats.inferences, " ".join(map(str, answers))


def on_new_thread() -> tuple[int, str]:
    """counted() run on a thread started for it."""
    box = []
    thread = threading.Thread(target=lambda: box.append(counted()))
    thread.start()
    thread.join()
    return box[0]


if __name__ == "__main__":
    for where in sys.argv[1:] or ["main", "main", "thread", "thread", "main"]:
        count, answer = counted() if where == "main" else on_new_thread()
        print(f"{where:6} {count} inferences, answer {answer}")
