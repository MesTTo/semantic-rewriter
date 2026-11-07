"""Purpose: the semantic rewriter's public API: answer a query over a knowledge base whose
predicates use other words than the query, by discovering and admitting
implications between predicates and evaluating on any engine that implements
semrewrite.protocols.Engine.

    from semrewrite import Rewriter, formats
    from semrewrite.engines import make as engine
    from semrewrite.judges import make as judge

    kb = formats.load("family.kb")
    rw = Rewriter(kb, judge("nli-deberta-large"), engine("python"))
    print(rw.ask("parent(X, tom)").answers)

Importing this module needs neither PyMeTTa nor any model; engines and judges
load their backends when made.
"""

from semrewrite.admission import Directional, Threshold
from semrewrite.discovery import Discoverer, ScoreCache
from semrewrite.kb import KB, atom
from semrewrite.model import Atom, Predicate, Proposal, Rewrite, Rule, parse_atom
from semrewrite.protocols import Admission, Budget, Capabilities, Engine, Judge, Program, Proof, Retriever, Solution
from semrewrite.rewriter import STRATEGIES, CapabilityError, Outcome, Rewriter, Strategy

__version__ = "0.1.0"

__all__ = [
    "KB", "STRATEGIES", "Admission", "Atom", "Budget", "Capabilities", "CapabilityError", "Directional", "Discoverer",
    "Engine", "Judge", "Outcome", "Predicate", "Program", "Proof", "Proposal", "Retriever", "Rewrite", "Rewriter",
    "Rule", "ScoreCache", "Solution", "Strategy", "Threshold", "atom", "parse_atom",
]
