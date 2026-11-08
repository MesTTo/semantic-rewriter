"""Purpose: the benchmark knowledge bases: four generated worlds (family, work,
geography, medicine) whose facts are spread across many surface predicates, with
trap predicates a similarity matcher would admit, author-labelled gold
implications, user rules written over vocabulary no fact uses, and queries.

Each world is generated from a seed, so a knowledge base is a pure function of
its seed. A true relation is written as a fact under one of its surface forms,
chosen at random: a parent link may be stored as (mother alice tom),
(mum alice tom), (has_mother tom alice) or (son_of tom alice), so a query needs
the rewrites that join those forms. Gold implications are the author's labels
of what the glosses entail in every situation by the meaning of their words,
audited once against every rewrite the seven judges admitted: the audit added
entailments the first labels missed (mother(y, x) entails has_mother(x, y),
grandparent entails ancestor, near is symmetric) and made geographic
containment transitive, as "located in" is in English. A rewrite counts as
true when the gold program entails it (KB.entails), user rules included.
Traps are predicates that look related to a query predicate but do not entail
it (guardian_of is not parent).
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Sequence

from semrewrite.kb import KB, atom
from semrewrite.model import Rule, keys_of

T, F = True, False

NAMES = ["ada", "abe", "ann", "ben", "bea", "cal", "cora", "dan", "dee", "eli", "eva", "finn", "fay", "gus", "gia", "hal", "hope", "ian", "ivy", "jon", "joy", "kai", "kim", "leo", "lia", "max", "mia", "ned", "nia", "otto", "ora", "pat", "pia", "quin", "rae", "ray", "rose", "sam", "sue", "ted", "tia", "uma", "val", "vic", "wes", "wren", "xan", "yara", "yves", "zed", "zoe", "ari", "bo", "cy", "di", "ed", "flo", "gil", "hux", "iris", "jude", "kit", "lou", "mo", "nell", "oz", "pru", "rex"]


def _choose(rng: random.Random, options: Sequence[tuple[float, tuple[str, str, str]]]) -> tuple[str, str, str]:
    """One of the facts, drawn with its weight."""
    weights = [w for w, _ in options]
    return rng.choices([fact for _, fact in options], weights=weights)[0]


def _rule(head: tuple[str, str, str], *body: tuple[str, str, str]) -> Rule:
    return Rule(atom(*head), tuple(atom(*b) for b in body))


def _transitive(name: str, step: str) -> tuple[Rule, Rule]:
    """name(X, Y) :- step(X, Y).  name(X, Z) :- step(X, Y), name(Y, Z)."""
    return (_rule((name, "X", "Y"), (step, "X", "Y")), _rule((name, "X", "Z"), (step, "X", "Y"), (name, "Y", "Z")))


def family(seed: int = 7) -> KB:
    """Three generations from three founding couples, parent links under seven surface forms."""
    rng = random.Random(seed)
    pool = iter(NAMES)
    gender: dict[str, str] = {}
    parents: dict[str, tuple[str, str]] = {}
    couples: list[tuple[str, str]] = []

    def person(g: str | None = None) -> str:
        name = next(pool)
        gender[name] = g or rng.choice("fm")
        return name

    def marry(p: str) -> tuple[str, str]:
        other = person("f" if gender[p] == "m" else "m")
        pair = (p, other) if gender[p] == "m" else (other, p)
        couples.append(pair)
        return pair

    def children(pair: tuple[str, str], lo: int, hi: int) -> list[str]:
        kids = [person() for _ in range(rng.randint(lo, hi))]
        for kid in kids:
            parents[kid] = pair
        return kids

    level1 = [k for _ in range(3) for k in children(marry(person("m")), 2, 3)]
    level2 = [k for p in level1[: 2 * len(level1) // 3] for k in children(marry(p), 1, 3)]
    level3 = [k for p in level2[: len(level2) // 2] for k in children(marry(p), 1, 2)]

    facts: set[tuple[str, str, str]] = set()
    for kid, (dad, mum) in parents.items():
        child_of = "son_of" if gender[kid] == "m" else "daughter_of"
        facts.add(_choose(rng, [
            (3, ("mother", mum, kid)), (1, ("mum", mum, kid)), (1, ("mom", mum, kid)),
            (1, ("biological_mother", mum, kid)), (1, ("has_mother", kid, mum)),
            (1, (child_of, kid, mum)),
        ]))
        facts.add(_choose(rng, [
            (3, ("father", dad, kid)), (1, ("dad", dad, kid)), (1, (child_of, kid, dad)),
        ]))
    for man, woman in couples:
        facts.add(_choose(rng, [
            (1, ("married_to", man, woman)), (1, ("married_to", woman, man)),
            (1, ("husband_of", man, woman)), (1, ("wife_of", woman, man)),
        ]))
    by_parents: dict[tuple[str, str], list[str]] = defaultdict(list)
    for kid, pair in parents.items():
        by_parents[pair].append(kid)
    for kids in by_parents.values():
        for i, a in enumerate(kids):
            for b in kids[i + 1:]:
                if rng.random() < 0.6:
                    x, y = (a, b) if rng.random() < 0.5 else (b, a)
                    facts.add(("brother_of" if gender[x] == "m" else "sister_of", x, y))
    grandkids = [k for k in level2 + level3 if parents[k][1] in parents]
    for kid in grandkids[:4]:
        facts.add(("grandmother_of", parents[parents[kid][1]][1], kid))
    adults = [p for p in gender if p not in level3]
    for kid in rng.sample(level2 + level3, 4):
        facts.add(("guardian_of", rng.choice([a for a in adults if a not in parents[kid]]), kid))
    for kid in rng.sample(level2, 2):
        facts.add(("stepmother_of", rng.choice([a for a in adults if gender[a] == "f" and a != parents[kid][1]]), kid))
    for kid in rng.sample(level2 + level3, 3):
        facts.add(("godfather_of", rng.choice([a for a in adults if gender[a] == "m" and a != parents[kid][0]]), kid))
    for kid in level2 + level3:
        dad, mum = parents[kid]
        for side in (dad, mum):
            if side in parents:
                for sib in by_parents[parents[side]]:
                    if sib != side and rng.random() < 0.3:
                        facts.add(("aunt_of" if gender[sib] == "f" else "uncle_of", sib, kid))

    glosses = {
        "mother": "{x} is the mother of {y}", "mum": "{x} is the mum of {y}", "mom": "{x} is the mom of {y}",
        "biological_mother": "{x} is the biological mother of {y}", "has_mother": "{x} has {y} as their mother",
        "father": "{x} is the father of {y}", "dad": "{x} is the dad of {y}",
        "son_of": "{x} is a son of {y}", "daughter_of": "{x} is a daughter of {y}",
        "parent": "{x} is a parent of {y}", "child": "{x} is a child of {y}",
        "married_to": "{x} is married to {y}", "spouse": "{x} is the spouse of {y}",
        "husband_of": "{x} is the husband of {y}", "wife_of": "{x} is the wife of {y}",
        "brother_of": "{x} is a brother of {y}", "sister_of": "{x} is a sister of {y}", "sibling": "{x} is a sibling of {y}",
        "grandparent": "{x} is a grandparent of {y}", "grandmother_of": "{x} is the grandmother of {y}",
        "ancestor": "{x} is an ancestor of {y}",
        "guardian_of": "{x} is the legal guardian of {y}", "stepmother_of": "{x} is the stepmother of {y}",
        "godfather_of": "{x} is the godfather of {y}", "aunt_of": "{x} is an aunt of {y}", "uncle_of": "{x} is an uncle of {y}",
    }
    gold = {
        ("mother", "parent", F), ("father", "parent", F),
        ("mum", "mother", F), ("mother", "mum", F), ("mom", "mother", F), ("mother", "mom", F),
        ("biological_mother", "mother", F), ("dad", "father", F), ("father", "dad", F),
        ("has_mother", "mother", T), ("mother", "has_mother", T), ("son_of", "child", F), ("daughter_of", "child", F),
        ("child", "parent", T), ("parent", "child", T),
        ("married_to", "spouse", F), ("spouse", "married_to", F), ("married_to", "married_to", T), ("spouse", "spouse", T),
        ("husband_of", "spouse", F), ("wife_of", "spouse", F),
        ("brother_of", "sibling", F), ("sister_of", "sibling", F), ("sibling", "sibling", T),
        ("grandmother_of", "grandparent", F), ("grandparent", "ancestor", F),
    }
    rules = (_rule(("grandparent", "X", "Z"), ("parent", "X", "Y"), ("parent", "Y", "Z")), *_transitive("ancestor", "parent"))
    married = [p for pair in couples for p in pair]
    siblings = [k for kids in by_parents.values() if len(kids) > 1 for k in kids]
    queries = (
        *(atom("parent", "X", c) for c in rng.sample(level2 + level3, 3)),
        *(atom("child", "X", p) for p in rng.sample([person for pair in by_parents for person in pair][:12], 2)),
        *(atom("mother", "X", c) for c in rng.sample(level1 + level2, 2)),
        atom("father", "X", rng.choice(level2)),
        *(atom("spouse", "X", p) for p in rng.sample(married, 2)),
        *(atom("sibling", "X", p) for p in rng.sample(siblings, 2)),
        *(atom("grandparent", "X", c) for c in rng.sample(level2 + level3, 2)),
        *(atom("ancestor", "X", c) for c in rng.sample(level3, 2)),
        atom("parent", "X", "Y"),
    )
    traps = frozenset({"guardian_of", "stepmother_of", "godfather_of", "aunt_of", "uncle_of", "grandmother_of"})
    return KB("family", glosses, tuple(atom(*f) for f in sorted(facts)), rules, keys_of(gold), queries, traps)


def work(seed: int = 11) -> KB:
    """Two companies' reporting trees, management links under four surface forms."""
    rng = random.Random(seed)
    pool = iter(NAMES)
    manages: list[tuple[str, str]] = []
    employer: dict[str, str] = {}
    peers: list[list[str]] = []

    def tree(org: str, widths: Sequence[int]) -> list[str]:
        top = next(pool)
        employer[top] = org
        level, everyone = [top], [top]
        for width in widths:
            nxt = []
            for boss in level:
                team = [next(pool) for _ in range(rng.randint(width - 1, width))]
                for member in team:
                    manages.append((boss, member))
                    employer[member] = org
                peers.append(team)
                nxt += team
            level = nxt
            everyone += nxt
        return everyone

    acme = tree("acme", (3, 2, 2))
    globex = tree("globex", (2, 2))
    outsiders = [next(pool) for _ in range(5)]
    facts: set[tuple[str, str, str]] = set()
    for boss, member in manages:
        facts.add(_choose(rng, [
            (3, ("manager_of", boss, member)), (1, ("supervises", boss, member)),
            (1, ("line_manager_of", boss, member)), (2, ("reports_to", member, boss)),
        ]))
    for person, org in employer.items():
        facts.add(_choose(rng, [
            (2, ("works_for", person, org)), (1, ("employed_by", person, org)),
            (1, ("employs", org, person)), (1, ("on_payroll_of", person, org)),
        ]))
    staff = acme + globex
    managed = {(b, m) for b, m in manages}
    for _ in range(4):
        senior, junior = rng.sample(staff, 2)
        if (senior, junior) not in managed:
            facts.add(("mentor_of", senior, junior))
    for person in outsiders[:3]:
        facts.add(("client_of", person, rng.choice(("acme", "globex"))))
    facts.add(("founded", outsiders[3], "acme"))
    facts.add(("founded", outsiders[4], "globex"))
    for person in outsiders[2:5]:
        facts.add(("invested_in", person, rng.choice(("acme", "globex"))))
    for team in peers:
        if len(team) > 1:
            facts.add(("colleague_of", team[0], team[1]))

    glosses = {
        "manager_of": "{x} is the manager of {y}", "supervises": "{x} supervises {y}",
        "line_manager_of": "{x} is the line manager of {y}", "reports_to": "{x} reports to {y}",
        "works_for": "{x} works for the company {y}", "employed_by": "{x} is employed by the company {y}",
        "employs": "the company {x} employs {y}", "on_payroll_of": "{x} is on the payroll of the company {y}",
        "boss_of": "{x} is the boss of {y}", "subordinate_of": "{x} is a subordinate of {y}",
        "employee_of": "{x} is an employee of the company {y}", "employer_of": "the company {x} is the employer of {y}",
        "skip_level_boss": "{x} is the boss of the boss of {y}", "superior_of": "{x} is above {y} in the chain of command",
        "mentor_of": "{x} is a mentor of {y}", "client_of": "{x} is a client of the company {y}",
        "founded": "{x} founded the company {y}", "invested_in": "{x} invested in the company {y}",
        "colleague_of": "{x} is a colleague of {y}",
    }
    gold = {
        ("manager_of", "supervises", F), ("supervises", "manager_of", F), ("line_manager_of", "manager_of", F),
        ("reports_to", "manager_of", T), ("manager_of", "reports_to", T),
        ("manager_of", "boss_of", F), ("boss_of", "manager_of", F),
        ("subordinate_of", "reports_to", F), ("reports_to", "subordinate_of", F),
        ("works_for", "employed_by", F), ("employed_by", "works_for", F),
        ("employs", "works_for", T), ("works_for", "employs", T), ("on_payroll_of", "works_for", F),
        ("works_for", "employee_of", F), ("employee_of", "works_for", F),
        ("employer_of", "employee_of", T), ("employee_of", "employer_of", T),
        ("colleague_of", "colleague_of", T),
        ("skip_level_boss", "superior_of", F), ("employed_by", "on_payroll_of", F),
    }
    rules = (
        _rule(("skip_level_boss", "X", "Z"), ("boss_of", "X", "Y"), ("boss_of", "Y", "Z")),
        *_transitive("superior_of", "boss_of"),
    )
    queries = (
        *(atom("boss_of", "X", p) for p in rng.sample(staff[1:], 3)),
        *(atom("subordinate_of", "X", b) for b in rng.sample(sorted({b for b, _ in manages}), 2)),
        *(atom("employee_of", "X", org) for org in ("acme", "globex")),
        *(atom("employer_of", "X", p) for p in rng.sample(staff, 2)),
        *(atom("skip_level_boss", "X", p) for p in rng.sample(acme[4:], 2)),
        *(atom("superior_of", "X", p) for p in rng.sample(acme[-6:], 2)),
        atom("boss_of", "X", "Y"),
    )
    traps = frozenset({"mentor_of", "client_of", "founded", "invested_in", "colleague_of"})
    return KB("work", glosses, tuple(atom(*f) for f in sorted(facts)), rules, keys_of(gold), queries, traps)


def geography(seed: int = 13) -> KB:
    """Continents, countries, regions and cities, containment under four surface forms."""
    rng = random.Random(seed)
    syllables = ["ka", "lo", "mi", "ra", "te", "su", "vo", "ne", "pa", "li", "do", "ri", "sa", "ko", "ve", "ta", "mu", "fi"]
    used: set[str] = set()

    def place() -> str:
        while True:
            name = "".join(rng.sample(syllables, 2)) + rng.choice("nsr")
            if name not in used:
                used.add(name)
                return name

    inside: list[tuple[str, str, str]] = []  # (kind, child, parent)
    countries: dict[str, list[str]] = {}
    capitals: dict[str, str] = {}
    cities_of: dict[str, list[str]] = {}
    continents = [place() for _ in range(2)]
    for continent in continents:
        for _ in range(rng.randint(2, 3)):
            country = place()
            inside.append(("country", country, continent))
            countries[country] = []
            cities_of[country] = []
            for _ in range(rng.randint(2, 3)):
                region = place()
                inside.append(("region", region, country))
                countries[country].append(region)
                for _ in range(2):
                    city = place()
                    inside.append(("city", city, region))
                    cities_of[country].append(city)
            capitals[country] = rng.choice(cities_of[country])
    by_continent: dict[str, list[str]] = defaultdict(list)
    for kind, child, parent in inside:
        if kind == "country":
            by_continent[parent].append(child)

    facts: set[tuple[str, str, str]] = set()
    for kind, child, parent in inside:
        options = [(3, ("located_in", child, parent)), (2, ("contains", parent, child))]
        if kind == "city":
            options.append((1, ("city_in", child, parent)))
        else:
            options.append((1, ("part_of", child, parent)))
        facts.add(_choose(rng, options))
    for country, city in capitals.items():
        facts.add(("capital_of", city, country))
    for members in by_continent.values():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if rng.random() < 0.7:
                    facts.add(rng.choice((("borders", a, b), ("borders", b, a), ("shares_border_with", a, b))))
    all_cities = [c for cs in cities_of.values() for c in cs]
    for country, cities in cities_of.items():
        others = [c for c in cities if c != capitals[country]]
        facts.add(("largest_city_of", rng.choice(others), country))
        if rng.random() < 0.5:
            facts.add(("former_capital_of", rng.choice(others), country))
    for _ in range(4):
        a, b = rng.sample(all_cities, 2)
        facts.add(("near", a, b))
        c, d = rng.sample(all_cities, 2)
        facts.add(("twinned_with", c, d))

    glosses = {
        "located_in": "{x} is located in {y}", "contains": "{x} contains {y}", "city_in": "{x} is a city in {y}",
        "part_of": "{x} is part of {y}", "capital_of": "{x} is the capital of {y}",
        "borders": "{x} borders {y}", "shares_border_with": "{x} shares a border with {y}",
        "within": "{x} lies within {y}", "encloses": "{x} encloses {y}", "neighbour_of": "{x} is a neighbour of {y}",
        "inside": "{x} is somewhere inside {y}",
        "largest_city_of": "{x} is the largest city of {y}", "former_capital_of": "{x} is a former capital of {y}",
        "near": "{x} is near {y}", "twinned_with": "{x} is twinned with {y}",
    }
    gold = {
        ("city_in", "located_in", F), ("part_of", "located_in", F), ("located_in", "part_of", F),
        ("contains", "located_in", T), ("located_in", "contains", T), ("capital_of", "located_in", F),
        ("within", "located_in", F), ("located_in", "within", F), ("encloses", "contains", F), ("contains", "encloses", F),
        ("borders", "shares_border_with", F), ("shares_border_with", "borders", F),
        ("borders", "borders", T), ("shares_border_with", "shares_border_with", T),
        ("neighbour_of", "borders", F), ("borders", "neighbour_of", F),
        ("inside", "located_in", F), ("capital_of", "city_in", F), ("largest_city_of", "city_in", F),
        ("near", "near", T), ("twinned_with", "twinned_with", T), ("borders", "near", F),
    }
    # "Located in" is transitive in English, and the facts state only the direct
    # step (city in region, region in country), so the knowledge base says so.
    rules = (*_transitive("inside", "within"),
             _rule(("located_in", "X", "Z"), ("located_in", "X", "Y"), ("located_in", "Y", "Z")))
    regions = [r for rs in countries.values() for r in rs]
    queries = (
        *(atom("within", "X", r) for r in rng.sample(regions, 2)),
        *(atom("within", c, "Y") for c in rng.sample(all_cities, 2)),
        *(atom("encloses", "X", c) for c in rng.sample(all_cities, 2)),
        *(atom("neighbour_of", "X", c) for c in rng.sample(list(countries), 2)),
        *(atom("inside", c, "Y") for c in rng.sample(all_cities, 2)),
        *(atom("inside", "X", c) for c in continents),
        *(atom("capital_of", "X", c) for c in rng.sample(list(countries), 2)),
    )
    traps = frozenset({"largest_city_of", "former_capital_of", "near", "twinned_with"})
    return KB("geography", glosses, tuple(atom(*f) for f in sorted(facts)), rules, keys_of(gold), queries, traps)


def medicine(seed: int = 17) -> KB:
    """Drugs, diseases and symptoms, causation under five surface forms beside correlational traps."""
    rng = random.Random(seed)
    diseases = ["flu", "migraine", "diabetes", "asthma", "neuropathy", "anaemia", "gastritis"]
    symptoms = ["fever", "cough", "headache", "nausea", "fatigue", "wheezing", "numbness", "dizziness", "rash"]
    drugs = ["fluvex", "migrol", "glucan", "breatha", "ferrum", "antacor", "painex", "dermol"]
    facts: set[tuple[str, str, str]] = set()
    causes: set[tuple[str, str]] = set()
    for disease in diseases:
        for symptom in rng.sample(symptoms, rng.randint(2, 3)):
            causes.add((disease, symptom))
    causes |= {("diabetes", "neuropathy"), ("gastritis", "anaemia")}
    side_effects = {(drug, rng.choice(symptoms)) for drug in rng.sample(drugs, 4)}
    for cause, effect in sorted(causes):
        facts.add(_choose(rng, [
            (2, ("causes", cause, effect)), (1, ("leads_to", cause, effect)),
            (2, ("symptom_of", effect, cause)),
        ]))
    for drug, effect in sorted(side_effects):
        facts.add(rng.choice((("side_effect_of", effect, drug), ("causes", drug, effect))))
    treats = {(drug, disease) for drug, disease in zip(drugs[:6], diseases[:6], strict=True)}
    treats |= {("painex", "migraine"), ("painex", "headache"), ("dermol", "rash")}
    for drug, target in sorted(treats):
        facts.add(_choose(rng, [
            (2, ("treats", drug, target)), (1, ("cures", drug, target)),
            (1, ("relieves", drug, target)),
        ]))
    for _ in range(4):
        a, b = rng.choice(diseases), rng.choice(symptoms)
        if (a, b) not in causes:
            facts.add(("correlated_with", a, b))
    for drug in rng.sample(drugs, 3):
        facts.add(("associated_with", drug, rng.choice(diseases)))
        facts.add(("prevents", rng.choice(drugs), rng.choice(diseases)))
        facts.add(("contraindicated_for", drug, rng.choice(diseases)))

    glosses = {
        "causes": "{x} causes {y}", "leads_to": "{x} leads to {y}", "symptom_of": "{x} is a symptom of {y}",
        "side_effect_of": "{x} is a side effect of {y}", "treats": "{x} treats {y}", "cures": "{x} cures {y}",
        "relieves": "{x} relieves {y}", "remedy_for": "{x} is a remedy for {y}", "brings_about": "{x} brings about {y}",
        "result_of": "{x} is a result of {y}", "indirectly_brings_about": "{x} brings about something that brings about {y}",
        "correlated_with": "{x} is correlated with {y}", "associated_with": "{x} is associated with {y}",
        "prevents": "{x} prevents {y}", "contraindicated_for": "{x} is contraindicated for {y}",
    }
    gold = {
        ("cures", "treats", F), ("relieves", "treats", F), ("treats", "remedy_for", F), ("remedy_for", "treats", F),
        ("leads_to", "causes", F), ("causes", "leads_to", F), ("symptom_of", "causes", T), ("side_effect_of", "causes", T),
        ("causes", "brings_about", F), ("brings_about", "causes", F), ("result_of", "causes", T), ("causes", "result_of", T),
        ("cures", "relieves", F), ("correlated_with", "associated_with", F), ("causes", "associated_with", F),
        ("treats", "associated_with", F), ("associated_with", "associated_with", T), ("correlated_with", "correlated_with", T),
    }
    rules = (_rule(("indirectly_brings_about", "X", "Z"), ("brings_about", "X", "Y"), ("brings_about", "Y", "Z")),)
    queries = (
        *(atom("remedy_for", "X", d) for d in ("flu", "migraine", "headache")),
        *(atom("brings_about", d, "Y") for d in rng.sample(diseases, 3)),
        *(atom("brings_about", "X", s) for s in rng.sample(symptoms, 2)),
        *(atom("result_of", s, "Y") for s in rng.sample(symptoms, 2)),
        atom("indirectly_brings_about", "diabetes", "Y"),
        atom("indirectly_brings_about", "gastritis", "Y"),
        atom("remedy_for", "X", "Y"),
    )
    traps = frozenset({"correlated_with", "associated_with", "prevents", "contraindicated_for"})
    return KB("medicine", glosses, tuple(atom(*f) for f in sorted(facts)), rules, keys_of(gold), queries, traps)


#: Every benchmark by name, each a function of its seed.
BENCHMARKS: dict[str, Callable[..., KB]] = {
    "family": family,
    "work": work,
    "geography": geography,
    "medicine": medicine,
}
