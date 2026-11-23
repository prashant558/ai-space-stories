#!/usr/bin/env python3
"""lore_schema.py — load, validate and sample the Space Lore Engine world schema.

Plain words: the schema (world_schema.json) lists the kinds of things in the galaxy (planets, factions, ...)
and the properties each can have. This tool turns it into random, consistent "tag lines" such as

    type: planet | name: Veyra-7 | climate: ice | size: medium | life: none | danger: high | mood: eerie | length: short

Notebook 05 sends those tags to a teacher model (Qwen2.5-7B) to write the lore; notebook 06 teaches our model tags -> lore.

Usage:
    python lore_schema.py --validate                 # check the schema, exit 1 on problems
    python lore_schema.py --sample 5 --seed 1        # print random tag lines
    python lore_schema.py --stats                    # how many combinations exist per type
    python lore_schema.py --selftest                 # validate + sample 20k entities + check every rule
"""
import argparse
import json
import math
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCHEMA = next((p for p in (os.path.join(HERE, "world_schema.json"),
                       os.path.join(HERE, "..", "data", "world_schema.json")) if os.path.exists(p)),
                      os.path.join(HERE, "world_schema.json"))   # next to this file, or the repo's data/ folder
NAME_STYLES = {"planet", "system", "faction", "species", "ship", "artifact", "person", "quest", "station"}


# ---------------------------------------------------------------- loading + validation
def load_schema(path=DEFAULT_SCHEMA):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _is_special(spec):
    return isinstance(spec, str) and spec.startswith("@")


def validate_schema(S):
    """Returns a list of problems (empty list = valid)."""
    p = []
    for key in ("universe", "lengths", "type_weights", "types", "names", "link_probability"):
        if key not in S:
            p.append(f"missing top-level key '{key}'")
    if p:
        return p
    types = S["types"]
    if set(S["type_weights"]) != set(types):
        p.append(f"type_weights keys {sorted(S['type_weights'])} != types {sorted(types)}")
    for name, (lo, hi) in S["lengths"].items():
        if not 0 < lo < hi:
            p.append(f"length '{name}' must be 0 < min < max")
    if not 0 <= S["link_probability"] <= 1:
        p.append("link_probability must be in [0, 1]")
    for t, T in types.items():
        for k in ("description", "name_style", "fields", "optional", "links", "rules"):
            if k not in T:
                p.append(f"{t}: missing '{k}'")
        if p:
            continue
        if T["name_style"] not in NAME_STYLES:
            p.append(f"{t}: unknown name_style '{T['name_style']}'")
        for f, spec in T["fields"].items():
            if _is_special(spec):
                if not (spec == "@character" or re.fullmatch(r"@int:\d+:\d+", spec)):
                    p.append(f"{t}.{f}: unknown special spec '{spec}'")
            elif not (isinstance(spec, list) and spec and all(isinstance(v, str) for v in spec)):
                p.append(f"{t}.{f}: must be a non-empty list of strings")
            elif len(set(spec)) != len(spec):
                p.append(f"{t}.{f}: duplicate values")
        if "mood" not in T["fields"]:
            p.append(f"{t}: every type needs a 'mood' field (it drives the writing style)")
        for f in T["optional"]:
            if f not in T["fields"]:
                p.append(f"{t}: optional field '{f}' is not a field")
        for rel, target in T["links"].items():
            if target not in types:
                p.append(f"{t}: link '{rel}' points to unknown type '{target}'")
        for i, r in enumerate(T["rules"]):
            for side in ("if", "then"):
                for f, v in r.get(side, {}).items():
                    if f not in T["fields"] or _is_special(T["fields"][f]):
                        p.append(f"{t} rule {i}: '{f}' is not a list field")
                        continue
                    vals = v if isinstance(v, list) else [v]
                    bad = [x for x in vals if x not in T["fields"][f]]
                    if bad:
                        p.append(f"{t} rule {i}: values {bad} not allowed for '{f}'")
    N = S["names"]
    for k in ("syllables_start", "syllables_mid", "syllables_end", "first_names", "ai_names"):
        if not N.get(k):
            p.append(f"names.{k} is empty")
    for k, v in N.items():
        if k.endswith("_pattern"):
            for pat in v:
                if not re.search(r"\{[NAB]\}", pat):
                    p.append(f"names.{k}: pattern '{pat}' has no {{N}}/{{A}}/{{B}} slot")
    return p


# ---------------------------------------------------------------- names
class NameGen:
    """Invents original names from syllables, so the training data never borrows franchise names."""

    def __init__(self, S, rng):
        self.N, self.rng, self.used = S["names"], rng, set()
        self.banned = {b.lower() for b in S["universe"]["banned_terms"]}

    def _word(self):
        N, r = self.N, self.rng
        w = r.choice(N["syllables_start"]) + ("" if r.random() < 0.35 else r.choice(N["syllables_mid"]))
        return w + r.choice(N["syllables_end"])

    def _fill(self, pattern):
        N, r = self.N, self.rng
        return (pattern.replace("{N}", self._word()).replace("{A}", r.choice(N["ship_words_a"]))
                .replace("{B}", r.choice(N["ship_words_b"])))

    def _make(self, style, role=None):
        N, r = self.N, self.rng
        if style == "planet":
            return self._word() + r.choice(N["planet_suffix"])
        if style == "system":
            return (r.choice(N["system_prefix"]) + self._word() + r.choice(N["system_suffix"])).strip()
        if style == "species":
            return "the " + self._word() + r.choice(["i", "ans", "ari", "", "eth"])
        if style == "person":
            if role == "AI companion":
                return r.choice(N["ai_names"]) + ("" if r.random() < 0.5 else f"-{r.randint(2, 99)}")
            return f"{r.choice(N['first_names'])} {self._word()}"
        key = {"faction": "faction_pattern", "ship": "ship_pattern", "artifact": "artifact_pattern",
               "station": "station_pattern", "quest": "quest_pattern"}[style]
        return self._fill(r.choice(N[key]))

    def new(self, style, role=None, max_tries=200):
        for _ in range(max_tries):
            name = self._make(style, role)
            low = name.lower()
            if low not in self.used and not any(b in low for b in self.banned):
                self.used.add(low)
                return name
        raise RuntimeError(f"could not find a new unique '{style}' name — add more syllables/patterns")


# ---------------------------------------------------------------- sampling
def _rule_ok(entity, rule):
    if all(entity.get(f) == v for f, v in rule["if"].items()):
        return all(entity.get(f) in (v if isinstance(v, list) else [v]) for f, v in rule["then"].items())
    return True


def sample_entity(S, rng, names, etype=None, optional_p=0.6):
    """One random, rule-consistent entity as a dict (insertion order = tag order)."""
    if etype is None:
        types, w = zip(*S["type_weights"].items())
        etype = rng.choices(types, weights=w)[0]
    T = S["types"][etype]
    for _ in range(100):                                  # rejection sampling until all rules hold
        fields = {}
        for f, spec in T["fields"].items():
            if f in T["optional"] and rng.random() > optional_p:
                continue
            if spec == "@character":
                fields[f] = None                          # filled after the rules are satisfied
            elif isinstance(spec, str) and spec.startswith("@int:"):
                lo, hi = map(int, spec.split(":")[1:])
                fields[f] = str(rng.randint(lo, hi))
            else:
                fields[f] = rng.choice(spec)
        if all(_rule_ok(fields, r) for r in T["rules"]):
            break
    else:
        raise RuntimeError(f"{etype}: rules too strict, no valid combination found in 100 tries")

    e = {"type": etype, "name": names.new(T["name_style"], role=fields.get("role"))}
    for f, v in fields.items():
        e[f] = names.new("person") if v is None else v
    for rel, target in T["links"].items():                # optional references to other entities
        if rng.random() < S["link_probability"]:
            e[rel] = names.new(S["types"][target]["name_style"])
    e["length"] = rng.choice(list(S["lengths"]))
    return e


def to_tags(entity):
    return " | ".join(f"{k}: {v}" for k, v in entity.items())


def parse_tags(line):
    out = {}
    for part in line.split(" | "):
        k, _, v = part.partition(": ")
        out[k.strip()] = v.strip()
    return out


def word_range(S, entity):
    return S["lengths"][entity["length"]]


def check_lore(S, entity, text):
    """Problems with a generated text for this entity (used to filter Phase 2 data). Empty list = OK."""
    probs = []
    low = text.lower()
    words = len(text.split())
    lo, hi = word_range(S, entity)
    if not (lo * 0.8 <= words <= hi * 1.25):
        probs.append(f"length {words} words, expected {lo}-{hi}")
    if entity["name"].lower().removeprefix("the ") not in text.lower():
        probs.append(f"name '{entity['name']}' missing")
    for rel in S["types"][entity["type"]]["links"]:            # linked names must appear too
        if rel in entity and entity[rel].lower().removeprefix("the ") not in text.lower():
            probs.append(f"{rel} '{entity[rel]}' missing")
    if entity["type"] == "ship_log":       # first-person log: the author says "I", not their own name
        if "day" in entity and not re.match(rf"\s*day\s*{entity['day']}\b", low):
            probs.append(f"log does not start with 'Day {entity['day']}'")
        if not re.search(r"\b(i|i'm|i've|my|me|we|our|us)\b", low):
            probs.append("log is not written in first person")
    hits = [b for b in S["universe"]["banned_terms"] if re.search(rf"\b{re.escape(b.lower())}\b", low)]
    if hits:
        probs.append(f"franchise terms {hits}")
    return probs


def combinations(S, etype):
    T = S["types"][etype]
    n = 1
    for f, spec in T["fields"].items():
        k = len(spec) if isinstance(spec, list) else (int(spec.split(":")[2]) - int(spec.split(":")[1]) + 1
                                                       if spec.startswith("@int:") else 1)
        n *= k + (1 if f in T["optional"] else 0)
    return n * len(S["lengths"])


# ---------------------------------------------------------------- self-test
def selftest(S, n=20000, seed=0):
    probs = validate_schema(S)
    assert not probs, probs
    rng = random.Random(seed)
    names = NameGen(S, rng)
    counts = {t: 0 for t in S["types"]}
    for _ in range(n):
        e = sample_entity(S, rng, names)
        T = S["types"][e["type"]]
        counts[e["type"]] += 1
        for r in T["rules"]:
            assert _rule_ok(e, r), (e, r)
        assert parse_tags(to_tags(e)) == e, "tag round-trip failed"
        for f in T["fields"]:
            assert f in e or f in T["optional"], (f, e)
    exp = {t: n * w / sum(S["type_weights"].values()) for t, w in S["type_weights"].items()}
    for t in counts:
        assert abs(counts[t] - exp[t]) < 0.1 * exp[t] + 50, (t, counts[t], exp[t])
    assert len(names.used) > 0.99 * n, "names are not unique enough"
    good = {"type": "planet", "name": "Veyra-7", "length": "short"}
    assert check_lore(S, good, "Veyra-7 is " + "cold " * 70) == []
    assert check_lore(S, good, "A planet. " * 5)            # too short + no name
    assert check_lore(S, good, "Veyra-7 has a Jedi temple. " + "x " * 70)
    linked = {"type": "faction", "name": "the Drenex Syndicate", "homeworld": "Quaix", "length": "short"}
    assert check_lore(S, linked, "Drenex Syndicate rules. " + "y " * 70), "missing link not caught"
    assert check_lore(S, linked, "Drenex Syndicate rules from Quaix. " + "y " * 70) == []
    log = {"type": "ship_log", "name": "The Long Quiet", "author": "Mara Velor", "day": "41", "length": "short"}
    assert check_lore(S, log, "Day 41. The Long Quiet drifts. I checked the drive again. " + "z " * 60) == []
    assert check_lore(S, log, "Mara Velor checked the Long Quiet drive. " + "z " * 60), "bad log not caught"
    print(f"selftest OK: {n:,} entities, all rules hold, {len(names.used):,} unique names, "
          f"type mix {dict(sorted(counts.items(), key=lambda x: -x[1]))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema", default=DEFAULT_SCHEMA)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--type", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    S = load_schema(a.schema)
    probs = validate_schema(S)
    if probs:
        print("Schema problems:\n- " + "\n- ".join(probs)); sys.exit(1)
    if a.validate:
        print(f"schema OK: {len(S['types'])} entity types")
    if a.stats:
        total = 0
        for t in S["types"]:
            c = combinations(S, t); total += c
            print(f"  {t:12s} {len(S['types'][t]['fields'])} fields  ~10^{math.log10(c):.1f} tag combinations")
        print(f"  total ~10^{math.log10(total):.1f} combinations (before names) — far more than 20k examples")
    if a.sample:
        rng = random.Random(a.seed)
        names = NameGen(S, rng)
        for _ in range(a.sample):
            print(to_tags(sample_entity(S, rng, names, a.type)))
    if a.selftest:
        selftest(S)
    if not (a.validate or a.stats or a.sample or a.selftest):
        ap.print_help()


if __name__ == "__main__":
    main()
