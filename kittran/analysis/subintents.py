"""Are the L2 manager's sub-intents well-formed? Rule checks, no model."""

from __future__ import annotations

import re

from runs import load_decisions, parser, rule

DB_VALUE = re.compile(r"\b\d+(?:\.\d+)?\s*dB\b", re.IGNORECASE)
TARGET = re.compile(r"TARGET\s+(\d+(?:\.\d+)?)\s*Mbit/s")
SLA_TOTAL = re.compile(r"shared SLA of (\d+(?:\.\d+)?) Mbit/s")
SHARE = re.compile(r"\b(FWA|MTC|BE)\s*\((\d+)%\)")
OBJECTIVE_WORDS = ("ceiling", "lowest snr", "minimum snr", "minimum link",
                   "if the cell is short", "short of", "shortfall", "hold", "keep")


def score(record: dict) -> dict:
    intent = record.get("operator_intent") or ""
    pc = record["decision"].get("pc_subintent") or ""
    ulra = record["decision"].get("ul_ra_subintent") or ""
    targets = TARGET.findall(intent) + SLA_TOTAL.findall(intent)
    shares = {name: int(pct) for name, pct in SHARE.findall(intent)}
    top = max(shares, key=shares.get) if shares else None
    lowered = pc.lower()
    verdicts = [x.get("verdict") or ("approve" if x.get("approved") else "revise")
                for x in (record.get("reflection") or [])]
    return {
        "no_db_value": not DB_VALUE.search(pc),
        "target_carried": (not targets) or any(t in ulra for t in targets),
        "top_slice_named": (top is None) or (top in ulra),
        "objectives": any(w in lowered for w in OBJECTIVE_WORDS),
        "verdicts": verdicts,
        "unresolved": any(x.get("unresolved") for x in (record.get("reflection") or [])),
        "bypassed": bool(record.get("bypassed")),
    }


RULES = ("no_db_value", "target_carried", "top_slice_named", "objectives")


def main() -> None:
    args = parser(__doc__.splitlines()[0]).parse_args()
    decisions = [d for d in load_decisions(args.run_dir) if d["agent"] == "l2_manager"]
    if not decisions:
        raise SystemExit("no l2_manager decision in this run")
    scored = [(d, score(d)) for d in decisions]
    if all(s["bypassed"] for _, s in scored):
        raise SystemExit("the L2 manager was bypassed in this run (--rrm); nothing to score")
    scored = [(d, s) for d, s in scored if not s["bypassed"]]

    rule(f"Sub-intent rule checks -- {args.run_dir}")
    print(f"  {'cycle':>5}  {'phase':<22} {'dB':>3} {'tgt':>3} {'top':>3} {'obj':>3}  critic")
    for d, s in scored:
        marks = "  ".join("ok " if s[r] else "BAD" for r in RULES)
        critic = " ".join(v[:3] for v in s["verdicts"]) or "-"
        if s["unresolved"]:
            critic += " (unresolved)"
        print(f"  {d['cycle']:>5}  {d['phase']:<22} {marks}  {critic}")
    print("\n  dB: no numeral in dB in the pc_subintent; tgt: a TARGET Mbit/s figure from\n"
          "  the operator intent appears in the ul_ra_subintent; top: the largest-share\n"
          "  slice is named there; obj: the pc_subintent speaks in objectives (ceiling,\n"
          "  link minimum, shortfall). critic: the reviewer's verdicts, in order.")

    rule("Rates")
    n = len(scored)
    for r in RULES:
        k = sum(1 for _, s in scored if s[r])
        print(f"  {r:<18} {k:>3}/{n}  {k / n * 100:5.1f}%")
    clean = sum(1 for _, s in scored if all(s[r] for r in RULES))
    print(f"  {'all four':<18} {clean:>3}/{n}  {clean / n * 100:5.1f}%")

    rule("Rules against the reviewer")
    reviewed = [(d, s) for d, s in scored if s["verdicts"]]
    if not reviewed:
        print("  reflection was off for l2 in this run")
        return
    agree = disagree_rules = disagree_critic = 0
    for _, s in reviewed:
        rules_ok = all(s[r] for r in RULES)
        critic_ok = s["verdicts"][0] in ("approve", "advise")
        if rules_ok == critic_ok:
            agree += 1
        elif rules_ok:
            disagree_critic += 1
        else:
            disagree_rules += 1
    print(f"  {agree} agree, {disagree_critic} rejected by the critic though the rules pass,")
    print(f"  {disagree_rules} passed by the critic though a rule fails -- on the FIRST proposal")
    print("  (the critic sees semantics the rules cannot; the rules see stale numbers\n"
          "   the critic has been measured to miss -- neither is ground truth)")


if __name__ == "__main__":
    main()
