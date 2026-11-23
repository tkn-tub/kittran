"""What the run spent on inference, and what the critic did with its share."""

from __future__ import annotations

from collections import Counter

from runs import load_decisions, load_negotiations, parser, rule

NEGOTIATION = "negotiation"


def main() -> None:
    args = parser(__doc__.splitlines()[0]).parse_args()
    decisions = load_decisions(args.run_dir)
    negotiations = load_negotiations(args.run_dir)

    acting = [d for d in decisions if d["agent"] != NEGOTIATION]
    traces = [x for d in acting for x in (d.get("reflection") or [])]
    revisions = [x for x in traces if not x["approved"]]
    advisories = [x for x in traces if x.get("verdict") == "advise"]
    neg_calls = sum(n.get("agent_calls", 0) for n in negotiations)

    rule(f"Call budget -- {args.run_dir}")
    counted = neg_calls + len(acting) + len(revisions) + len(traces)
    print(f"  negotiation (tenant + mediator)      {neg_calls:>4}")
    print(f"  acting-agent proposals (l2/pc/ulra)  {len(acting):>4}")
    print(f"  reflection critic                    {len(traces):>4}")
    print(f"  revision calls                       {len(revisions):>4}")
    print(f"  {'total':<36} {counted:>4}")

    timed = [d for d in decisions if "llm_seconds" in d]
    rule("Measured inference")
    if not timed:
        print("  This run predates per-call timing, so no duration was recorded.")
        print("  Derive wall-clock from file timestamps, and treat it as an estimate.")
    else:
        seconds = sum(d["llm_seconds"] for d in timed)
        calls = sum(d["llm_calls"] for d in timed)
        retries = sum(d["llm_retries"] for d in timed)
        print(f"  {calls} call(s) over {seconds / 60:.1f} min "
              f"-> {seconds / calls:.1f} s/call")
        print(f"  {retries} retry(ies) -- a malformed response resampled")
        holds = sum(d.get("llm_fallbacks", 0) for d in timed)
        if holds:
            print(f"  {holds} hold(s) -- a call that gave up after every attempt")
        with_tokens = [d for d in timed if "llm_tokens_in" in d]
        if with_tokens:
            tin = sum(d["llm_tokens_in"] for d in with_tokens)
            tout = sum(d["llm_tokens_out"] for d in with_tokens)
            n = sum(d["llm_calls"] for d in with_tokens)
            print(f"  {tin:,} prompt + {tout:,} completion tokens over {n} call(s) "
                  f"-> {tin / n:,.0f} in / {tout / n:,.0f} out per call")
        slowest = sorted(timed, key=lambda d: -d["llm_seconds"])[:5]
        print("\n  slowest decisions:")
        for d in slowest:
            print(f"    cycle {d['cycle']:>3} {d['agent']:<12} "
                  f"{d['llm_seconds']:>7.1f}s over {d['llm_calls']} call(s)")

    graded = [d for d in decisions if "clamped" in d]
    rule("Guardrails")
    if not graded:
        print("  This run predates the clamp record.")
    else:
        print("  proposals the clamp rewrote, by role (the blocked-out-of-bounds rate):")
        for role in ("negotiation", "pc_agent", "ulra_agent"):
            mine = [d for d in graded if d["agent"] == role]
            if not mine:
                continue
            hit = [d for d in mine if d["clamped"]]
            keys = Counter(k for d in hit for k in d["clamped"])
            top = ", ".join(f"{k} x{n}" for k, n in keys.most_common(3))
            print(f"    {role:<12} {len(hit):>3}/{len(mine):<3}"
                  f"{('  ' + top) if top else ''}")

    rule("Reflection")
    if not traces:
        print("  Reflection was off for every role in this run.")
        return
    print(f"  {len(traces)} critic round(s): {len(traces) - len(revisions)} shipped, "
          f"{len(revisions)} sent back")
    if advisories:
        print(f"  of those shipped, {len(advisories)} carried an advisory")
    else:
        print("  (no advisories -- this run predates the three-way verdict, so the")
        print("   critic could only approve or reject)")
    by_role = Counter(d["agent"] for d in acting
                      if any(not x["approved"] for x in (d.get("reflection") or [])))
    if by_role:
        print("\n  decisions sent back, by role:")
        for role, n in by_role.most_common():
            total = sum(1 for d in acting if d["agent"] == role)
            print(f"    {role:<12} {n}/{total}")
    unresolved = [x for x in traces if x.get("unresolved")]
    if unresolved:
        print(f"\n  {len(unresolved)} decision(s) shipped still rejected "
              f"(reflection budget spent)")


if __name__ == "__main__":
    main()
