from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from .schemas import Critique


def reflect_and_refine(
    role: str,
    context: str,
    initial,
    reviser: Callable,
    critic: Callable[[str, str, object], Critique],
    max_reflections: int,
    *,
    verbose: bool = True,
) -> Tuple[object, Optional[List[dict]]]:
    decision = initial
    trace: List[dict] = []
    approved_decision = None

    for round_idx in range(1, max_reflections + 2):
        critique = critic(role, context, decision)
        entry = {
            "round": round_idx,
            "verdict": critique.verdict,
            "approved": critique.approved,
            "issues": critique.issues,
            "guidance": critique.guidance,
        }
        trace.append(entry)
        if critique.approved:
            if verbose:
                issues = " ".join((critique.issues or "").split())
                note = f": advisory -- {issues[:150]}" if critique.verdict == "advise" else ""
                print(f"      [REFLECT {role}] round {round_idx}: "
                      f"{critique.verdict}d{note}")
            approved_decision = decision
            break
        if round_idx > max_reflections:
            entry["unresolved"] = True
            if verbose:
                issues = " ".join((critique.issues or "").split())
                print(f"      [REFLECT {role}] round {round_idx}: UNRESOLVED, "
                      f"applying anyway -- {issues[:120]}")
            break
        if verbose:
            issues = " ".join((critique.issues or "").split())
            print(f"      [REFLECT {role}] round {round_idx}: revise -- {issues[:150]}")
        decision = reviser(decision, critique)

    return (approved_decision if approved_decision is not None else decision), trace or None


def make_mock_reviser(produce: Callable):
    def reviser(_previous, critique):
        return produce(critique)
    return reviser
