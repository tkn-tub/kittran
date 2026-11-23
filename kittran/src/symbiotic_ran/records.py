from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import List


@dataclass
class TenantState:

    id: str
    name: str
    intent_mbps: float
    min_acceptable_mbps: float
    utility_weight: float
    willingness_to_pay: float

    first_proposed_mbps: float = 0.0
    final_proposed_mbps: float = 0.0
    final_accept: bool = False
    rounds_seen: int = 0

    conceded_mbps: float = 0.0
    below_minimum: bool = False
    satisfied: bool = False

    def observe(self, proposed_mbps: float, accept: bool) -> None:
        if self.rounds_seen == 0:
            self.first_proposed_mbps = proposed_mbps
        self.final_proposed_mbps = proposed_mbps
        self.final_accept = accept
        self.rounds_seen += 1

    def finalize(self, achievable_mbps: float, consensus_mbps: float, ran_off: bool) -> None:
        self.conceded_mbps = round(
            self.first_proposed_mbps - self.final_proposed_mbps, 2)
        self.below_minimum = achievable_mbps < self.min_acceptable_mbps
        self.satisfied = (not ran_off) and consensus_mbps >= self.min_acceptable_mbps

    def summary(self) -> str:
        return (
            f"{self.id} ({self.name}): wanted {self.intent_mbps:.1f}, floor "
            f"{self.min_acceptable_mbps:.1f}, settled at "
            f"{self.final_proposed_mbps:.1f} Mbps (accepted={self.final_accept}, "
            f"conceded {self.conceded_mbps:+.1f}), "
            f"{'SATISFIED' if self.satisfied else 'UNSATISFIED'}"
            f"{', channel below its floor' if self.below_minimum else ''}"
        )


@dataclass
class NegotiationRecord:

    step: int
    phase: int
    phase_label: str
    achievable_mbps: float
    reachable_mbps: float
    confidence_interval: List[float]
    interval_center_mbps: float

    tenant_states: List[TenantState]
    rounds_used: int

    consensus_mbps: float
    ran_off: bool
    decision: str
    decision_reasoning: str = ""
    shares: dict = field(default_factory=dict)

    max_rounds: int = 0
    half_width_mbps: float = 0.0
    ran_off_threshold_mbps: float = 0.0

    backend: str = "unknown"
    agent_calls: int = 0
    created_at: str = ""

    @property
    def unsatisfied_tenants(self) -> List[str]:
        return [t.id for t in self.tenant_states if not t.satisfied]

    @property
    def sla_fraction(self) -> float:
        if self.ran_off or self.achievable_mbps <= 0:
            return 0.0
        return self.consensus_mbps / self.achievable_mbps

    def summary(self) -> str:
        what = "RAN OFF" if self.ran_off else f"SLA={self.consensus_mbps:.1f} Mbps"
        return (f"{what} via {self.decision} in {self.rounds_used} round(s) "
                f"[{self.agent_calls} agent calls, phase {self.phase}]")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sla_fraction"] = round(self.sla_fraction, 4)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "NegotiationRecord":
        known = {f.name for f in fields(cls)}
        ts_known = {f.name for f in fields(TenantState)}

        states = [
            TenantState(**{k: v for k, v in raw.items() if k in ts_known})
            for raw in d.get("tenant_states", [])
        ]

        d = {k: v for k, v in d.items() if k in known}
        d["tenant_states"] = states
        return cls(**d)
