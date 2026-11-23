from __future__ import annotations

import operator
from typing import Annotated, List, Literal, TypedDict

from pydantic import BaseModel, Field

Priority = Literal["low", "normal", "high"]


class TenantProposal(BaseModel):

    proposed_mbps: float = Field(
        description="The throughput in Mbps this tenant proposes for the shared SLA."
    )
    accept: bool = Field(
        description="Whether this tenant accepts the mediator's current suggested SLA."
    )
    reasoning: str = Field(
        description="One short sentence explaining the tenant's position."
    )


class MediatorDecision(BaseModel):

    suggested_sla_mbps: float = Field(
        description="The mediator's suggested consensus SLA in Mbps for this round "
                    "(total cell throughput the tenants collectively commit to)."
    )
    consensus_reached: bool = Field(
        description="Whether the mediator judges that the tenants have reached consensus."
    )
    switch_ran_off: bool = Field(
        description="Whether the mediator recommends switching the RAN off entirely to save cost."
    )
    share_tenant_A: float = Field(
        default=0.34,
        description="Fraction of delivered capacity for tenant_A (Premium Video, "
                    "slice FWA). The three shares should sum to 1.0.",
    )
    share_tenant_B: float = Field(
        default=0.33,
        description="Fraction of delivered capacity for tenant_B (Fleet Telemetry, "
                    "slice MTC). The three shares should sum to 1.0.",
    )
    share_tenant_C: float = Field(
        default=0.33,
        description="Fraction of delivered capacity for tenant_C (Best-Effort IoT, "
                    "slice BE). The three shares should sum to 1.0.",
    )
    approve_energy_ues: list[str] = Field(
        default_factory=list,
        description="Only when a RADIO CAPACITY REPORT is shown: the UE ids whose "
                    "extra transmit power the tenants accept, so Power Control "
                    "must drive them to their channel ceiling. Leave empty to "
                    "keep the SLA within the capacity held now. Ignored otherwise.",
    )
    reasoning: str = Field(
        description="One or two sentences explaining the mediator's decision."
    )

    def shares(self) -> dict:
        return {
            "tenant_A": self.share_tenant_A,
            "tenant_B": self.share_tenant_B,
            "tenant_C": self.share_tenant_C,
        }


class NegotiationOutcome(BaseModel):

    phase: int
    label: str
    consensus_mbps: float
    ran_off: bool
    rounds_used: int
    confidence_interval: List[float]
    achievable_mbps: float


class L2Decision(BaseModel):

    pc_subintent: str = Field(
        description="1-2 sentence OBJECTIVE for the Power Control agent, which "
                    "sets a per-UE target SNR every cycle. Name UEs, the "
                    "direction to err in, and Mbit/s goals. Never a dB value: "
                    "the agent re-reads this every cycle against fresh channel "
                    "ceilings, and a dB number written once goes stale."
    )
    ul_ra_subintent: str = Field(
        description="1-2 sentence instruction for the UL Resource Allocation agent, "
                    "which sets a per-slice throttle in Mbit/s and a priority."
    )


class PCDecision(BaseModel):

    fwa_1_target_snr_db: float = Field(description="New target SNR in dB for UE fwa_1")
    fwa_1_reasoning: str = Field(description="One-sentence reason for fwa_1's value")
    fwa_2_target_snr_db: float = Field(description="New target SNR in dB for UE fwa_2")
    fwa_2_reasoning: str = Field(description="One-sentence reason for fwa_2's value")
    mtc_1_target_snr_db: float = Field(description="New target SNR in dB for UE mtc_1")
    mtc_1_reasoning: str = Field(description="One-sentence reason for mtc_1's value")
    iot_1_target_snr_db: float = Field(description="New target SNR in dB for UE iot_1")
    iot_1_reasoning: str = Field(description="One-sentence reason for iot_1's value")

    def per_ue(self) -> dict:
        return {
            "fwa_1": (self.fwa_1_target_snr_db, self.fwa_1_reasoning),
            "fwa_2": (self.fwa_2_target_snr_db, self.fwa_2_reasoning),
            "mtc_1": (self.mtc_1_target_snr_db, self.mtc_1_reasoning),
            "iot_1": (self.iot_1_target_snr_db, self.iot_1_reasoning),
        }


class ULRADecision(BaseModel):

    fwa_throttle_mbps: float = Field(description="Max throughput in Mbit/s for the FWA slice")
    fwa_priority: Priority = Field(description="Scheduler priority for FWA")
    fwa_reasoning: str = Field(description="One-sentence reason for the FWA decision")
    mtc_throttle_mbps: float = Field(description="Max throughput in Mbit/s for the MTC slice")
    mtc_priority: Priority = Field(description="Scheduler priority for MTC")
    mtc_reasoning: str = Field(description="One-sentence reason for the MTC decision")
    be_throttle_mbps: float = Field(description="Max throughput in Mbit/s for the BE slice")
    be_priority: Priority = Field(description="Scheduler priority for BE")
    be_reasoning: str = Field(description="One-sentence reason for the BE decision")

    def per_slice(self) -> dict:
        return {
            "FWA": (self.fwa_throttle_mbps, self.fwa_priority, self.fwa_reasoning),
            "MTC": (self.mtc_throttle_mbps, self.mtc_priority, self.mtc_reasoning),
            "BE": (self.be_throttle_mbps, self.be_priority, self.be_reasoning),
        }


class RRMDecision(PCDecision, ULRADecision):
    pass


class Critique(BaseModel):

    verdict: Literal["approve", "advise", "revise"] = Field(
        description="'approve' -- the proposal is correct, apply as-is. "
                    "'advise' -- the proposal should be applied, but note a "
                    "concern in `issues` (use this for a target above its "
                    "channel ceiling, or a jump wider than the per-cycle clamp "
                    "when the direction is right). "
                    "'revise' -- the proposal is wrong and must be corrected; "
                    "reserve this for errors that change what the agent does."
    )
    issues: str = Field(
        default="",
        max_length=800,
        description="At most three short sentences, under 800 characters, "
                    "naming the concrete problems with the proposal. State "
                    "the conclusion only -- do not reason, weigh or "
                    "deliberate in this field. Empty string when the verdict "
                    "is 'approve'."
    )
    guidance: str = Field(
        default="",
        max_length=400,
        description="At most two short sentences, under 400 characters, "
                    "telling the agent what to set instead. No deliberation. "
                    "Required for 'revise'; empty otherwise."
    )

    @property
    def approved(self) -> bool:
        return self.verdict != "revise"


class SimState(TypedDict):

    step: int
    cycle: int
    phase: int
    phase_label: str

    base_mcs_index: int
    ues: dict
    slices: dict
    cell: dict
    prb_fraction: float
    delivered_mbps: float
    last_delivered_mbps: float
    best_delivered_mbps: float
    achievable_mbps: float
    reachable_mbps: float

    operator_intent: str
    pc_subintent: str
    ul_ra_subintent: str
    intent_serial: int
    l2_seen_serial: int
    pc_seen_serial: int
    ulra_seen_serial: int
    rrm_exhausted_cycles: int
    negotiated_achievable_mbps: float
    energy_approved_ues: list
    energy_accepted_db: dict
    reported_ues: dict | None
    telemetry_trusted: dict

    sla_mbps: float
    tenant_shares: dict
    slice_targets: dict
    tenant_intents: dict
    ran_off: bool
    pending_negotiation: bool
    negotiation_reason: str

    history: list
    negotiation_log: list
    decisions: Annotated[list, operator.add]
