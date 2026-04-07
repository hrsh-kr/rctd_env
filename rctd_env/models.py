"""
RCTD Environment — Type-safe data contracts.

Inherits from the OpenEnv framework base classes (Action, Observation, State)
to ensure full spec compliance and automatic WebSocket/REST integration.
"""

from typing import Any, Dict, List, Optional

from openenv.core import Action, Observation, State
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class RCTDAction(Action):
    """An action the agent can take in the RCTD environment.

    Inherits from openenv.core.Action (provides ``metadata`` field).

    Action Types:
        read_evidence     (cost 1) — Read an evidence item. Reveals text and
                                     *apparent* support (may be noisy).
        run_experiment    (cost 3) — Deep-verify an evidence item. Strips noise
                                     and reveals ground-truth support.
        consult_expert    (cost 2) — Get a probabilistic hint about whether a
                                     hypothesis is likely correct.
        discard_hypothesis(cost 0) — Remove a hypothesis from consideration.
        submit_answer     (cost 0) — Terminal action. Agent's final guess.
    """

    type: str = Field(
        ...,
        description="One of: read_evidence, run_experiment, consult_expert, "
                    "discard_hypothesis, submit_answer",
    )
    evidence_id: Optional[int] = Field(
        default=None,
        description="Target evidence index (for read_evidence, run_experiment)",
    )
    hypothesis_id: Optional[int] = Field(
        default=None,
        description="Target hypothesis index (for consult_expert, "
                    "discard_hypothesis, submit_answer)",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class EvidenceItem(BaseModel):
    """A single piece of evidence that has been revealed to the agent."""

    evidence_id: int
    text: str
    apparent_support: List[int] = Field(
        default_factory=list,
        description="Hypothesis IDs this evidence *appears* to support "
                    "(may be noisy unless verified via run_experiment)",
    )
    apparent_contradiction: List[int] = Field(
        default_factory=list,
        description="Hypothesis IDs this evidence *appears* to contradict "
                    "(may be noisy unless verified via run_experiment)",
    )
    confidence: float = Field(
        default=0.5,
        description="How confident the evidence source appears (0.0–1.0)",
    )
    verified: bool = Field(
        default=False,
        description="True if the agent used run_experiment on this evidence",
    )


class ExpertHint(BaseModel):
    """Hint returned by the consult_expert action."""

    hypothesis_id: int
    hint_text: str
    estimated_probability: float = Field(
        default=0.5,
        description="Expert's estimated probability that this hypothesis is correct",
    )


class RCTDObservation(Observation):
    """What the agent sees after each step.

    Inherits from openenv.core.Observation (provides ``done``, ``reward``,
    ``metadata`` fields).
    """

    # --- Hypotheses ---
    hypotheses: List[str] = Field(
        default_factory=list,
        description="Text descriptions of all hypotheses in the episode",
    )
    active_hypothesis_ids: List[int] = Field(
        default_factory=list,
        description="IDs of hypotheses not yet discarded",
    )

    # --- Evidence gathered so far ---
    revealed_evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence items the agent has read or verified",
    )

    # --- Expert hints gathered ---
    expert_hints: List[ExpertHint] = Field(
        default_factory=list,
        description="Hints from consult_expert actions",
    )

    # --- Budget & progress ---
    budget_remaining: int = 0
    total_evidence_count: int = 0
    step_count: int = 0

    # --- Context for LLMs ---
    action_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Chronological list of past actions and their outcomes",
    )
    message: str = ""

    # --- Terminal metrics (populated only when done=True) ---
    metrics: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# State (episode metadata — includes hidden ground truth)
# ---------------------------------------------------------------------------

class RCTDState(State):
    """Episode metadata. Includes ground truth (not exposed to agent).

    Inherits from openenv.core.State (provides ``episode_id``, ``step_count``).
    """

    # Task configuration
    task_id: str = "medium"
    seed: int = 0
    num_hypotheses: int = 4
    num_evidence: int = 8
    noise_level: float = 0.3
    total_budget: int = 15

    # Ground truth (hidden from agent)
    true_hypothesis_id: int = -1
    scenario_theme: str = ""

    # Tracking
    correct_discards: int = 0
    incorrect_discards: int = 0
    evidence_read: int = 0
    experiments_run: int = 0
    experts_consulted: int = 0
    raw_reward: float = 0.0
