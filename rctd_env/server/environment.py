"""
RCTD Environment — Core game logic and Epistemic Engine.

This is the heart of the environment. It implements:
  1. Deterministic scenario generation from seed
  2. Five distinct action types with budget costs
  3. Noise/reliability mechanics for evidence
  4. Rich reward shaping (terminal + step-wise)
  5. Comprehensive evaluation metrics on termination

Performance target: <1ms per step() call.
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openenv.core import Environment

from ..models import (
    EvidenceItem,
    ExpertHint,
    RCTDAction,
    RCTDObservation,
    RCTDState,
)

# ═══════════════════════════════════════════════════════════════════════════
# Scenario Templates — 5 hard-coded, polished themes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScenarioTemplate:
    """A real-world scenario theme for episode generation."""
    theme: str
    domain: str
    hypothesis_pool: List[str]
    evidence_templates: List[Dict[str, Any]]


SCENARIO_TEMPLATES: List[ScenarioTemplate] = [
    # ── 1. Medical Diagnosis ──────────────────────────────────────────────
    ScenarioTemplate(
        theme="medical_diagnosis",
        domain="Medicine",
        hypothesis_pool=[
            "The patient's condition is caused by a bacterial infection",
            "The patient's condition is caused by a viral infection",
            "The patient's condition is caused by an environmental toxin",
            "The patient's condition is caused by an autoimmune disorder",
            "The patient's condition is caused by a genetic mutation",
        ],
        evidence_templates=[
            {"text": "Broad-spectrum antibiotics show no clinical improvement after 72 hours", "supports": [1, 2, 3, 4], "contradicts": [0], "base_confidence": 0.8},
            {"text": "Viral particles detected in patient blood sample via PCR", "supports": [1], "contradicts": [], "base_confidence": 0.6},
            {"text": "Patient reports no known chemical or pollution exposure", "supports": [0, 1, 3, 4], "contradicts": [2], "base_confidence": 0.7},
            {"text": "White blood cell count is significantly elevated", "supports": [0, 1, 3], "contradicts": [], "base_confidence": 0.5},
            {"text": "Symptoms appeared suddenly after visiting a new geographic region", "supports": [0, 1, 2], "contradicts": [3, 4], "base_confidence": 0.6},
            {"text": "Family history reveals no similar conditions in relatives", "supports": [0, 1, 2], "contradicts": [4], "base_confidence": 0.7},
            {"text": "Anti-nuclear antibody test returns positive", "supports": [3], "contradicts": [0, 1, 2], "base_confidence": 0.75},
            {"text": "Electron microscopy shows characteristic viral inclusion bodies", "supports": [1], "contradicts": [0, 2, 3, 4], "base_confidence": 0.85},
            {"text": "Tissue biopsy reveals inflammatory infiltrate with eosinophils", "supports": [2, 3], "contradicts": [], "base_confidence": 0.65},
            {"text": "Genetic sequencing reveals a known pathogenic variant", "supports": [4], "contradicts": [0, 1, 2], "base_confidence": 0.9},
        ],
    ),
    # ── 2. Market Disruption Analysis ─────────────────────────────────────
    ScenarioTemplate(
        theme="market_analysis",
        domain="Finance",
        hypothesis_pool=[
            "The market disruption is driven by a supply-chain shock",
            "The market disruption is driven by a demand-side shift in consumer behavior",
            "The market disruption is driven by new regulatory policy",
            "The market disruption is driven by a competitor's technological breakthrough",
            "The market disruption is driven by macroeconomic currency fluctuations",
        ],
        evidence_templates=[
            {"text": "Major shipping routes report 40% capacity reduction this quarter", "supports": [0], "contradicts": [1], "base_confidence": 0.8},
            {"text": "Consumer sentiment surveys show dramatically changed preferences", "supports": [1], "contradicts": [], "base_confidence": 0.6},
            {"text": "Government announced new tariffs on key raw materials last month", "supports": [0, 2], "contradicts": [], "base_confidence": 0.75},
            {"text": "Competitor filed 12 patents in the last 60 days in this domain", "supports": [3], "contradicts": [], "base_confidence": 0.7},
            {"text": "Exchange rate volatility has tripled compared to historical average", "supports": [4], "contradicts": [], "base_confidence": 0.65},
            {"text": "Warehouse inventories are at record highs, not lows", "supports": [1, 2, 3, 4], "contradicts": [0], "base_confidence": 0.8},
            {"text": "Social media trend analysis shows viral adoption of alternative products", "supports": [1, 3], "contradicts": [], "base_confidence": 0.5},
            {"text": "Central bank issued emergency interest rate guidance", "supports": [4], "contradicts": [1, 3], "base_confidence": 0.7},
            {"text": "Industry insiders report regulatory review is imminent", "supports": [2], "contradicts": [], "base_confidence": 0.55},
            {"text": "Competitor's stock price surged 30% on undisclosed news", "supports": [3], "contradicts": [0, 2, 4], "base_confidence": 0.6},
        ],
    ),
    # ── 3. Cybersecurity Incident ─────────────────────────────────────────
    ScenarioTemplate(
        theme="security_incident",
        domain="Cybersecurity",
        hypothesis_pool=[
            "The breach was caused by an insider threat (malicious employee)",
            "The breach was caused by an external APT (advanced persistent threat)",
            "The breach was caused by a software vulnerability (zero-day exploit)",
            "The breach was caused by a social engineering / phishing attack",
            "The breach was caused by a misconfigured cloud infrastructure",
        ],
        evidence_templates=[
            {"text": "Access logs show login from an employee's credentials at 3 AM local time", "supports": [0, 3], "contradicts": [], "base_confidence": 0.6},
            {"text": "Network traffic analysis reveals C2 beacon patterns to known APT infrastructure", "supports": [1], "contradicts": [0, 4], "base_confidence": 0.75},
            {"text": "Affected systems are all running an unpatched version of the framework", "supports": [2], "contradicts": [], "base_confidence": 0.7},
            {"text": "Multiple employees received targeted spearphishing emails last week", "supports": [3], "contradicts": [], "base_confidence": 0.65},
            {"text": "Cloud storage bucket was publicly accessible for 6 months per audit", "supports": [4], "contradicts": [1, 2], "base_confidence": 0.85},
            {"text": "Exfiltrated data includes files only accessible to senior engineers", "supports": [0], "contradicts": [4], "base_confidence": 0.7},
            {"text": "Malware sample matches signatures from a known nation-state group", "supports": [1], "contradicts": [0, 3, 4], "base_confidence": 0.6},
            {"text": "The exploit payload targets a CVE published 48 hours ago", "supports": [2], "contradicts": [0, 3], "base_confidence": 0.8},
            {"text": "An employee recently gave notice and was denied a promotion", "supports": [0], "contradicts": [1, 2], "base_confidence": 0.5},
            {"text": "Multi-factor authentication bypass was used in the initial access", "supports": [1, 2, 3], "contradicts": [4], "base_confidence": 0.65},
        ],
    ),
    # ── 4. Climate Event Attribution ──────────────────────────────────────
    ScenarioTemplate(
        theme="climate_attribution",
        domain="Environmental Science",
        hypothesis_pool=[
            "The extreme weather event is primarily driven by natural oceanic cycles (e.g. El Niño)",
            "The extreme weather event is primarily driven by anthropogenic greenhouse gas emissions",
            "The extreme weather event is primarily driven by volcanic aerosol forcing",
            "The extreme weather event is primarily driven by urban heat island effects",
            "The extreme weather event is primarily driven by deforestation-induced albedo changes",
        ],
        evidence_templates=[
            {"text": "Sea surface temperature anomalies match El Niño signature patterns", "supports": [0], "contradicts": [2, 3], "base_confidence": 0.7},
            {"text": "Global CO₂ concentration has reached a new record of 425 ppm", "supports": [1], "contradicts": [], "base_confidence": 0.9},
            {"text": "Satellite data shows a stratospheric aerosol increase from a recent eruption", "supports": [2], "contradicts": [0], "base_confidence": 0.75},
            {"text": "Temperature anomaly is localized to metropolitan areas only", "supports": [3], "contradicts": [0, 1, 2], "base_confidence": 0.65},
            {"text": "Regional deforestation rate has increased 200% in 5 years", "supports": [4], "contradicts": [], "base_confidence": 0.6},
            {"text": "Historical records show similar events occurred during past El Niño cycles", "supports": [0], "contradicts": [1, 3, 4], "base_confidence": 0.7},
            {"text": "Climate model projections under RCP 8.5 predicted this event magnitude", "supports": [1], "contradicts": [0], "base_confidence": 0.8},
            {"text": "Rural stations 50km away show identical temperature spikes", "supports": [0, 1, 2, 4], "contradicts": [3], "base_confidence": 0.75},
            {"text": "Aerosol optical depth measurements show volcanic plume overhead", "supports": [2], "contradicts": [0, 4], "base_confidence": 0.8},
            {"text": "Land-use satellite comparison shows dramatic vegetation cover loss", "supports": [4], "contradicts": [3], "base_confidence": 0.7},
        ],
    ),
    # ── 5. Historical Artifact Authentication ─────────────────────────────
    ScenarioTemplate(
        theme="artifact_authentication",
        domain="Archaeology",
        hypothesis_pool=[
            "The artifact is an authentic item from the claimed historical period",
            "The artifact is a modern forgery created with period-appropriate materials",
            "The artifact is genuine but misdated (from a different era than claimed)",
            "The artifact is a contemporary replica made for educational purposes",
            "The artifact is a composite of authentic fragments and modern restoration",
        ],
        evidence_templates=[
            {"text": "Radiocarbon dating places the material within the claimed period ±50 years", "supports": [0], "contradicts": [1, 3], "base_confidence": 0.8},
            {"text": "Microscopic analysis reveals tool marks consistent with modern machinery", "supports": [1, 3], "contradicts": [0], "base_confidence": 0.7},
            {"text": "The artistic style is inconsistent with the claimed period's conventions", "supports": [1, 2, 3], "contradicts": [0], "base_confidence": 0.6},
            {"text": "Chemical composition of pigments matches known historical sources", "supports": [0, 2, 4], "contradicts": [1, 3], "base_confidence": 0.75},
            {"text": "Provenance records show a gap of 200 years with no documentation", "supports": [1, 2, 3], "contradicts": [], "base_confidence": 0.55},
            {"text": "Thermoluminescence dating gives a date 300 years after the claimed period", "supports": [2], "contradicts": [0, 1, 3], "base_confidence": 0.85},
            {"text": "X-ray fluorescence reveals two distinct layers of material composition", "supports": [4], "contradicts": [0, 1, 3], "base_confidence": 0.8},
            {"text": "Similar artifacts from verified excavations share identical iconography", "supports": [0, 2], "contradicts": [1, 3], "base_confidence": 0.7},
            {"text": "A museum catalog from 1920 lists an item with matching description", "supports": [0, 2], "contradicts": [3], "base_confidence": 0.65},
            {"text": "UV fluorescence shows modern adhesive at several joint points", "supports": [4], "contradicts": [0], "base_confidence": 0.75},
        ],
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Internal data structures (not exposed to agents)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HiddenEvidence:
    """Server-side ground truth for one evidence item."""
    evidence_id: int
    text: str
    true_support: List[int]       # Hypothesis IDs it genuinely supports
    true_contradiction: List[int] # Hypothesis IDs it genuinely contradicts
    reliability: float            # 0.5–1.0; low → support may flip
    base_confidence: float        # How confident the source appears


@dataclass
class EpisodeData:
    """All hidden state for one episode."""
    seed: int
    theme: str
    domain: str
    true_hypothesis_id: int
    hypotheses: List[str]
    evidence: List[HiddenEvidence]
    active_hypothesis_ids: List[int] = field(default_factory=list)
    revealed_evidence: Dict[int, EvidenceItem] = field(default_factory=dict)
    expert_hints: List[ExpertHint] = field(default_factory=list)
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    budget_remaining: int = 15
    total_budget: int = 15
    step_count: int = 0
    discarded_true: bool = False
    correct_discards: int = 0
    incorrect_discards: int = 0
    evidence_read: int = 0
    experiments_run: int = 0
    experts_consulted: int = 0
    raw_reward: float = 0.0
    submitted: bool = False
    submitted_hypothesis: Optional[int] = None


# ═══════════════════════════════════════════════════════════════════════════
# Action costs
# ═══════════════════════════════════════════════════════════════════════════

ACTION_COSTS: Dict[str, int] = {
    "read_evidence": 1,
    "run_experiment": 3,
    "consult_expert": 2,
    "discard_hypothesis": 0,
    "submit_answer": 0,
}

VALID_ACTION_TYPES = set(ACTION_COSTS.keys())


# ═══════════════════════════════════════════════════════════════════════════
# The Environment
# ═══════════════════════════════════════════════════════════════════════════

class RCTDEnvironment(Environment[RCTDAction, RCTDObservation, RCTDState]):
    """Research Coordination & Truth Discovery Environment.

    Inherits from openenv.core.Environment for full framework compliance.
    Evaluates agent reasoning under conflicting, noisy evidence.

    Key mechanics:
      - Evidence may be noisy (support flips based on reliability).
      - ``run_experiment`` bypasses noise to reveal ground truth.
      - ``consult_expert`` provides probabilistic hints.
      - Budget constrains total actions; agents must be efficient.
      - Rich terminal metrics enable fine-grained evaluation.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._episode: Optional[EpisodeData] = None
        self._state = RCTDState()
        self._episode_id: Optional[str] = None

    def get_metadata(self):
        """Return environment metadata for the OpenEnv framework."""
        from openenv.core.env_server.interfaces import EnvironmentMetadata
        return EnvironmentMetadata(
            name="RCTD Environment",
            description="Research Coordination & Truth Discovery — "
                        "Epistemic reasoning under uncertainty",
            version="1.0.0",
        )

    # ───────────────────────────────────────────────────────────────────
    # reset()
    # ───────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "medium",
        **kwargs: Any,
    ) -> RCTDObservation:
        """Start a new episode. Deterministic given seed + task_id."""
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        self._episode_id = episode_id or str(uuid.uuid4())

        # Get task configuration
        config = TASK_CONFIGS.get(task_id, TASK_CONFIGS["medium"])

        # Generate the episode
        self._episode = _generate_episode(
            seed=seed,
            num_hypotheses=config["num_hypotheses"],
            num_evidence=config["num_evidence"],
            noise_level=config["noise_level"],
            budget=config["budget"],
        )

        # Update state
        self._state = RCTDState(
            episode_id=self._episode_id,
            step_count=0,
            task_id=task_id,
            seed=seed,
            num_hypotheses=config["num_hypotheses"],
            num_evidence=config["num_evidence"],
            noise_level=config["noise_level"],
            total_budget=config["budget"],
            true_hypothesis_id=self._episode.true_hypothesis_id,
            scenario_theme=self._episode.theme,
        )

        return RCTDObservation(
            done=False,
            reward=None,
            hypotheses=self._episode.hypotheses,
            active_hypothesis_ids=list(self._episode.active_hypothesis_ids),
            revealed_evidence=[],
            expert_hints=[],
            budget_remaining=self._episode.budget_remaining,
            total_evidence_count=len(self._episode.evidence),
            step_count=0,
            action_history=[],
            message=(
                f"Welcome to the {self._episode.domain} investigation. "
                f"There are {len(self._episode.hypotheses)} competing hypotheses "
                f"and {len(self._episode.evidence)} evidence items to investigate. "
                f"You have a budget of {self._episode.budget_remaining} action points. "
                f"Use your budget wisely to determine the truth."
            ),
            metrics=None,
        )

    # ───────────────────────────────────────────────────────────────────
    # step()
    # ───────────────────────────────────────────────────────────────────

    def step(
        self,
        action: RCTDAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> RCTDObservation:
        """Process one agent action. Returns updated observation."""
        ep = self._episode
        if ep is None:
            raise RuntimeError("Must call reset() before step()")

        if ep.submitted:
            raise RuntimeError("Episode already terminated. Call reset().")

        # Validate action type
        if action.type not in VALID_ACTION_TYPES:
            return self._make_obs(
                reward=-0.01,
                message=f"Invalid action type '{action.type}'. "
                        f"Valid types: {sorted(VALID_ACTION_TYPES)}",
            )

        # Check budget
        cost = ACTION_COSTS[action.type]
        if cost > ep.budget_remaining and action.type != "submit_answer":
            # Force submission when budget is exhausted
            return self._force_budget_exhausted()

        ep.step_count += 1
        self._state.step_count = ep.step_count

        # Dispatch to action handler
        handler = {
            "read_evidence": self._handle_read_evidence,
            "run_experiment": self._handle_run_experiment,
            "consult_expert": self._handle_consult_expert,
            "discard_hypothesis": self._handle_discard_hypothesis,
            "submit_answer": self._handle_submit_answer,
        }[action.type]

        return handler(action)

    # ───────────────────────────────────────────────────────────────────
    # state property
    # ───────────────────────────────────────────────────────────────────

    @property
    def state(self) -> RCTDState:
        """Return current episode metadata (includes hidden ground truth)."""
        if self._episode:
            self._state.correct_discards = self._episode.correct_discards
            self._state.incorrect_discards = self._episode.incorrect_discards
            self._state.evidence_read = self._episode.evidence_read
            self._state.experiments_run = self._episode.experiments_run
            self._state.experts_consulted = self._episode.experts_consulted
            self._state.raw_reward = self._episode.raw_reward
        return self._state

    # ───────────────────────────────────────────────────────────────────
    # Action Handlers
    # ───────────────────────────────────────────────────────────────────

    def _handle_read_evidence(self, action: RCTDAction) -> RCTDObservation:
        ep = self._episode
        eid = action.evidence_id

        if eid is None or eid < 0 or eid >= len(ep.evidence):
            return self._make_obs(
                reward=-0.01,
                message=f"Invalid evidence_id={eid}. "
                        f"Valid range: 0–{len(ep.evidence) - 1}",
            )

        if eid in ep.revealed_evidence:
            return self._make_obs(
                reward=0.0,
                message=f"Evidence E{eid} has already been read.",
            )

        # Deduct cost
        ep.budget_remaining -= ACTION_COSTS["read_evidence"]
        ep.evidence_read += 1

        # Get the hidden evidence
        hidden = ep.evidence[eid]

        # Apply noise: low reliability → support may flip
        rng = random.Random(ep.seed * 1000 + eid)
        apparent_support = list(hidden.true_support)
        apparent_contradiction = list(hidden.true_contradiction)

        if rng.random() > hidden.reliability:
            # Noise: flip support — remove some true supports, add false ones
            if apparent_support and rng.random() < 0.5:
                apparent_support.pop(rng.randrange(len(apparent_support)))
            # Add a false support
            false_candidates = [
                h for h in ep.active_hypothesis_ids
                if h not in hidden.true_support and h != ep.true_hypothesis_id
            ]
            if false_candidates:
                apparent_support.append(rng.choice(false_candidates))
            # Noise on contradictions too: may drop a real contradiction
            if apparent_contradiction and rng.random() < 0.4:
                apparent_contradiction.pop(rng.randrange(len(apparent_contradiction)))

        item = EvidenceItem(
            evidence_id=eid,
            text=hidden.text,
            apparent_support=apparent_support,
            apparent_contradiction=apparent_contradiction,
            confidence=hidden.base_confidence,
            verified=False,
        )
        ep.revealed_evidence[eid] = item

        # Step reward proportional to action cost
        step_reward = -float(ACTION_COSTS['read_evidence'])
        ep.raw_reward += step_reward

        ep.action_history.append({
            "step": ep.step_count,
            "action": "read_evidence",
            "evidence_id": eid,
            "cost": ACTION_COSTS["read_evidence"],
        })

        # Check if budget exhausted
        if ep.budget_remaining <= 0:
            return self._force_budget_exhausted()

        return self._make_obs(
            reward=self._normalize_step_reward(step_reward),
            message=f"Read evidence E{eid}: \"{hidden.text}\" "
                    f"— Appears to support hypothesis/hypotheses: "
                    f"{[f'H{s}' for s in apparent_support]} "
                    f"(confidence: {hidden.base_confidence:.0%})",
        )

    def _handle_run_experiment(self, action: RCTDAction) -> RCTDObservation:
        ep = self._episode
        eid = action.evidence_id

        if eid is None or eid < 0 or eid >= len(ep.evidence):
            return self._make_obs(
                reward=-0.01,
                message=f"Invalid evidence_id={eid}. "
                        f"Valid range: 0–{len(ep.evidence) - 1}",
            )

        if eid not in ep.revealed_evidence:
            return self._make_obs(
                reward=-0.01,
                message=f"Must read evidence E{eid} first before running experiment.",
            )

        if ep.revealed_evidence[eid].verified:
            return self._make_obs(
                reward=0.0,
                message=f"Evidence E{eid} has already been verified.",
            )

        # Deduct cost
        ep.budget_remaining -= ACTION_COSTS["run_experiment"]
        ep.experiments_run += 1

        # Reveal ground truth (strips noise)
        hidden = ep.evidence[eid]
        verified_item = EvidenceItem(
            evidence_id=eid,
            text=hidden.text,
            apparent_support=list(hidden.true_support),
            apparent_contradiction=list(hidden.true_contradiction),
            confidence=1.0,  # Verified = full confidence
            verified=True,
        )
        ep.revealed_evidence[eid] = verified_item

        # Step reward proportional to action cost
        step_reward = -float(ACTION_COSTS['run_experiment'])
        ep.raw_reward += step_reward

        ep.action_history.append({
            "step": ep.step_count,
            "action": "run_experiment",
            "evidence_id": eid,
            "cost": ACTION_COSTS["run_experiment"],
        })

        if ep.budget_remaining <= 0:
            return self._force_budget_exhausted()

        return self._make_obs(
            reward=self._normalize_step_reward(step_reward),
            message=f"Experiment on E{eid} complete. VERIFIED support: "
                    f"{[f'H{s}' for s in hidden.true_support]} "
                    f"(confidence: 100%)",
        )

    def _handle_consult_expert(self, action: RCTDAction) -> RCTDObservation:
        ep = self._episode
        hid = action.hypothesis_id

        if hid is None or hid not in ep.active_hypothesis_ids:
            return self._make_obs(
                reward=-0.01,
                message=f"Invalid or already-discarded hypothesis_id={hid}.",
            )

        # Deduct cost
        ep.budget_remaining -= ACTION_COSTS["consult_expert"]
        ep.experts_consulted += 1

        # Generate probabilistic hint (deterministic from seed)
        rng = random.Random(ep.seed * 2000 + hid)

        if hid == ep.true_hypothesis_id:
            # True hypothesis: expert gives high probability (0.6–0.9)
            prob = rng.uniform(0.6, 0.9)
            hint_text = _generate_expert_hint_text(ep.theme, hid, prob, is_true=True, rng=rng)
        else:
            # False hypothesis: expert gives low probability (0.1–0.45)
            prob = rng.uniform(0.1, 0.45)
            hint_text = _generate_expert_hint_text(ep.theme, hid, prob, is_true=False, rng=rng)

        hint = ExpertHint(
            hypothesis_id=hid,
            hint_text=hint_text,
            estimated_probability=round(prob, 2),
        )
        ep.expert_hints.append(hint)

        step_reward = -float(ACTION_COSTS['consult_expert'])
        ep.raw_reward += step_reward

        ep.action_history.append({
            "step": ep.step_count,
            "action": "consult_expert",
            "hypothesis_id": hid,
            "cost": ACTION_COSTS["consult_expert"],
        })

        if ep.budget_remaining <= 0:
            return self._force_budget_exhausted()

        return self._make_obs(
            reward=self._normalize_step_reward(step_reward),
            message=f"Expert consulted on H{hid}: \"{hint_text}\" "
                    f"(estimated probability: {prob:.0%})",
        )

    def _handle_discard_hypothesis(self, action: RCTDAction) -> RCTDObservation:
        ep = self._episode
        hid = action.hypothesis_id

        if hid is None or hid not in ep.active_hypothesis_ids:
            return self._make_obs(
                reward=-0.01,
                message=f"Invalid or already-discarded hypothesis_id={hid}.",
            )

        # Cannot discard the last remaining hypothesis
        if len(ep.active_hypothesis_ids) <= 1:
            return self._make_obs(
                reward=-0.01,
                message="Cannot discard the last remaining hypothesis. "
                        "Use submit_answer instead.",
            )

        ep.active_hypothesis_ids.remove(hid)

        if hid == ep.true_hypothesis_id:
            # Agent discarded the truth — severe penalty
            ep.discarded_true = True
            ep.incorrect_discards += 1
            step_reward = -5.0
            msg = f"Discarded hypothesis H{hid}: \"{ep.hypotheses[hid]}\". Noted."
        else:
            # Correct discard
            ep.correct_discards += 1
            step_reward = 2.0
            msg = f"Discarded hypothesis H{hid}: \"{ep.hypotheses[hid]}\". " \
                  f"Search space narrowed to {len(ep.active_hypothesis_ids)} hypotheses."

        ep.raw_reward += step_reward

        ep.action_history.append({
            "step": ep.step_count,
            "action": "discard_hypothesis",
            "hypothesis_id": hid,
            "cost": 0,
        })

        return self._make_obs(
            reward=self._normalize_step_reward(step_reward),
            message=msg,
        )

    def _handle_submit_answer(self, action: RCTDAction) -> RCTDObservation:
        ep = self._episode
        hid = action.hypothesis_id

        if hid is None or hid < 0 or hid >= len(ep.hypotheses):
            return self._make_obs(
                reward=-0.01,
                message=f"Invalid hypothesis_id={hid} for submission. "
                        f"Valid: {list(range(len(ep.hypotheses)))}",
            )

        ep.submitted = True
        ep.submitted_hypothesis = hid

        # Terminal reward
        correct = (hid == ep.true_hypothesis_id)

        if correct and ep.budget_remaining > 0:
            terminal_reward = 100.0
        elif correct and ep.budget_remaining == 0:
            terminal_reward = 50.0
        elif not correct and ep.discarded_true:
            terminal_reward = -100.0
        else:
            terminal_reward = -50.0

        ep.raw_reward += terminal_reward

        ep.action_history.append({
            "step": ep.step_count,
            "action": "submit_answer",
            "hypothesis_id": hid,
            "cost": 0,
            "correct": correct,
        })

        # Compute terminal metrics
        metrics = _compute_metrics(ep)

        if correct:
            msg = (f"✓ CORRECT! The answer is H{hid}: \"{ep.hypotheses[hid]}\". "
                   f"Efficiency: {metrics['efficiency_score']:.0%}")
        else:
            msg = (f"✗ INCORRECT. You answered H{hid}: \"{ep.hypotheses[hid]}\". "
                   f"The true answer was H{ep.true_hypothesis_id}: "
                   f"\"{ep.hypotheses[ep.true_hypothesis_id]}\".")

        # Normalize total raw reward to 0.0–1.0 for the observation
        normalized = self._normalize_terminal_reward(ep.raw_reward)

        return RCTDObservation(
            done=True,
            reward=normalized,
            hypotheses=ep.hypotheses,
            active_hypothesis_ids=list(ep.active_hypothesis_ids),
            revealed_evidence=list(ep.revealed_evidence.values()),
            expert_hints=list(ep.expert_hints),
            budget_remaining=ep.budget_remaining,
            total_evidence_count=len(ep.evidence),
            step_count=ep.step_count,
            action_history=ep.action_history,
            message=msg,
            metrics=metrics,
        )

    # ───────────────────────────────────────────────────────────────────
    # Helpers
    # ───────────────────────────────────────────────────────────────────

    def _force_budget_exhausted(self) -> RCTDObservation:
        """Budget hit zero — force termination.

        Unlike voluntary submission, budget exhaustion picks the
        most-supported hypothesis from gathered evidence. This is
        intentionally weaker than the agent choosing deliberately
        (the agent is penalized for running out of budget through
        reduced efficiency score and the budget_exhausted failure mode).
        """
        ep = self._episode
        ep.submitted = True

        # Pick the most-supported hypothesis from gathered evidence
        if ep.active_hypothesis_ids:
            auto_answer = _find_best_hypothesis_from_evidence(
                ep.active_hypothesis_ids,
                ep.revealed_evidence,
                ep.expert_hints,
            )
        else:
            auto_answer = 0

        ep.submitted_hypothesis = auto_answer
        correct = (auto_answer == ep.true_hypothesis_id)

        terminal_reward = -50.0 if not correct else 50.0
        ep.raw_reward += terminal_reward

        ep.action_history.append({
            "step": ep.step_count,
            "action": "budget_exhausted",
            "auto_answer": auto_answer,
            "correct": correct,
        })

        metrics = _compute_metrics(ep)
        metrics["failure_mode"] = "budget_exhausted"

        normalized = self._normalize_terminal_reward(ep.raw_reward)

        msg = (
            f"⚠ Budget exhausted! Auto-submitted H{auto_answer}. "
            + ("Correct!" if correct else
               f"Wrong — true answer was H{ep.true_hypothesis_id}.")
        )

        return RCTDObservation(
            done=True,
            reward=normalized,
            hypotheses=ep.hypotheses,
            active_hypothesis_ids=list(ep.active_hypothesis_ids),
            revealed_evidence=list(ep.revealed_evidence.values()),
            expert_hints=list(ep.expert_hints),
            budget_remaining=0,
            total_evidence_count=len(ep.evidence),
            step_count=ep.step_count,
            action_history=ep.action_history,
            message=msg,
            metrics=metrics,
        )

    def _make_obs(self, reward: float, message: str) -> RCTDObservation:
        """Build an observation from current episode state."""
        ep = self._episode
        return RCTDObservation(
            done=False,
            reward=self._normalize_step_reward(reward),
            hypotheses=ep.hypotheses,
            active_hypothesis_ids=list(ep.active_hypothesis_ids),
            revealed_evidence=list(ep.revealed_evidence.values()),
            expert_hints=list(ep.expert_hints),
            budget_remaining=ep.budget_remaining,
            total_evidence_count=len(ep.evidence),
            step_count=ep.step_count,
            action_history=ep.action_history,
            message=message,
            metrics=None,
        )

    @staticmethod
    def _normalize_step_reward(raw: float) -> float:
        """Map step rewards to a reasonable 0–1 scale. Uses sigmoid-like mapping."""
        # Step rewards are small (-5 to +2), map to ~0.0–1.0
        return round(1.0 / (1.0 + math.exp(-raw)), 4)

    @staticmethod
    def _normalize_terminal_reward(raw: float) -> float:
        """Map cumulative raw reward to 0.0–1.0.

        Raw range: approx -120 (worst) to +120 (best).
        """
        # Sigmoid normalization centered at 0
        return round(1.0 / (1.0 + math.exp(-raw / 30.0)), 4)


# ═══════════════════════════════════════════════════════════════════════════
# Task Configurations
# ═══════════════════════════════════════════════════════════════════════════

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "num_hypotheses": 3,
        "num_evidence": 6,
        "noise_level": 0.1,
        "budget": 20,
        "description": "Introductory difficulty — low noise, generous budget",
    },
    "medium": {
        "num_hypotheses": 4,
        "num_evidence": 8,
        "noise_level": 0.3,
        "budget": 15,
        "description": "Standard challenge — moderate noise, requires evidence evaluation",
    },
    "hard": {
        "num_hypotheses": 5,
        "num_evidence": 10,
        "noise_level": 0.45,
        "budget": 12,
        "description": "Expert difficulty — high noise, tight budget, deep reasoning required",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Expert Hint Generation (Domain-Specific, Graduated)
# ═══════════════════════════════════════════════════════════════════════════

# Per-domain expert hint templates. Each list has 3 entries:
#   [0] = high confidence (prob > 0.7)
#   [1] = moderate confidence (prob 0.4–0.7)
#   [2] = low confidence (prob < 0.4)
_EXPERT_HINT_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "medical_diagnosis": {
        "positive": [
            "The clinical presentation, combined with lab markers, strongly aligns with H{hid}. I've seen this pattern repeatedly in my practice.",
            "Several diagnostic indicators are consistent with H{hid}, though I'd recommend confirmatory testing before concluding.",
            "There are some features suggestive of H{hid}, but the differential is still broad at this stage.",
        ],
        "negative": [
            "The symptom profile has critical inconsistencies with H{hid} — multiple pathognomonic signs are absent.",
            "While H{hid} was initially plausible, the accumulating evidence is making me increasingly skeptical.",
            "H{hid} cannot be ruled out entirely, but other etiologies seem more consistent with the observed data.",
        ],
    },
    "market_analysis": {
        "positive": [
            "The macroeconomic signals and sector-specific data strongly corroborate H{hid}. My quantitative models converge on this explanation.",
            "H{hid} is consistent with several leading indicators I track, though the causal link isn't conclusive yet.",
            "H{hid} has some supporting data points, but the market dynamics are complex enough that I'd want more confirmation.",
        ],
        "negative": [
            "Key market fundamentals directly contradict H{hid} — the correlations break down under scrutiny.",
            "My models show diminishing support for H{hid} as more data comes in. The trend is moving away from this thesis.",
            "H{hid} is one possible factor, but it's unlikely to be the primary driver based on what I'm seeing.",
        ],
    },
    "security_incident": {
        "positive": [
            "The attack signature, TTPs, and lateral movement pattern are highly characteristic of H{hid}. I've investigated similar incidents before.",
            "Several forensic indicators point toward H{hid}, though we need to complete the full kill-chain analysis.",
            "H{hid} is among the plausible scenarios, but the evidence is still circumstantial at this point.",
        ],
        "negative": [
            "The IOCs and forensic timeline have significant gaps that are inconsistent with H{hid}.",
            "Some aspects of H{hid} don't match the attack surface analysis. I'm leaning away from this theory.",
            "H{hid} is possible but less likely given the IR data collected so far.",
        ],
    },
    "climate_attribution": {
        "positive": [
            "The spatial and temporal patterns in the observational data are strongly consistent with H{hid} — multiple independent datasets converge.",
            "Modeling results and proxy data offer moderate support for H{hid}, though natural variability adds uncertainty.",
            "H{hid} is within the range of plausible forcings, but the attribution signal is not yet robust.",
        ],
        "negative": [
            "The observational record shows patterns that are fundamentally inconsistent with H{hid} as the primary driver.",
            "Recent paleoclimate comparisons weaken the case for H{hid}. The forcing magnitude doesn't match.",
            "H{hid} likely contributes but is unlikely to be the dominant factor based on current analysis.",
        ],
    },
    "artifact_authentication": {
        "positive": [
            "The material composition, patina development, and stylistic elements are all strongly consistent with H{hid}. This is a compelling case.",
            "Several analytical results support H{hid}, but I'd want to see additional provenance documentation.",
            "H{hid} remains plausible based on initial examination, though the evidence is not yet definitive.",
        ],
        "negative": [
            "Critical anachronisms and material inconsistencies make H{hid} very unlikely in my professional assessment.",
            "The isotopic analysis and tool-mark patterns are creating doubt about H{hid}.",
            "H{hid} cannot be excluded, but it doesn't fit the overall pattern of evidence as cleanly as alternatives.",
        ],
    },
}


def _generate_expert_hint_text(
    theme: str,
    hid: int,
    prob: float,
    is_true: bool,
    rng: random.Random,
) -> str:
    """Generate a domain-specific expert hint with graduated confidence."""
    templates = _EXPERT_HINT_TEMPLATES.get(theme)

    if templates is None:
        # Fallback for unknown themes
        if is_true:
            return f"Based on my expertise, hypothesis H{hid} shows indicative evidence in its favor."
        return f"Based on my expertise, hypothesis H{hid} has aspects that raise questions."

    key = "positive" if is_true else "negative"
    variants = templates[key]

    # Select based on probability level
    if prob > 0.7:
        hint = variants[0]
    elif prob > 0.4:
        hint = variants[1]
    else:
        hint = variants[2]

    return hint.format(hid=hid)


# ═══════════════════════════════════════════════════════════════════════════
# Evidence-Based Hypothesis Selection
# ═══════════════════════════════════════════════════════════════════════════


def _find_best_hypothesis_from_evidence(
    active_hypothesis_ids: List[int],
    revealed_evidence: Dict[int, EvidenceItem],
    expert_hints: List[ExpertHint],
) -> int:
    """Pick the most-supported active hypothesis from gathered evidence.

    Used by _force_budget_exhausted() to make a best-effort answer
    when the agent runs out of budget. Weights verified evidence
    higher than unverified, and factors in expert opinions.
    """
    if not active_hypothesis_ids:
        return 0

    support_counts: Dict[int, float] = {h: 0.0 for h in active_hypothesis_ids}

    # Count evidence support (verified evidence gets double weight)
    for ev in revealed_evidence.values():
        weight = 2.0 if ev.verified else 1.0
        for h in ev.apparent_support:
            if h in support_counts:
                support_counts[h] += weight * ev.confidence

    # Factor in expert hints
    for hint in expert_hints:
        if hint.hypothesis_id in support_counts:
            support_counts[hint.hypothesis_id] += hint.estimated_probability

    return max(support_counts, key=support_counts.get)


# ═══════════════════════════════════════════════════════════════════════════
# Episode Generation (Deterministic Epistemic Engine)
# ═══════════════════════════════════════════════════════════════════════════

def _generate_episode(
    seed: int,
    num_hypotheses: int,
    num_evidence: int,
    noise_level: float,
    budget: int,
) -> EpisodeData:
    """Generate a complete episode from seed. STRICTLY DETERMINISTIC.

    Evidence rules:
      - 60–70% of evidence implicitly supports the true hypothesis
      - 1–2 high-confidence misleading contradictions
      - At least 2 strong, indisputable support items for truth
      - False hypotheses get 20–40% accidental support
      - Reliability varies: low reliability → noise flips support
    """
    rng = random.Random(seed)

    # Select scenario
    template = rng.choice(SCENARIO_TEMPLATES)

    # Select hypotheses (subset of pool)
    assert num_hypotheses <= len(template.hypothesis_pool), \
        f"Requested {num_hypotheses} hypotheses but template only has {len(template.hypothesis_pool)}"
    hypothesis_indices = rng.sample(range(len(template.hypothesis_pool)), num_hypotheses)
    hypotheses = [template.hypothesis_pool[i] for i in hypothesis_indices]

    # Remap indices: original template IDs → new 0..N-1 IDs
    index_remap = {old: new for new, old in enumerate(hypothesis_indices)}

    # Select ground truth
    true_id = rng.randrange(num_hypotheses)

    # Select and adapt evidence
    available_evidence = list(range(len(template.evidence_templates)))
    rng.shuffle(available_evidence)
    selected_evidence_indices = available_evidence[:num_evidence]

    evidence_items: List[HiddenEvidence] = []
    strong_support_count = 0

    for new_eid, orig_eid in enumerate(selected_evidence_indices):
        tmpl = template.evidence_templates[orig_eid]

        # Remap support/contradiction to our hypothesis indices
        true_support = [
            index_remap[s] for s in tmpl["supports"]
            if s in index_remap
        ]
        true_contradiction = [
            index_remap[c] for c in tmpl["contradicts"]
            if c in index_remap
        ]

        # Ensure evidence distribution rules
        # If this evidence doesn't yet support truth and we need more support:
        if true_id not in true_support and rng.random() < 0.65:
            true_support.append(true_id)

        # Determine reliability based on noise level
        reliability = rng.uniform(max(0.5, 1.0 - noise_level * 1.5), 1.0)

        # Create at least 2 strong support items for the true hypothesis
        base_confidence = tmpl["base_confidence"]
        if true_id in true_support and strong_support_count < 2:
            reliability = max(reliability, 0.85)
            base_confidence = max(base_confidence, 0.8)
            strong_support_count += 1

        # Create 1-2 misleading high-confidence contradictions
        if (new_eid < 2 and true_id in true_support
                and rng.random() < noise_level):
            # Make this a misleading item
            reliability = rng.uniform(0.4, 0.6)
            base_confidence = rng.uniform(0.7, 0.9)  # Looks confident but unreliable

        evidence_items.append(HiddenEvidence(
            evidence_id=new_eid,
            text=tmpl["text"],
            true_support=true_support,
            true_contradiction=true_contradiction,
            reliability=round(reliability, 3),
            base_confidence=round(base_confidence, 2),
        ))

    # Ensure false hypotheses have 20-40% accidental support
    for h_id in range(num_hypotheses):
        if h_id == true_id:
            continue
        supporting = sum(1 for e in evidence_items if h_id in e.true_support)
        target_support = rng.randint(
            max(1, int(num_evidence * 0.2)),
            max(2, int(num_evidence * 0.4)),
        )
        while supporting < target_support:
            candidate = rng.choice(evidence_items)
            if h_id not in candidate.true_support:
                candidate.true_support.append(h_id)
                supporting += 1

    return EpisodeData(
        seed=seed,
        theme=template.theme,
        domain=template.domain,
        true_hypothesis_id=true_id,
        hypotheses=hypotheses,
        evidence=evidence_items,
        active_hypothesis_ids=list(range(num_hypotheses)),
        budget_remaining=budget,
        total_budget=budget,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Terminal Metrics
# ═══════════════════════════════════════════════════════════════════════════

def _compute_metrics(ep: EpisodeData) -> Dict[str, Any]:
    """Compute rich evaluation metrics for the info dict."""
    total_evidence = len(ep.evidence)
    revealed = len(ep.revealed_evidence)

    # Success
    success = (ep.submitted_hypothesis == ep.true_hypothesis_id)

    # Efficiency: budget remaining / total budget
    efficiency = ep.budget_remaining / ep.total_budget if ep.total_budget > 0 else 0

    # Evidence utilization
    utilization = revealed / total_evidence if total_evidence > 0 else 0

    # Failure mode classification
    failure_mode = None
    if not success:
        if ep.discarded_true:
            failure_mode = "discarded_correct_hypothesis"
        elif ep.budget_remaining <= 0:
            failure_mode = "budget_exhausted"
        elif revealed < total_evidence * 0.3:
            failure_mode = "insufficient_evidence"
        else:
            # Check if agent was misled by noisy evidence
            noisy_reads = sum(
                1 for eid, item in ep.revealed_evidence.items()
                if not item.verified and ep.evidence[eid].reliability < 0.7
            )
            if noisy_reads > 0:
                failure_mode = "misled_by_noise"
            else:
                failure_mode = "reasoning_error"

    return {
        "success": success,
        "efficiency_score": round(efficiency, 3),
        "evidence_utilization": round(utilization, 3),
        "steps_taken": ep.step_count,
        "evidence_read": ep.evidence_read,
        "experiments_run": ep.experiments_run,
        "experts_consulted": ep.experts_consulted,
        "correct_discards": ep.correct_discards,
        "incorrect_discards": ep.incorrect_discards,
        "budget_used": ep.total_budget - ep.budget_remaining,
        "failure_mode": failure_mode,
        "raw_reward": round(ep.raw_reward, 2),
        "scenario_theme": ep.theme,
        "true_hypothesis": ep.true_hypothesis_id,
        "submitted_hypothesis": ep.submitted_hypothesis,
    }
