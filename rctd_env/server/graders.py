"""
RCTD Environment — Programmatic graders.

Each grader runs one or more episodes for a specific task and produces
a score between 0.0 and 1.0 based on four components:

  Accuracy  (0.0 or 0.60) — Did the agent find the correct hypothesis?
  Efficiency(0.00–0.20)   — Budget remaining / total budget
  Utilization(0.00–0.10)  — Proportion of evidence meaningfully gathered
  Process   (0.00–0.10)   — Quality of elimination strategy

The grader accepts a policy callable: policy(observation) -> action.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ..models import RCTDAction, RCTDObservation
from .environment import RCTDEnvironment


# ═══════════════════════════════════════════════════════════════════════════
# Grader
# ═══════════════════════════════════════════════════════════════════════════

def grade_episode(
    env: RCTDEnvironment,
    policy: Callable[[RCTDObservation], RCTDAction],
    task_id: str = "medium",
    seed: int = 0,
    max_steps: int = 50,
) -> Dict[str, Any]:
    """Run a single episode and return a graded result.

    Args:
        env: The RCTD environment instance.
        policy: A callable that takes an observation and returns an action.
        task_id: One of "easy", "medium", "hard".
        seed: Random seed for reproducibility.
        max_steps: Safety limit to prevent infinite loops.

    Returns:
        Dict with 'score' (0.0–1.0), component scores, and episode metrics.
    """
    obs = env.reset(seed=seed, task_id=task_id)
    steps = 0

    while not obs.done and steps < max_steps:
        action = policy(obs)
        obs = env.step(action)
        steps += 1

    # If policy didn't submit, force a submission
    if not obs.done:
        obs = env.step(RCTDAction(
            type="submit_answer",
            hypothesis_id=obs.active_hypothesis_ids[0] if obs.active_hypothesis_ids else 0,
        ))

    metrics = obs.metrics or {}

    # ── Component scores ──────────────────────────────────────────────
    # Accuracy (60% weight)
    accuracy = 0.60 if metrics.get("success", False) else 0.0

    # Efficiency (20% weight): budget remaining / total budget
    efficiency_raw = metrics.get("efficiency_score", 0.0)
    efficiency = round(efficiency_raw * 0.20, 4)

    # Evidence utilization (10% weight)
    utilization_raw = metrics.get("evidence_utilization", 0.0)
    # Reward moderate utilization, penalize extremes (0% or 100%)
    # Optimal is ~40-70% utilization
    if utilization_raw < 0.3:
        util_score = utilization_raw / 0.3  # Linear ramp up
    elif utilization_raw > 0.8:
        util_score = max(0, 1.0 - (utilization_raw - 0.8) / 0.2)  # Penalize over-reading
    else:
        util_score = 1.0
    utilization = round(util_score * 0.10, 4)

    # Process quality (10% weight)
    correct_discards = metrics.get("correct_discards", 0)
    incorrect_discards = metrics.get("incorrect_discards", 0)
    total_hypotheses = env.state.num_hypotheses

    process_raw = 0.0
    if total_hypotheses > 1:
        # Reward correct eliminations
        max_discards = total_hypotheses - 1
        process_raw += (correct_discards / max_discards) * 0.7
        # Penalize incorrect discards
        process_raw -= incorrect_discards * 0.5
    process_raw = max(0.0, min(1.0, process_raw))
    process = round(process_raw * 0.10, 4)

    # Accuracy gate: when the agent gets the answer wrong, auxiliary
    # components (efficiency, utilization, process) contribute at 25% rate.
    # This prevents reward confounding where a random agent can score higher
    # on hard tasks (tight budget → early submit → high efficiency) than on
    # easy tasks. Wrong answers are capped at ~0.10 max from auxiliaries.
    accuracy_gate = 1.0 if accuracy > 0 else 0.25

    # Apply gate to auxiliary components
    efficiency_gated = round(efficiency * accuracy_gate, 4)
    utilization_gated = round(utilization * accuracy_gate, 4)
    process_gated = round(process * accuracy_gate, 4)

    # Total score
    score = round(accuracy + efficiency_gated + utilization_gated + process_gated, 4)
    score = max(0.0, min(1.0, score))

    return {
        "score": score,
        "components": {
            "accuracy": accuracy,
            "efficiency": efficiency_gated,
            "utilization": utilization_gated,
            "process": process_gated,
        },
        "accuracy_gate": accuracy_gate,
        "metrics": metrics,
        "task_id": task_id,
        "seed": seed,
        "steps": steps,
    }


def grade_task(
    env: RCTDEnvironment,
    policy: Callable[[RCTDObservation], RCTDAction],
    task_id: str = "medium",
    num_episodes: int = 10,
    base_seed: int = 0,
) -> Dict[str, Any]:
    """Grade a policy across multiple episodes for one task.

    Returns aggregate scores and per-episode breakdowns.
    """
    results = []
    for i in range(num_episodes):
        result = grade_episode(
            env=env,
            policy=policy,
            task_id=task_id,
            seed=base_seed + i,
        )
        results.append(result)

    scores = [r["score"] for r in results]
    success_rate = sum(1 for r in results if r["metrics"].get("success")) / len(results)

    # Failure mode distribution
    failure_modes: Dict[str, int] = {}
    for r in results:
        fm = r["metrics"].get("failure_mode")
        if fm:
            failure_modes[fm] = failure_modes.get(fm, 0) + 1

    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "task_id": task_id,
        "num_episodes": num_episodes,
        "average_score": round(avg_score, 4),
        "min_score": round(min(scores), 4) if scores else 0.0,
        "max_score": round(max(scores), 4) if scores else 0.0,
        "success_rate": round(success_rate, 4),
        "failure_modes": failure_modes,
        "per_episode": results,
    }


def grade_all_tasks(
    env: RCTDEnvironment,
    policy: Callable[[RCTDObservation], RCTDAction],
    num_episodes: int = 10,
    base_seed: int = 0,
) -> Dict[str, Any]:
    """Grade a policy across all three tasks."""
    results = {}
    for task_id in ["easy", "medium", "hard"]:
        results[task_id] = grade_task(
            env=env,
            policy=policy,
            task_id=task_id,
            num_episodes=num_episodes,
            base_seed=base_seed,
        )

    overall = sum(r["average_score"] for r in results.values()) / 3
    return {
        "overall_score": round(overall, 4),
        "tasks": results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Built-in Baseline Policies (for grader testing)
# ═══════════════════════════════════════════════════════════════════════════

import random as stdlib_random


def random_policy(obs: RCTDObservation) -> RCTDAction:
    """Random agent: samples random valid actions."""
    rng = stdlib_random.Random()

    if obs.step_count > 15:
        # Safety: submit after many steps
        hid = rng.choice(obs.active_hypothesis_ids) if obs.active_hypothesis_ids else 0
        return RCTDAction(type="submit_answer", hypothesis_id=hid)

    action_type = rng.choice([
        "read_evidence", "read_evidence",  # Higher weight for reading
        "consult_expert",
        "discard_hypothesis",
        "submit_answer",
    ])

    if action_type == "read_evidence":
        # Pick an un-read evidence
        read_ids = {e.evidence_id for e in obs.revealed_evidence}
        unread = [i for i in range(obs.total_evidence_count) if i not in read_ids]
        if unread:
            return RCTDAction(type="read_evidence", evidence_id=rng.choice(unread))
        else:
            # All read, submit
            hid = rng.choice(obs.active_hypothesis_ids) if obs.active_hypothesis_ids else 0
            return RCTDAction(type="submit_answer", hypothesis_id=hid)

    elif action_type == "consult_expert":
        if obs.active_hypothesis_ids and obs.budget_remaining >= 2:
            return RCTDAction(
                type="consult_expert",
                hypothesis_id=rng.choice(obs.active_hypothesis_ids),
            )
        else:
            hid = rng.choice(obs.active_hypothesis_ids) if obs.active_hypothesis_ids else 0
            return RCTDAction(type="submit_answer", hypothesis_id=hid)

    elif action_type == "discard_hypothesis":
        if len(obs.active_hypothesis_ids) > 1:
            return RCTDAction(
                type="discard_hypothesis",
                hypothesis_id=rng.choice(obs.active_hypothesis_ids),
            )
        else:
            hid = obs.active_hypothesis_ids[0] if obs.active_hypothesis_ids else 0
            return RCTDAction(type="submit_answer", hypothesis_id=hid)

    else:  # submit_answer
        hid = rng.choice(obs.active_hypothesis_ids) if obs.active_hypothesis_ids else 0
        return RCTDAction(type="submit_answer", hypothesis_id=hid)


def heuristic_policy(obs: RCTDObservation) -> RCTDAction:
    """Smart heuristic agent: reads, verifies suspicious evidence, eliminates, submits.

    Strategy:
      1. Read all affordable evidence
      2. Verify (run_experiment) any evidence with low confidence or contradictions
      3. Consult expert when top-2 hypotheses are close
      4. Discard least-supported hypotheses
      5. Submit the most-supported hypothesis
    """
    read_ids = {e.evidence_id for e in obs.revealed_evidence}
    verified_ids = {e.evidence_id for e in obs.revealed_evidence if e.verified}
    unread = [i for i in range(obs.total_evidence_count) if i not in read_ids]

    # Phase 1: Read evidence (keep enough budget for 1 verify + 1 expert)
    min_reserve = 5  # 3 for experiment + 2 for expert
    if unread and obs.budget_remaining >= min_reserve + 1:
        return RCTDAction(type="read_evidence", evidence_id=unread[0])

    # Phase 2: Verify suspicious evidence (low confidence or has contradictions)
    if obs.budget_remaining >= 3:
        suspicious = [
            e for e in obs.revealed_evidence
            if not e.verified and (
                e.confidence < 0.65
                or len(e.apparent_contradiction) > 0
            )
        ]
        if suspicious:
            # Verify the most suspicious item first
            target = min(suspicious, key=lambda e: e.confidence)
            return RCTDAction(type="run_experiment", evidence_id=target.evidence_id)

    # Compute net support scores
    support_counts: Dict[int, float] = {h: 0.0 for h in obs.active_hypothesis_ids}
    for ev in obs.revealed_evidence:
        weight = 2.0 if ev.verified else 1.0
        for h in ev.apparent_support:
            if h in support_counts:
                support_counts[h] += weight * ev.confidence
        for h in ev.apparent_contradiction:
            if h in support_counts:
                support_counts[h] -= weight * ev.confidence

    for hint in obs.expert_hints:
        if hint.hypothesis_id in support_counts:
            support_counts[hint.hypothesis_id] += hint.estimated_probability

    if not support_counts:
        hid = obs.active_hypothesis_ids[0] if obs.active_hypothesis_ids else 0
        return RCTDAction(type="submit_answer", hypothesis_id=hid)

    # Phase 3: Consult expert when top-2 are close or when we haven't verified
    sorted_hyps = sorted(support_counts.items(), key=lambda x: x[1], reverse=True)
    consulted_ids = {h.hypothesis_id for h in obs.expert_hints}
    if (
        len(sorted_hyps) >= 2
        and obs.budget_remaining >= 2
        and sorted_hyps[0][1] - sorted_hyps[1][1] < 1.5
        and sorted_hyps[0][0] not in consulted_ids
    ):
        return RCTDAction(type="consult_expert", hypothesis_id=sorted_hyps[0][0])

    # Phase 4: Discard weakest
    if len(obs.active_hypothesis_ids) > 2:
        weakest = min(support_counts, key=support_counts.get)
        return RCTDAction(type="discard_hypothesis", hypothesis_id=weakest)

    # Phase 5: Submit strongest
    strongest = max(support_counts, key=support_counts.get)
    return RCTDAction(type="submit_answer", hypothesis_id=strongest)


