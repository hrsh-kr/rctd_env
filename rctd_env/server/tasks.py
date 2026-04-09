"""
RCTD Environment — Task definitions.

Three difficulty levels with clear progression:
  easy   → Introductory (random agent can stumble into correct answer)
  medium → Standard LLM challenge (requires evidence evaluation)
  hard   → Expert (requires deep reasoning, contradiction tracking, budget optimization)
"""

from __future__ import annotations

from typing import Any, Dict, List


# ═══════════════════════════════════════════════════════════════════════════
# Task Definitions
# ═══════════════════════════════════════════════════════════════════════════

TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "id": "easy",
        "name": "Introductory Investigation",
        "description": (
            "A straightforward investigation with 3 hypotheses, low noise, "
            "and a generous budget. Evidence mostly points clearly toward the "
            "truth. Designed as a baseline for random and greedy agents."
        ),
        "difficulty": "easy",
        "num_hypotheses": 3,
        "num_evidence": 6,
        "noise_level": 0.1,
        "budget": 20,
        "expected_success_rate": {
            "random": "~35%",
            "heuristic": "~70%",
            "llm_baseline": "~85%",
        },
        "action_schema": {
            "type": {
                "type": "string",
                "enum": [
                    "read_evidence",
                    "run_experiment",
                    "consult_expert",
                    "discard_hypothesis",
                    "submit_answer",
                ],
                "description": "The type of action to take",
            },
            "evidence_id": {
                "type": "integer",
                "description": "Target evidence index (0-5)",
                "required_for": ["read_evidence", "run_experiment"],
            },
            "hypothesis_id": {
                "type": "integer",
                "description": "Target hypothesis index (0-2)",
                "required_for": [
                    "consult_expert",
                    "discard_hypothesis",
                    "submit_answer",
                ],
            },
        },
    },
    "medium": {
        "id": "medium",
        "name": "Standard Investigation",
        "description": (
            "A moderately challenging investigation with 4 hypotheses and "
            "meaningful noise. Some evidence is unreliable and may mislead. "
            "Agents must evaluate evidence quality and manage their budget "
            "to identify the correct hypothesis."
        ),
        "difficulty": "medium",
        "num_hypotheses": 4,
        "num_evidence": 8,
        "noise_level": 0.3,
        "budget": 12,
        "expected_success_rate": {
            "random": "~25%",
            "heuristic": "~50%",
            "llm_baseline": "~65%",
        },
        "action_schema": {
            "type": {
                "type": "string",
                "enum": [
                    "read_evidence",
                    "run_experiment",
                    "consult_expert",
                    "discard_hypothesis",
                    "submit_answer",
                ],
                "description": "The type of action to take",
            },
            "evidence_id": {
                "type": "integer",
                "description": "Target evidence index (0-7)",
                "required_for": ["read_evidence", "run_experiment"],
            },
            "hypothesis_id": {
                "type": "integer",
                "description": "Target hypothesis index (0-3)",
                "required_for": [
                    "consult_expert",
                    "discard_hypothesis",
                    "submit_answer",
                ],
            },
        },
    },
    "hard": {
        "id": "hard",
        "name": "Expert Investigation",
        "description": (
            "The most challenging investigation with 5 hypotheses, high noise, "
            "and a tight budget. Evidence frequently misleads, multiple hypotheses "
            "have strong apparent support, and the agent must carefully track "
            "contradictions, verify suspicious evidence, and optimize budget "
            "allocation under time pressure."
        ),
        "difficulty": "hard",
        "num_hypotheses": 5,
        "num_evidence": 10,
        "noise_level": 0.50,
        "budget": 8,
        "expected_success_rate": {
            "random": "~20%",
            "heuristic": "~35%",
            "llm_baseline": "~45%",
        },
        "action_schema": {
            "type": {
                "type": "string",
                "enum": [
                    "read_evidence",
                    "run_experiment",
                    "consult_expert",
                    "discard_hypothesis",
                    "submit_answer",
                ],
                "description": "The type of action to take",
            },
            "evidence_id": {
                "type": "integer",
                "description": "Target evidence index (0-9)",
                "required_for": ["read_evidence", "run_experiment"],
            },
            "hypothesis_id": {
                "type": "integer",
                "description": "Target hypothesis index (0-4)",
                "required_for": [
                    "consult_expert",
                    "discard_hypothesis",
                    "submit_answer",
                ],
            },
        },
    },
}


def get_task_list() -> List[Dict[str, Any]]:
    """Return summary of all tasks for the /tasks endpoint."""
    return [
        {
            "id": t["id"],
            "name": t["name"],
            "description": t["description"],
            "difficulty": t["difficulty"],
            "action_schema": t["action_schema"],
        }
        for t in TASKS.values()
    ]
