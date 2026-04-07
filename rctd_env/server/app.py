"""
RCTD Environment — FastAPI server.

Uses openenv.core.create_fastapi_app() for standard endpoints (/ws, /reset,
/step, /state, /health, /schema, /docs) and adds custom endpoints for
competition requirements (/tasks, /baseline, /grader).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import Query
from openenv.core import create_fastapi_app

from ..models import RCTDAction, RCTDObservation
from .environment import RCTDEnvironment
from .graders import (
    grade_all_tasks,
    grade_episode,
    heuristic_policy,
    random_policy,
)
from .tasks import TASKS, get_task_list

# ═══════════════════════════════════════════════════════════════════════════
# Create app via OpenEnv framework — gives us /ws, /reset, /step, /state,
# /health, /schema/action, /schema/observation, /docs — all wired up.
# ═══════════════════════════════════════════════════════════════════════════

app = create_fastapi_app(
    env=RCTDEnvironment,
    action_cls=RCTDAction,
    observation_cls=RCTDObservation,
    max_concurrent_envs=100,
)


# ═══════════════════════════════════════════════════════════════════════════
# Competition-Required Custom Endpoints
# ═══════════════════════════════════════════════════════════════════════════


@app.get("/tasks")
async def tasks():
    """Return list of tasks and the action schema."""
    return {
        "tasks": get_task_list(),
        "num_tasks": len(TASKS),
        "action_types": [
            {
                "name": "read_evidence",
                "cost": 1,
                "description": "Read an evidence item (may be noisy)",
                "requires": "evidence_id",
            },
            {
                "name": "run_experiment",
                "cost": 3,
                "description": "Deep-verify evidence (reveals ground truth, bypasses noise)",
                "requires": "evidence_id",
            },
            {
                "name": "consult_expert",
                "cost": 2,
                "description": "Get probabilistic hint about a hypothesis",
                "requires": "hypothesis_id",
            },
            {
                "name": "discard_hypothesis",
                "cost": 0,
                "description": "Remove a hypothesis from consideration",
                "requires": "hypothesis_id",
            },
            {
                "name": "submit_answer",
                "cost": 0,
                "description": "Final answer (terminal action)",
                "requires": "hypothesis_id",
            },
        ],
    }


@app.get("/baseline")
async def baseline(
    num_episodes: int = Query(default=10, ge=1, le=100),
):
    """Run baseline agents and return scores for all 3 tasks.

    This endpoint runs both a random and heuristic agent to produce
    reproducible baseline scores.
    """
    env = RCTDEnvironment()
    results = {}

    for agent_name, policy in [("random", random_policy), ("heuristic", heuristic_policy)]:
        agent_results = grade_all_tasks(
            env=env,
            policy=policy,
            num_episodes=num_episodes,
            base_seed=42,
        )
        results[agent_name] = {
            "overall_score": agent_results["overall_score"],
            "tasks": {
                task_id: {
                    "average_score": task_data["average_score"],
                    "success_rate": task_data["success_rate"],
                    "failure_modes": task_data["failure_modes"],
                }
                for task_id, task_data in agent_results["tasks"].items()
            },
        }

    return {
        "baseline_scores": results,
        "num_episodes_per_task": num_episodes,
        "seed": 42,
    }


@app.get("/grader")
async def grader(
    task_id: str = Query(default="medium", pattern="^(easy|medium|hard)$"),
    seed: int = Query(default=0),
    agent: str = Query(default="heuristic", pattern="^(random|heuristic)$"),
):
    """Run the grader for a single episode and return the score.

    Returns a score between 0.0 and 1.0 with component breakdown.
    """
    env = RCTDEnvironment()
    policy = random_policy if agent == "random" else heuristic_policy

    result = grade_episode(
        env=env,
        policy=policy,
        task_id=task_id,
        seed=seed,
    )

    return {
        "score": result["score"],
        "components": result["components"],
        "task_id": result["task_id"],
        "seed": result["seed"],
        "steps": result["steps"],
        "success": result["metrics"].get("success", False),
        "failure_mode": result["metrics"].get("failure_mode"),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Server Entry Point
# ═══════════════════════════════════════════════════════════════════════════


def main():
    """Start the uvicorn server. Used as the `server` script entry point."""
    import os
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    workers = int(os.environ.get("WORKERS", "4"))

    uvicorn.run(
        "rctd_env.server.app:app",
        host=host,
        port=port,
        workers=workers,
    )


if __name__ == "__main__":
    main()
