"""
RCTD Environment — Client.

Inherits from openenv.core.EnvClient for WebSocket-based communication.
Also provides a convenience local-mode wrapper for script/notebook usage.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from openenv.core import EnvClient
from openenv.core.env_client import StepResult

from .models import (
    EvidenceItem,
    ExpertHint,
    RCTDAction,
    RCTDObservation,
    RCTDState,
)


class RCTDEnv(EnvClient[RCTDAction, RCTDObservation, RCTDState]):
    """WebSocket client for interacting with a remote RCTD environment.

    Inherits from openenv.core.EnvClient — handles WebSocket connection,
    session management, Docker launch, and lifecycle automatically.

    Usage (async — preferred for remote servers):
    ```python
    from rctd_env import RCTDEnv

    async with RCTDEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(seed=42, task_id="medium")
        while not result.done:
            action = decide(result.observation)
            result = await env.step(action)
    ```

    Usage (sync wrapper — for scripts):
    ```python
    env = RCTDEnv(base_url="http://localhost:8000").sync()
    with env:
        result = env.reset(seed=42, task_id="medium")
        result = env.step(RCTDAction(type="read_evidence", evidence_id=0))
    ```

    For local usage without a server, use RCTDLocalEnv below.
    """

    def _step_payload(self, action: RCTDAction) -> Dict[str, Any]:
        """Convert an RCTDAction to the JSON payload expected by the server."""
        payload: Dict[str, Any] = {"type": action.type}
        if action.evidence_id is not None:
            payload["evidence_id"] = action.evidence_id
        if action.hypothesis_id is not None:
            payload["hypothesis_id"] = action.hypothesis_id
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[RCTDObservation]:
        """Convert a server JSON response to a StepResult wrapping RCTDObservation."""
        obs_data = payload.get("observation", payload)

        revealed = [
            EvidenceItem(**e) if isinstance(e, dict) else e
            for e in obs_data.get("revealed_evidence", [])
        ]
        hints = [
            ExpertHint(**h) if isinstance(h, dict) else h
            for h in obs_data.get("expert_hints", [])
        ]

        obs = RCTDObservation(
            done=obs_data.get("done", False),
            reward=obs_data.get("reward"),
            hypotheses=obs_data.get("hypotheses", []),
            active_hypothesis_ids=obs_data.get("active_hypothesis_ids", []),
            revealed_evidence=revealed,
            expert_hints=hints,
            budget_remaining=obs_data.get("budget_remaining", 0),
            total_evidence_count=obs_data.get("total_evidence_count", 0),
            step_count=obs_data.get("step_count", 0),
            action_history=obs_data.get("action_history", []),
            message=obs_data.get("message", ""),
            metrics=obs_data.get("metrics"),
        )

        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=obs.done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> RCTDState:
        """Convert a server JSON response to an RCTDState."""
        return RCTDState(**payload)


class RCTDLocalEnv:
    """Local-only client for direct Python usage (no server needed).

    Usage:
    ```python
    from rctd_env import RCTDLocalEnv, RCTDAction

    env = RCTDLocalEnv()
    obs = env.reset(task_id="medium", seed=42)
    obs = env.step(RCTDAction(type="read_evidence", evidence_id=0))
    ```
    """

    def __init__(self) -> None:
        from .server.environment import RCTDEnvironment
        self._env = RCTDEnvironment()

    def reset(
        self,
        seed: Optional[int] = None,
        task_id: str = "medium",
        episode_id: Optional[str] = None,
    ) -> RCTDObservation:
        """Start a new episode."""
        return self._env.reset(seed=seed, episode_id=episode_id, task_id=task_id)

    def step(self, action: RCTDAction) -> RCTDObservation:
        """Take an action in the environment."""
        return self._env.step(action)

    @property
    def state(self) -> RCTDState:
        """Get current episode metadata."""
        return self._env.state
